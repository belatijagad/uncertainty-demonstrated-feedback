# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional, Literal

from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.pipelines import pipeline
from transformers import PreTrainedTokenizer, PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.trainers.base import BaseTrainer
from alignment.utils import pad_to_length


logger = logging.getLogger(__name__)

class DPOTrainer(BaseTrainer):
    """A trainer for Direct Preference Optimization."""
    def __init__(self, model: PreTrainedModel, adapter_name: str, config: DictConfig, 
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, optimizer: Optimizer, device: str,
                 eval_dataloader: Optional[DataLoader] = None, callbacks: Optional[list[TrainerCallback]] = None):
        super().__init__(model, config, adapter_name, tokenizer, train_dataloader, optimizer, device, eval_dataloader, callbacks)
                
    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.Tensor, dict]:
        """Computes the DPO loss and metrics for a given batch."""
        policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.model, batch)
        
        with self.model.disable_adapter(), torch.inference_mode():
            ref_chosen_logps, ref_rejected_logps = self._concatenated_forward(self.model, batch)

        loss, chosen_rewards, rejected_rewards = self._dpo_loss(
            policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}rewards/chosen": chosen_rewards.mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().item(),
            f"{prefix}rewards/accuracies": reward_accuracies.mean().item(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
        }
        if train_eval == "train":
            metrics["lr"] = self.scheduler.get_last_lr()[0]
            
        return loss, metrics
        
    def _generate_samples(self) -> tuple[list[str], list[str]]:
        """Generate samples from policy for wandb logging using KeyDataset."""        
        self.model.eval()

        eval_dataset = self.eval_dataloader.dataset
        prompts = [example["prompt"] for example in eval_dataset]

        max_prompt_len = self.config.get("max_prompt_length", self.config.max_length // 2)
        gen_kwargs = {
            "max_new_tokens": self.config.max_length - max_prompt_len,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "truncation": True,
        }

        policy_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            device=self.device
        )
        all_policy_samples = []
        with torch.inference_mode():
            for prompt in tqdm(prompts, desc="Generating policy samples"):
                policy_out = policy_generator(prompt, **gen_kwargs)
                all_policy_samples.append(policy_out[0]["generated_text"])
                        
        self.model.train()
                
        return prompts, all_policy_samples

    def _concatenated_forward(self, model: PreTrainedModel, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass on a concatenated batch of chosen and rejected examples."""
        concatenated_batch = self._concatenated_input(batch)
        all_logits = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
        )

        len_chosen = batch["chosen_labels"].shape[0]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        del concatenated_batch, all_logits, all_logps

        return chosen_logps, rejected_logps

    def _concatenated_input(self, batch: dict) -> dict:
        """Pads and concatenates the chosen and rejected examples in a batch."""
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        return {
            "concatenated_input_ids": torch.cat([
                pad_to_length(batch["chosen_input_ids"], max_length, self.tokenizer.pad_token_id, 1),
                pad_to_length(batch["rejected_input_ids"], max_length, self.tokenizer.pad_token_id, 1),
            ], dim=0),
            "concatenated_attention_mask": torch.cat([
                pad_to_length(batch["chosen_attention_mask"], max_length, 0, 1),
                pad_to_length(batch["rejected_attention_mask"], max_length, 0, 1),
            ], dim=0),
            "concatenated_labels": torch.cat([
                pad_to_length(batch["chosen_labels"], max_length, -100, 1),
                pad_to_length(batch["rejected_labels"], max_length, -100, 1),
            ], dim=0),
        }
    
    def _dpo_loss(self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, 
                  ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the DPO loss and rewards."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        beta = self.config.beta
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(beta * logits)
        
        chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        
        return losses.mean(), chosen_rewards, rejected_rewards
    