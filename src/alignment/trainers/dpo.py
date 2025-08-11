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

import tqdm
from wandb.sdk.wandb_run import Run
from typing import Optional, Literal
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.trainers.base import BaseTrainer
from alignment.utils import pad_to_length

class DPOTrainer(BaseTrainer):
    """A trainer for Direct Preference Optimization."""
    def __init__(self, policy: PreTrainedModel, ref_policy: PreTrainedModel, config: DictConfig, 
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, optimizer: Optimizer,
                 eval_dataloader: Optional[DataLoader] = None, callbacks: Optional[list[TrainerCallback]] = None,
                 wandb_run: Optional[Run] = None):
        super().__init__(policy, config, tokenizer, train_dataloader, optimizer, eval_dataloader, callbacks, wandb_run)
        
        self.ref_policy = ref_policy
        self.ref_policy.to(self.device)
        self.ref_policy.eval()
        
    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.Tensor, dict]:
        """Computes the DPO loss and metrics for a given batch."""
        policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.model, batch)
        
        with torch.inference_mode():
            ref_chosen_logps, ref_rejected_logps = self._concatenated_forward(self.ref_policy, batch)

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
        
    def _generate_samples(self) -> list[str, str, str]:
        """Generate samples from policy and reference models for wandb logging."""
        self.model.eval()
        
        all_policy_samples, all_reference_samples = [], []

        with torch.inference_mode():
            eval_pbar = tqdm(self.eval_dataloader, desc="Generating samples", 
                           bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")
            for batch in eval_pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                policy_output = self.model.generate(
                    batch['prompt_input_ids'], 
                    attention_mask=batch['prompt_attention_mask'], 
                    max_length=self.config.max_length, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.pad_token_id
                )
                policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
                all_policy_samples.extend(policy_output_decoded)

                reference_output = self.ref_policy.generate(
                    batch['prompt_input_ids'], 
                    attention_mask=batch['prompt_attention_mask'], 
                    max_length=self.config.max_length, 
                    do_sample=True, 
                    pad_token_id=self.tokenizer.pad_token_id
                )
                reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
                all_reference_samples.extend(reference_output_decoded)
                        
        self.model.train()
                
        return all_policy_samples, all_reference_samples

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
    