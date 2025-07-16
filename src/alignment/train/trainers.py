# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang 2024
# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
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

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

from tqdm import tqdm
from typing import Optional, Literal
from collections import defaultdict
from omegaconf import DictConfig
from alignment.train.callbacks import TrainerCallback

from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model: PreTrainedModel, config: DictConfig,
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader,
                 optimizer: Optimizer, callbacks: Optional[list[TrainerCallback]] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.callbacks = callbacks if callbacks is not None else []
        
        self.device= 'cuda' if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.scheduler.warmup_steps,
            num_training_steps=num_train_steps
        )
        self.step_counter = 0
        
    @abstractmethod
    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.tensor, dict]:
        """
        Computes the loss and a dictionary of metrics for a given batch
        """
        pass
    
    def train(self):
        """Main training loop."""
        print(f"Starting {self.__class__.__name__} training...")
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        pbar = tqdm(total=num_train_steps, desc="Training Steps")
        data_iterator = iter(self.train_dataloader)
        
        grad_acc_steps = self.config.trainer.get("gradient_accumulation_steps", 1)

        for cb in self.callbacks: cb.on_train_begin(trainer=self)
        self.optimizer.zero_grad()

        for i in range(num_train_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss, metrics = self._get_batch_metrics(batch, "train")
            loss = loss / grad_acc_steps
            loss.backward()

            if (i + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step_counter += 1

                pbar.update(grad_acc_steps)
                pbar.set_postfix({"loss": metrics.get("loss", 0.0)})
                
                for cb in self.callbacks:
                    cb.on_step_end(step=self.step_counter, metrics=metrics, trainer=self)
        
        for cb in self.callbacks: cb.on_train_end(trainer=self)
        pbar.close()
    
    def evaluate(self) -> dict:
        """Main evaluation loop."""
        print(f"Running {self.__class__.__name__} evaluation...")
        self.model.eval()
        all_metrics = defaultdict(list)

        with torch.inference_mode():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                _, metrics = self._get_batch_metrics(batch, "eval")
                for key, value in metrics.items():
                    all_metrics[key].append(value)

        final_metrics = {key: torch.tensor(values).mean().item() for key, values in all_metrics.items()}
        
        self.model.train()
        print(f"Evaluation finished: {final_metrics}")
        return final_metrics

    def save(self):
        """Saves the model, optimizer, and scheduler states."""
        output_dir = self.config.model.output_path
        print(f"Saving model and training states to {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.optimizer.state_dict(), f"{output_dir}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{output_dir}/scheduler.pt")
        
    @staticmethod
    def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int = -100) -> torch.FloatTensor:
        """Computes the log probabilities of the given labels under the given logits."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id)

        # Set pad token labels to 0 to avoid index errors; the loss is masked anyway
        labels[labels == label_pad_token_id] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * loss_mask).sum(-1)
    
class SFTTrainer(BaseTrainer):
    """A trainer for Supervised Fine-Tuning."""
    def __init__(self, model: PreTrainedModel, config: DictConfig, tokenizer: PreTrainedTokenizer, 
                 train_dataloader: DataLoader, eval_dataloader: DataLoader, optimizer: Optimizer, 
                 callbacks: list[TrainerCallback] = None):
        super().__init__(model, config, tokenizer, train_dataloader, eval_dataloader, optimizer, callbacks)

    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.Tensor, dict]:
        """Computes the SFT loss and metrics for a given batch."""
        policy_logits = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        ).logits

        policy_logps = self._get_batch_logps(
            logits=policy_logits,
            labels=batch['labels'],
        )
        
        loss = -policy_logps.mean()
        
        metrics = {"loss": loss.item()}
        if train_eval == "train":
            metrics["lr"] = self.scheduler.get_last_lr()[0]
        else: # during eval, add perplexity
            metrics["perplexity"] = torch.exp(loss).item()
            
        return loss, metrics

class DPOTrainer(BaseTrainer):
    """A trainer for Direct Preference Optimization."""
    def __init__(self, policy: PreTrainedModel, ref_policy: PreTrainedModel, config: DictConfig, 
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader, 
                 optimizer: Optimizer, callbacks: Optional[list[TrainerCallback]] = None):
        super().__init__(policy, config, tokenizer, train_dataloader, eval_dataloader, optimizer, callbacks)
        
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
        
        def pad(tensor, value):
            return F.pad(tensor, (0, max_length - tensor.shape[1]), mode='constant', value=value)

        return {
            "concatenated_input_ids": torch.cat([
                pad(batch["chosen_input_ids"], self.tokenizer.pad_token_id),
                pad(batch["rejected_input_ids"], self.tokenizer.pad_token_id),
            ], dim=0),
            "concatenated_attention_mask": torch.cat([
                pad(batch["chosen_attention_mask"], 0),
                pad(batch["rejected_attention_mask"], 0),
            ], dim=0),
            "concatenated_labels": torch.cat([
                pad(batch["chosen_labels"], -100),
                pad(batch["rejected_labels"], -100),
            ], dim=0),
        }
    
    def _dpo_loss(self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, 
                  ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the DPO loss and rewards."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        beta = self.config.model.beta
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(beta * logits)
        
        chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        
        return losses.mean(), chosen_rewards, rejected_rewards
    