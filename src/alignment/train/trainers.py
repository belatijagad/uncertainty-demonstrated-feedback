# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang 2024
# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from transformers import PreTrainedTokenizer, PreTrainedModel

from tqdm import tqdm
from typing import Optional, Literal
from collections import defaultdict
from omegaconf import DictConfig
from alignment.train.callbacks import TrainerCallback

class SFTTrainer:
    def __init__(self, model: PreTrainedModel, config, tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader, optimizer: Optimizer, callbacks: list[TrainerCallback] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.scheduler.warmup_steps + 1))
        )
        self.step_counter = 0
        
        self.callbacks = callbacks if callbacks is not None else []

    def train(self):
        """Main SFT training loop."""
        print("Starting SFT training...")
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        
        self.model.train()
        pbar = tqdm(total=num_train_steps)
        data_iterator = iter(self.train_dataloader)
        
        for cb in self.callbacks: cb.on_train_begin(trainer=self)

        for _ in range(num_train_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.step_counter += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
            
            train_metrics = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
            for cb in self.callbacks:
                cb.on_step_end(step=self.step_counter, metrics=train_metrics, trainer=self)
                    
        for cb in self.callbacks: cb.on_train_end(trainer=self)
        pbar.close()

    def evaluate(self):
        """Main SFT evaluation loop."""
        print("Running SFT evaluation...")
        self.model.eval()
        all_losses = []

        with torch.inference_mode():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                all_losses.append(loss.item())
        
        avg_loss = sum(all_losses) / len(all_losses)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.model.train()
        print(f"Evaluation finished: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")
        return {"loss": avg_loss, "perplexity": perplexity}

    def save(self):
        """Saves the model using the standard Hugging Face method."""
        print(f"Saving SFT model to {self.config.model.output_path}")
        self.model.save_pretrained(self.config.model.output_path)
        self.tokenizer.save_pretrained(self.config.model.output_path)

class DPOTrainer:
    def __init__(self, policy: PreTrainedModel, ref_policy: PreTrainedModel, config: DictConfig, tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader, optimizer: Optimizer, callbacks: Optional[list[TrainerCallback]] = None):
        self.policy = policy
        self.ref_policy = ref_policy
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.run_dir = config.run_dir
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy.to(self.device)
        self.ref_policy.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.scheduler.warmup_steps + 1)))
        self.step_counter = 0
        self.callbacks = callbacks if callbacks is not None else []
        
    def concatenated_forward(self, model: PreTrainedModel, batch: dict) -> tuple:
        """
        Performs a forward pass on a concatenated batch, with robust error handling.
        """
        concatenated_batch = self.concatenated_input(batch)
        
        all_logits = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # label_pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=-100,
        )

        len_chosen = batch["chosen_labels"].shape[0]
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        return chosen_logps, rejected_logps


    def get_batch_loss_metrics(self, batch: dict, train_eval: Literal["train", "eval"] = "train") -> tuple:
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
        
        if "reference_chosen_logps" in batch:
            ref_chosen_logps = batch["reference_chosen_logps"]
            ref_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.inference_mode():
                ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.ref_policy, batch)

        loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean(),
            f"{prefix}rewards/rejected": rejected_rewards.mean(),
            f"{prefix}rewards/accuracies": reward_accuracies.mean(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean(),
        }
        
        return loss, metrics

    def train(self):
        """The main training loop."""
        print("Starting training...")
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        
        self.policy.train()
        self.ref_policy.eval()
        
        for cb in self.callbacks:
            cb.on_train_begin(trainer=self)

        pbar = tqdm(total=num_train_steps, desc="Training Steps")
        data_iterator = iter(self.train_dataloader)
        
        for _ in range(num_train_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss, metrics = self.get_batch_loss_metrics(batch, train_eval="train")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.trainer.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.step_counter += 1
            pbar.update(1)

            if self.step_counter % self.config.trainer.eval_steps == 0:
                eval_metrics = self.evaluate()
                for cb in self.callbacks:
                    cb.on_eval_end(metrics=eval_metrics, trainer=self)
                                
            for cb in self.callbacks:
                cb.on_step_end(step=self.step_counter, metrics=metrics, trainer=self)
            
            pbar.close()
            
        print("Training finished.")

    def evaluate(self) -> dict:
        print("Running evaluation...")
        self.policy.eval()
        all_metrics = defaultdict(list)

        with torch.inference_mode():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                loss, metrics = self.get_batch_loss_metrics(batch, train_eval="eval")
                
                all_metrics["eval_loss"].append(loss.item())
                for key, value in metrics.items():
                    all_metrics[key].append(value.item())

        final_metrics = {key: torch.tensor(values).mean().item() for key, values in all_metrics.items()}
        
        self.policy.train()
        print(f"Evaluation finished: {final_metrics}")
        return final_metrics

    def save(self):
        """Saves the model using the standard Hugging Face method."""
        print(f"Saving DPO model to {self.config.model.output_path}")
        self.policy.save_pretrained(self.config.model.output_path)
        self.tokenizer.save_pretrained(self.config.model.output_path)

    def concatenated_input(self, batch: dict) -> dict:
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        
        def pad(tensor, value):
            return F.pad(tensor, (0, max_length - tensor.shape[1]), mode='constant', value=value)

        pad_token_id = self.tokenizer.pad_token_id

        concatenated_batch = {
            "concatenated_input_ids": torch.cat([
                pad(batch["chosen_input_ids"], pad_token_id),
                pad(batch["rejected_input_ids"], pad_token_id),
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
        return concatenated_batch
    
    @staticmethod
    def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)
        
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps) -> tuple:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        
        beta = self.config.model.beta
        losses = -F.logsigmoid(beta * logits)
        
        chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        
        return losses.mean(), chosen_rewards, rejected_rewards
    