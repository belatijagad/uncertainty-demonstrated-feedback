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
from tqdm import tqdm
from typing import Optional, Literal
from collections import defaultdict
from omegaconf import DictConfig
from abc import ABC, abstractmethod

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, get_linear_schedule_with_warmup
from huggingface_hub import upload_folder

from alignment.callbacks import TrainerCallback

logger = logging.getLogger(__name__)

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
        
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_train_steps
        )
        self.step_counter = 0
        
    @abstractmethod
    def _get_batch_metrics(self, batch: dict, train_eval: Literal["train", "eval"]) -> tuple[torch.Tensor, dict]:
        """
        Computes the loss and a dictionary of metrics for a given batch
        """
        pass
    
    def train(self) -> None:
        """Main training loop."""
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        pbar = tqdm(total=num_train_steps, desc="Training", 
                    bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")
        data_iterator = iter(self.train_dataloader)
        
        grad_acc_steps = self.config.get("gradient_accumulation_steps", 1)
        
        logger.info(f"=>> Running Training {self.model.config.name_or_path} for {self.config.epochs} epochs.")

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step_counter += 1

                pbar.update(grad_acc_steps)
                tqdm_metrics = {k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) 
                            for k, v in metrics.items()}
                pbar.set_postfix(tqdm_metrics)
                
            for cb in self.callbacks:
                cb.on_step_end(step=self.step_counter, metrics=metrics, trainer=self)

            if (i + 1) % self.config.logging_steps == 0:
                metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                        for k, v in metrics.items()])
                report = f"Step {self.step_counter:>6} | {metrics_str}"
                logger.info(report)

            if (i + 1) % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Step {self.step_counter} Evaluation: {eval_metrics}")

            if (i + 1) % self.config.save_steps == 0:
                save_path = f"checkpoint-{self.step_counter}"
                self.save(output_dir=save_path)
                logger.info(f"Saved checkpoint to {save_path}")
        
        for cb in self.callbacks: cb.on_train_end(trainer=self)
        pbar.close()
        logger.info("=>> Training complete.")

    def evaluate(self) -> dict[str, torch.Tensor]:
        self.model.eval()
        all_metrics = defaultdict(list)

        with torch.inference_mode():
            eval_pbar = tqdm(self.eval_dataloader, desc="Evaluating", 
                            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")
            for batch in eval_pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                _, metrics = self._get_batch_metrics(batch, "eval")
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                tqdm_metrics = {k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) 
                            for k, v in metrics.items()}
                eval_pbar.set_postfix(tqdm_metrics)

        final_metrics = {key: torch.tensor(values).mean().item() for key, values in all_metrics.items()}
        
        eval_metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                    for k, v in final_metrics.items()])
        logger.info(f"Evaluation Results | {eval_metrics_str}")
        
        self.model.train()
        return final_metrics

    def save(self, output_dir) -> None:
        """Saves the model, optimizer, and scheduler states."""
        logger.info(f"Saving {self.model.config.name_or_path} to {output_dir}.")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.optimizer.state_dict(), f"{output_dir}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{output_dir}/scheduler.pt")
        logger.info("Saving complete.")

    def push_to_hub(
        self,
        folder_path: str,
        commit_message: Optional[str] = "End of training",
        token: Optional[str] = None,
    ) -> str:
        logger.info("Uploading model to huggingface hub repository {}...")
        return upload_folder(
            repo_id=self.config.repo_id,
            folder_path=folder_path,
            commit_message=commit_message,
            token=token
        )
        
    @staticmethod
    def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, label_pad_token_id: int = -100) -> torch.FloatTensor:
        """Computes the log probabilities of the given labels under the given logits."""
        if logits.shape[:-1] != labels.shape:
            logger.error(f"Logits have shape {logits.shape[:-1]} while labels have shape {labels.shape}.")
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != label_pad_token_id)

        # Set pad token labels to 0 to avoid index errors; the loss is masked anyway
        labels[labels == label_pad_token_id] = 0
        
        # Check if the first token is a pad token.
        if (labels[:, 0] == label_pad_token_id).any():
            logger.warning("Found pad token at the beginning of the sequence. This might lead to unexpected behavior.")

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        return (per_token_logps * loss_mask).sum(-1)
    