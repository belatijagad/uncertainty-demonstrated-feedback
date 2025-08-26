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

import os
import math
import json
import logging
import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Literal

from tqdm import tqdm
from omegaconf import DictConfig
from collections import defaultdict

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, get_linear_schedule_with_warmup
from huggingface_hub import upload_folder
from peft import PeftModel

from alignment.callbacks import TrainerCallback

logger = logging.getLogger(__name__)

@dataclass
class TrainerState:
    epoch: Optional[float] = 0
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    train_batch_size: Optional[int] = None
    num_train_epochs: int = 0
    log_history: list[dict[str, float]] = None
    best_metric: Optional[float] = None
    best_global_step: Optional[int] = None
    best_global_checkpoint: Optional[int] = None
    last_metrics: dict[str, float] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.last_metrics is None:
            self.last_metrics = {}

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def compute_steps(self, args, max_steps):
        """
        Calculates and stores the absolute value for logging,
        eval, and save steps based on if it was a proportion
        or not.
        """
        for step_kind in ("logging", "eval", "save"):
            num_steps = getattr(args, f"{step_kind}_steps")
            if num_steps is not None:
                if num_steps < 1:
                    num_steps = math.ceil(max_steps * num_steps)
                setattr(self, f"{step_kind}_steps", num_steps)

    def init_training_references(self, trainer, max_steps, num_train_epochs, trial):
        self.max_steps = max_steps
        self.num_train_epochs = num_train_epochs
        

@dataclass
class TrainerControl:
    ...

class BaseTrainer(ABC):
    def __init__(self, model: PreTrainedModel, config: DictConfig,
                 tokenizer: PreTrainedTokenizer, train_dataloader: DataLoader, optimizer: Optimizer,
                 eval_dataloader: Optional[DataLoader] = None, callbacks: Optional[list[TrainerCallback]] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.callbacks = callbacks if callbacks is not None else []
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.model.to(self.device)

        self.is_peft_model = isinstance(self.model, PeftModel)
        
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_train_steps
        )
        self.global_step = 0

        if self.is_peft_model:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"LoRA model detected: {trainable_params:,} trainable / {total_params:,} total parameters "
                        f"({trainable_params/total_params:.2%})")
        else:
            logger.info("Standard (non-LoRA) model detected")

        self.state = TrainerState()
        self.control = TrainerControl()
            
    def train(self) -> None:
        """Main training loop."""
        num_train_steps = self.config.epochs * len(self.train_dataloader)
        pbar = tqdm(total=num_train_steps, desc="Training", 
                    bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}")
        data_iterator = iter(self.train_dataloader)
        
        grad_acc_steps = self.config.get("gradient_accumulation_steps", 1)
        
        logger.info(f"=>> Running Training {self.model.config.name_or_path} for {self.config.epochs} epochs.")

        for cb in self.callbacks: 
            cb.on_train_begin(args=None, state=self.state, control=self.control)
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
                self.global_step += 1

                pbar.update(grad_acc_steps)
                tqdm_metrics = {k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) 
                            for k, v in metrics.items()}
                pbar.set_postfix(tqdm_metrics)
                
                self.state.global_step = self.global_step
                self.state.last_metrics = metrics
                
            for cb in self.callbacks:
                cb.on_step_end(args=None, state=self.state, control=self.control)

            if (i + 1) % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                eval_metrics = self.evaluate()

                # TODO: This code is DPO-specific, won't work for SFT and stuffs.
                #       Need to move this part to DPO code somehow without rewriting the whole training loop.
                policy_samples, ref_samples = None, None
                sample_prompts = []

                if self.config.sample_during_eval:
                    policy_samples, ref_samples = self._generate_samples()
                    
                    for batch in self.eval_dataloader:
                        if "prompt_input_ids" in batch:
                            prompts = self.tokenizer.batch_decode(batch["prompt_input_ids"], skip_special_tokens=True)
                            sample_prompts.extend(prompts)
                        if len(sample_prompts) >= len(policy_samples):
                            break

                for cb in self.callbacks:
                    cb.on_eval_end(
                        args=None, 
                        state=self.state, 
                        control=self.control,
                        eval_metrics=eval_metrics,
                        policy_samples=policy_samples,
                        ref_samples=ref_samples,
                        sample_prompts=sample_prompts
                    )

            if (i + 1) % self.config.save_steps == 0:
                save_path = f"checkpoint-{self.global_step}"
                self.save(output_dir=save_path)
                self.push_to_hub(
                    folder_path=save_path,
                    commit_message=f"Step {self.global_step}",
                    token=os.environ.get("HUGGINGFACE_API_KEY", None),
                )
                logger.info(f"Saved checkpoint to {save_path}.")
        
        for cb in self.callbacks: 
            cb.on_train_end(args=None, state=self.state, control=self.control)
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
        
        self.model.train()
        return final_metrics

    def save(self, output_dir: str) -> None:
        """Saves the model, optimizer, and scheduler states."""
        logger.info(f"Saving {self.model.config.name_or_path} to {output_dir}.")
        os.makedirs(output_dir, exist_ok=True)
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
        revision: Optional[str] = None,
    ) -> str:
        logger.info(f"Uploading model to huggingface hub repository {self.config.repo_id}...")
        return upload_folder(
            repo_id=self.config.repo_id,
            folder_path=folder_path,
            commit_message=commit_message,
            token=token,
            revision=revision,
        )
    
    @abstractmethod
    def _get_batch_metrics(self, batch: dict[str, torch.Tensor], train_eval: Literal["train", "eval"]) -> tuple[torch.Tensor, dict[str, float]]:
        pass

    @abstractmethod
    def _generate_samples(self) -> tuple[list[str], list[str]]:
        pass
        
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
    