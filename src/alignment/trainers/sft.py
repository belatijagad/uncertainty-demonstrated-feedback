from pathlib import Path
from typing import Literal, Optional

import torch
from omegaconf import DictConfig
from peft import PeftModelForCausalLM
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from alignment.callbacks import TrainerCallback
from alignment.trainers.base import BaseTrainer


class SFTTrainer(BaseTrainer):
    """A trainer for Supervised Fine-Tuning."""

    def __init__(
        self,
        model: PeftModelForCausalLM,
        adapter_name: str,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        device: str,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        lora_save_path: Optional[Path] = None,
    ) -> None:
        super().__init__(
            model=model,
            adapter_name=adapter_name,
            config=config,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            eval_dataloader=eval_dataloader,
            callbacks=callbacks,
            lora_save_path=lora_save_path,
        )

    def _get_batch_metrics(
        self, batch: dict, train_eval: Literal["train", "eval"]
    ) -> tuple[torch.Tensor, dict]:
        """Computes the SFT loss and metrics for a given batch."""
        policy_logits = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits

        policy_logps = self._get_batch_logps(
            logits=policy_logits,
            labels=batch["labels"],
        )

        loss = -policy_logps.mean()

        metrics = {"loss": loss.item()}
        if train_eval == "train":
            metrics["lr"] = self.scheduler.get_last_lr()[0]
        else:  # during eval, add perplexity
            metrics["perplexity"] = torch.exp(loss).item()

        return loss, metrics
