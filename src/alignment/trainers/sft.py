import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from typing import Literal
from omegaconf import DictConfig
from alignment.callbacks.callbacks import TrainerCallback
from alignment.trainers.base import BaseTrainer

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
