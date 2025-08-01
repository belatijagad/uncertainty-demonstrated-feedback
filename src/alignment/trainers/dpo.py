import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from typing import Optional, Literal
from omegaconf import DictConfig
from alignment.callbacks.callbacks import TrainerCallback

from alignment.trainers.base import BaseTrainer

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
    