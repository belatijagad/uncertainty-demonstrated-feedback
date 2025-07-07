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
from alignment.utils import batched_generate


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
        token_mask = batch["labels"][:, 1:] != -100
        token_counts = token_mask.sum(dim=1).clamp_min(1)
        avg_policy_logps = policy_logps / token_counts

        loss = -avg_policy_logps.mean()

        metrics = {"loss": loss.item()}
        if train_eval == "train":
            metrics["lr"] = self.scheduler.get_last_lr()[0]
        else:  # during eval, add perplexity
            metrics["perplexity"] = torch.exp(loss).item()

        return loss, metrics

    def _generate_samples(self) -> tuple[list[str], list[str]]:
        if self.eval_dataloader is None or not hasattr(self.eval_dataloader, "dataset"):
            return [], []

        eval_dataset = self.eval_dataloader.dataset
        if len(eval_dataset) == 0:
            return [], []

        num_samples = min(int(self.config.get("num_sample_prompts", 3)), len(eval_dataset))

        prompts: list[str] = []
        for idx in range(num_samples):
            example = eval_dataset[idx]
            messages = example.get("messages") if isinstance(example, dict) else None
            if not messages:
                continue

            prompt_messages = messages[:-1] if len(messages) > 1 else messages
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        if not prompts:
            return [], []

        max_length = int(self.config.get("max_length", 1024))
        prompt_token_lens = [
            len(self.tokenizer(prompt, add_special_tokens=True).input_ids) for prompt in prompts
        ]
        max_prompt_len = max(prompt_token_lens, default=max_length)
        max_new_tokens = max(1, max_length - max_prompt_len)

        self.model.eval()
        self.model.set_adapter(self.adapter_name)

        generations = batched_generate(
            prompts,
            max_new_tokens=max_new_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            num_return_sequences=1,
            do_sample=True,
            disable_peft_adapter=False,
            adapter_name=self.adapter_name,
        )

        samples = [texts[0] if texts else "" for texts in generations]

        self.model.train()

        return prompts, samples
