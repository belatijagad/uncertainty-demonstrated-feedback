from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase
from trl.trainer.dpo_trainer import DPOTrainer


class DITTOTrainer(DPOTrainer):
    """DPOTrainer variant that keeps raw text metadata for the DITTO collator."""

    @staticmethod
    def tokenize_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        tokenized = DPOTrainer.tokenize_row(
            features,
            processing_class,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            add_special_tokens=add_special_tokens,
        )

        # Preserve metadata so the DITTO collator can build new preference pairs.
        if "example_id" in features:
            tokenized["example_id"] = features["example_id"]
        if "prompt" in features:
            tokenized["ditto_prompt"] = features["prompt"]
        if "chosen" in features:
            tokenized["ditto_chosen"] = features["chosen"]

        return tokenized
