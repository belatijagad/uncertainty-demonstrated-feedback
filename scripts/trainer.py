from __future__ import annotations
from typing import Optional, Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from trl.trainer.dpo_trainer import DPOTrainer


class DITTOTrainer(DPOTrainer):
    """Modify the eval_dataloader behavior to not use resampled (cached) generations"""

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
    ) -> DataLoader:

        data_collator = self.data_collator
        if hasattr(data_collator, "set_mode"):
            data_collator.set_mode(training=is_training)

        return super()._get_dataloader(
            dataset=dataset,
            description=description,
            batch_size=batch_size,
            sampler_fn=sampler_fn,
            is_training=is_training,
            dataloader_key=dataloader_key,
        )
