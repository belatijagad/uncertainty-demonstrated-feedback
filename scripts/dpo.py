import os
import logging
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.trainers import DPOTrainer
from alignment.collators import OfflineDPODataCollator
from alignment.utils import seed_everything

logger = logging.getLogger(__name__)

def process_data(full_dataset: Dataset, config: dict[str, Any], seed: int) -> tuple[Dataset, Dataset]:
    subset_size = config["subset_size"]

    logger.info(f"Using a subset of the data: {subset_size}")
    if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
        num_samples = int(len(full_dataset) * subset_size)
        logger.info(f"Using {subset_size:.2%} of the data, resulting in {num_samples} samples.")
    elif isinstance(subset_size, int):
        num_samples = subset_size
        logger.info(f"Using a subset of {num_samples} samples.")
    else:
        logger.error(f"Invalid subset_size: {subset_size}. Must be a float < 1.0 or an int.")
        raise ValueError(f"Invalid subset_size: {subset_size}.")
    
    if num_samples > len(full_dataset):
        raise ValueError(f"Number of samples {num_samples} exceeds dataset size {len(full_dataset)}.")
    full_dataset = full_dataset.shuffle(seed=seed).select(range(num_samples))

    dataset_splited = full_dataset.train_test_split(test_size=0.1, seed=seed)
    return dataset_splited["train"], dataset_splited["test"]

@hydra.main(version_base=None, config_path="../configs/dpo", config_name="default")
def main(config: DictConfig):
    load_dotenv()
    OmegaConf.resolve(config)
    seed_everything(config.seed)
    
    model_path = config.model.name_or_path
    potential_local_path = os.path.join(get_original_cwd(), model_path)
    
    if os.path.isdir(potential_local_path):
        logger.info(f"Loading model from local directory: {potential_local_path}")
        model_load_path = potential_local_path
    else:
        logger.info(f"Loading model from Hugging Face Hub: {model_path}")
        model_load_path = model_path
    
    policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    logger.info("Policy loaded successfully.")

    if config.enable_lora:
        lora_config = LoraConfig(**config.lora)
        policy = get_peft_model(policy, lora_config, adapter_name="dpo")
        policy.set_adapter("dpo")

    ref_policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    logger.info("Reference policy loaded successfully.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully.")

    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    logger.info("Dataset loaded successfully.")

    train_dataset, eval_dataset = process_data(full_dataset, config.dataset, seed=config.seed)
        
    data_collator = OfflineDPODataCollator(
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=True, 
        collate_fn=data_collator, 
        num_workers=config.dataset.num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=False, 
        collate_fn=data_collator, 
        num_workers=config.dataset.num_workers
    )
    
    optimizer = AdamW(policy.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    callbacks = []
    
    run = None
    if config.wandb.enabled:
        run = wandb.init(
            project=config.wandb.project,
            config=config,
            name=config.wandb.name,
            group=config.wandb.group,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
        )

    trainer = DPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        config=config.trainer,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        callbacks=callbacks,
        wandb_run=run,
    )
    
    trainer.train()

    logger.info(f"Start evaluating model {policy.config.name_or_path}.")
    trainer.evaluate()
    logger.info("Evaluation complete.")

    folder_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "model"
    folder_path.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir=folder_path)

    huggingface_token = os.environ.get("HUGGINGFACE_API_KEY", None)
    if config.trainer.push_to_hub and huggingface_token:
        logger.info("Pushing model to hub.")
        trainer.push_to_hub(folder_path=folder_path, commit_message=config.commit_message, token=huggingface_token)
        logger.info("Successfully pushed the model.")
    
if __name__ == "__main__":
    main()
