import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.trainers import DPOTrainer
from alignment.collators import OfflineDPODataCollator
from alignment.callbacks import LoggingCallback, WandbCallback
from alignment.utils import seed_everything, process_data

logger = logging.getLogger(__name__)

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

    torch_dtype = torch.bfloat16 if config.model.use_bf16 else torch.float32
    
    policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    logger.info("Policy loaded successfully.")

    if config.enable_lora:
        lora_config = LoraConfig(**config.lora)
        policy = get_peft_model(policy, lora_config, adapter_name="dpo")
        policy.set_adapter("dpo")

    ref_policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    logger.info("Reference policy loaded successfully.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully.")

    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    logger.info("Dataset loaded successfully.")

    train_dataset, eval_dataset, _ = process_data(full_dataset, config.dataset, seed=config.seed, logger=logger)
        
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
    logging_callback = LoggingCallback(logging_steps=config.trainer.get("logging_steps", 500))
    callbacks.append(logging_callback)
    
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
        wandb_callback = WandbCallback(wandb_run=run)
        callbacks.append(wandb_callback)

    trainer = DPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        config=config.trainer,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        callbacks=callbacks,
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
