import os
import logging
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

import wandb
import hydra
from typing import Any
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.utils import generate_rejected_responses

from vllm import LLM

from alignment.trainers import DITTOTrainer
from alignment.utils import seed_everything
from alignment.collators import DITTODataCollator
from alignment.callbacks import ResampleCallback, LoggingCallback, WandbCallback, LoraCallback

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.CRITICAL)

torch.set_float32_matmul_precision('high')

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

def process_dataset(dataset: DatasetDict, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """
    Uses only the 'negative' split and divides it into a train and eval set
    by randomly sampling N data points for evaluation.
    """
    logger.info("Using only the 'negative' split for both training and evaluation.")
    negative_data = dataset['negative']

    num_eval_samples = config.get("num_eval_samples", 2)

    if len(negative_data) <= num_eval_samples:
        raise ValueError(
            f"The 'negative' split size ({len(negative_data)}) is too small for the requested "
            f"number of evaluation samples ({num_eval_samples}). Please use a smaller N."
        )

    shuffled_data = negative_data.shuffle(seed=config.get("seed"))
    
    eval_dataset = shuffled_data.select(range(num_eval_samples))
    train_dataset = shuffled_data.select(range(num_eval_samples, len(shuffled_data)))
    
    logger.info(f"Split the 'negative' data into: train={len(train_dataset)}, eval={len(eval_dataset)}")

    logger.info("--- Sanity Check: Random Evaluation Sample ---")
    sample = eval_dataset[0]
    if 'prompt' in sample and 'chosen' in sample and 'rejected' in sample:
        prompt_str = sample['prompt']
        chosen_str = sample['chosen']
        rejected_str = sample['rejected']
        
        logger.info(f"Prompt:   {(prompt_str[:250] if prompt_str else 'None')}...")
        logger.info(f"Chosen:   {(chosen_str[:250] if chosen_str else 'None')}...")
        logger.info(f"Rejected: {(rejected_str[:250] if rejected_str else 'None')}...")

    return train_dataset, eval_dataset

@hydra.main(version_base=None, config_path="../configs/ditto", config_name="sanity-check")
def main(config: DictConfig):
    load_dotenv()
    login(token=os.environ.get("HUGGINGFACE_API_KEY"))
    OmegaConf.resolve(config)
    seed_everything(config.seed)
    
    model_path = config.model.name_or_path
    potential_local_path = os.path.join(get_original_cwd(), model_path)
    adapter_name = config.model.adapter_name
    
    if os.path.isdir(potential_local_path):
        logger.info(f"Loading model from local directory: {potential_local_path}")
        model_load_path = potential_local_path
    else:
        logger.info(f"Loading model from Hugging Face Hub: {model_path}")
        model_load_path = model_path
    
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, 
        dtype=torch.bfloat16 if config.model.use_bf16 else torch.float32, 
        attn_implementation=config.model.get("attn_implementation", "sdpa"))
    logger.info("Model loaded successfully.")

    lora_config_dict = OmegaConf.to_container(config.lora, resolve=True)
    lora_config = LoraConfig(**lora_config_dict, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)

    project_root = Path(get_original_cwd())
    lora_adapter_path = project_root / "src" / "alignment" / "temp" / adapter_name
    lora_adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_adapter_path))

    llm = None
    if config.trainer.get("use_vllm"):
        vllm_config = config.trainer.get("vllm")
        llm = LLM(
            model=model.name_or_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=vllm_config.get("gpu_mem_util"),
            max_num_seqs=vllm_config.get("batch_size")
            * vllm_config.get("rescale_batch")
            * vllm_config.get("bootstrap_count"),
            max_model_len=config.get("max_length"),
            seed=42,
            max_num_batched_tokens=4096,
            enable_sleep_mode=vllm_config.get("enable_sleep_mode"),
            logprobs_mode="processed_logprobs",
            enable_lora=True,
        )
        if vllm_config.get("enable_sleep_mode"):
            llm.sleep(level=2)

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully.")

    full_dataset = load_dataset(config.dataset.name_or_path)
    logger.info("Dataset loaded successfully.")

    run = None
    callbacks = []
    if config.wandb.enabled:
        run = wandb.init(
            project=config.wandb.project,
            config=OmegaConf.to_container(config, resolve=True),
            name=config.wandb.name,
            group=config.wandb.group,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
        )
        wandb_callback = WandbCallback(wandb_run=run)
        callbacks.append(wandb_callback)

    train_dataset, eval_dataset = process_dataset(full_dataset, config.dataset)
    eval_dataset = generate_rejected_responses(eval_dataset, model, tokenizer, config, logger)
        
    data_collator = DITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=model if not config.trainer.use_vllm else llm,
        lora_adapter_path=lora_adapter_path,
        lora_config=config.lora,
        **config.resample
    )

    eval_collator = DITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=eval_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=model if not config.trainer.use_vllm else llm,
        lora_adapter_path=lora_adapter_path,
        lora_config=config.lora,
        **config.resample,
        mode="eval",
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
        collate_fn=eval_collator, 
        num_workers=config.dataset.num_workers
    )
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    
    callbacks.extend([
        ResampleCallback(collator=data_collator, model=model, resample_rate=config.get("resample_rate", 20)), 
        LoggingCallback(logging_steps=config.trainer.get("logging_steps", 500)),
        LoraCallback(model=model, save_path=lora_adapter_path, adapter_name=adapter_name),
    ])

    trainer = DITTOTrainer(
        model=model,
        vllm_model=llm,
        adapter_name=adapter_name,
        lora_save_path=lora_adapter_path,
        config=config.trainer,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        device=config.model.get("device", "cpu"),
        callbacks=callbacks,
    )
    
    trainer.train()

    logger.info(f"Start evaluating model {model.config.name_or_path}.")
    trainer.evaluate()
    logger.info("Evaluation complete.")

    wandb.finish()

    folder_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "model"
    folder_path.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir=folder_path)

    huggingface_token = os.environ.get("HUGGINGFACE_API_KEY", None)
    if config.trainer.push_to_hub and huggingface_token:
        logger.info("Pushing model to hub.")
        trainer.push_to_hub(folder_path=folder_path, commit_message=config.commit_message, token=huggingface_token)
        logger.info("Successfully pushed the model.")
    
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    