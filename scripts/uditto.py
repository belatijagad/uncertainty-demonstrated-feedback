import os
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

import wandb
import hydra
from typing import Any
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import login, create_repo, HfApi
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.trainers import DITTOTrainer
from alignment.collators import DITTODataCollator
from alignment.collators.uditto import UDITTODataCollator
from alignment.estimators.msp import MSP
from alignment.utils import seed_everything, generate_rejected_responses, build_model_card
from alignment.callbacks import ResampleCallback, LoggingCallback, WandbCallback, LoraCallback

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)

torch.set_float32_matmul_precision('high')

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

def process_dataset(dataset: DatasetDict, config: dict[str, Any], logger) -> tuple[Dataset, Dataset]:
    train_split = dataset["train"] if "train" in dataset else dataset["val"]
    eval_split = dataset["val"] if "val" in dataset else dataset["test"]
        
    author_id = config.get("author_id")
    if author_id is None:
        raise ValueError("Configuration error: `config.dataset.author_id` must be specified.")

    train_samples_per_author = config.get("train_samples_per_author", 100)
    eval_samples_per_author = config.get("eval_samples_per_author", 20)
    
    logger.info(f"Processing dataset: selecting author '{author_id}' with "
                f"up to {train_samples_per_author} train and {eval_samples_per_author} eval samples.")
    
    train_author_data = defaultdict(list)
    for example in train_split:
        author = example["author_id"]
        train_author_data[author].append(example)
    
    eval_author_data = defaultdict(list)
    for example in eval_split:
        author = example["author_id"]
        eval_author_data[author].append(example)
    
    if author_id not in train_author_data:
        raise ValueError(f"Specified author_id '{author_id}' not found in the train split.")
    if author_id not in eval_author_data:
        raise ValueError(f"Specified author_id '{author_id}' not found in the eval split.")
        
    selected_authors = [author_id]
    
    logger.info(f"Selected author: {selected_authors[0]}")
    
    train_data, eval_data = [], []
    
    for author in selected_authors:
        author_train_samples = train_author_data[author]
        random.shuffle(author_train_samples)
        train_data.extend(author_train_samples[:train_samples_per_author])
        
        author_eval_samples = eval_author_data[author]
        random.shuffle(author_eval_samples)
        eval_data.extend(author_eval_samples[:eval_samples_per_author])
    
    random.shuffle(train_data)
    random.shuffle(eval_data)
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    logger.info(f"Final dataset sizes: train={len(train_dataset)}, eval={len(eval_dataset)}")

    return train_dataset, eval_dataset

@hydra.main(version_base=None, config_path="../configs/ditto", config_name="default")
def main(config: DictConfig):
    load_dotenv()
    login(token=os.environ.get("HUGGINGFACE_API_KEY"))
    OmegaConf.resolve(config)
    seed_everything(config.seed)
    
    model_path = config.model.name_or_path
    potential_local_path = os.path.join(get_original_cwd(), model_path)
    adapter_name = config.model.get("adapter_name", "ditto")
    
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
    if config.trainer.get("use_vllm") and VLLM_AVAILABLE:
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

    train_dataset, eval_dataset = process_dataset(full_dataset, config.dataset, logger)
    eval_dataset = generate_rejected_responses(eval_dataset, model, tokenizer, config, logger)
        
    estimator = MSP()

    data_collator = UDITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=model if not config.trainer.use_vllm else llm,
        lora_adapter_path=lora_adapter_path,
        lora_config=config.lora,
        estimator=estimator,
        **config.resample
    )

    eval_collator = UDITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=eval_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=model if not config.trainer.use_vllm else llm,
        lora_adapter_path=lora_adapter_path,
        lora_config=config.lora,
        estimator=estimator,
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
    if config.trainer.push_to_hub:
        if not huggingface_token:
            logger.warning("Push to hub requested but HUGGINGFACE_API_KEY is not set; skipping upload.")
        else:
            repo_id = config.trainer.repo_id
            logger.info(f"Pushing model artifacts to Hugging Face Hub repo '{repo_id}'.")
            create_repo(repo_id=repo_id, token=huggingface_token, exist_ok=True)

            model_card_path = folder_path / "README.md"
            model_card_content = build_model_card(config)
            model_card_path.write_text(model_card_content, encoding="utf-8")

            api = HfApi(token=huggingface_token)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(folder_path),
                commit_message="Upload DITTO model artifacts",
                ignore_patterns=["*.tmp", "wandb/**"],
            )
            logger.info("Successfully pushed the model and model card.")
    
if __name__ == "__main__":
    main()
