import os
import logging
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv

import wandb
import hydra
from typing import Any, Union
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from alignment.trainers import DITTOTrainer
from alignment.callbacks import ResampleCallback, LoggingCallback, WandbCallback, LoraCallback
from alignment.collators import DITTODataCollator
from alignment.utils import seed_everything, batched_generate

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)

def process_dataset(dataset: DatasetDict, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """
    Uses only the 'positive' split and divides it into a train and eval set
    by randomly sampling N data points for evaluation.
    """
    logger.info("Using only the 'positive' split for both training and evaluation.")
    positive_data = dataset['positive']

    num_eval_samples = config.get("num_eval_samples", 2)

    if len(positive_data) <= num_eval_samples:
        raise ValueError(
            f"The 'positive' split size ({len(positive_data)}) is too small for the requested "
            f"number of evaluation samples ({num_eval_samples}). Please use a smaller N."
        )

    shuffled_data = positive_data.shuffle(seed=config.get("seed"))
    
    eval_dataset = shuffled_data.select(range(num_eval_samples))
    train_dataset = shuffled_data.select(range(num_eval_samples, len(shuffled_data)))
    
    logger.info(f"Split the 'positive' data into: train={len(train_dataset)}, eval={len(eval_dataset)}")
            
    return train_dataset, eval_dataset

def generate_rejected_responses(
    dataset: Dataset,
    model: Union[AutoModelForCausalLM | PeftModel | LLM],
    tokenizer: AutoTokenizer,
    config: DictConfig,
) -> Dataset:
    """
    Generates rejected responses using a pipeline and KeyDataset for efficient,
    streaming batch processing.
    """
    logger.info("Generating rejected responses...")

    generated_texts = []

    generations = batched_generate(
        dataset["prompt"],
        max_new_tokens=config.dataset.max_length // 2,
        model=model,
        tokenizer=tokenizer,
        device=config.model.device,
        num_return_sequences=1,
        do_sample=False,
        disable_peft_adapter=config.model.get("disable_adapter_during_eval", True),
        adapter_name="ditto",
    )

    generated_texts = [
        text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token
        for texts in generations
        for text in texts[:1]
    ]
            
    if "rejected" in dataset.column_names:
        dataset = dataset.remove_columns("rejected")
    updated_dataset = dataset.add_column("rejected", generated_texts)

    logger.info(f"Generated {len(updated_dataset)} rejected responses")
    return updated_dataset

@hydra.main(version_base=None, config_path="../configs/ditto", config_name="sanity-check")
def main(config: DictConfig):
    load_dotenv()
    login(token=os.environ.get("HUGGINGFACE_API_KEY"))
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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path, 
        dtype=torch.bfloat16 if config.model.use_bf16 else torch.float32, 
        attn_implementation=config.model.get("attn_implementation", "sdpa"))
    logger.info("Model loaded successfully.")

    lora_config_dict = OmegaConf.to_container(config.lora, resolve=True)
    lora_config = LoraConfig(**lora_config_dict, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ditto")
    model.set_adapter("ditto")
    lora_adapter_path = Path(__file__).parent.parent / "src/alignment/temp"
    lora_adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_adapter_path))

    try:
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

        train_dataset, eval_dataset = process_dataset(full_dataset, config.dataset)
        eval_dataset = generate_rejected_responses(eval_dataset, model, tokenizer, config)
            
        data_collator = DITTODataCollator(
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            max_length=config.dataset.max_length,
            max_prompt_length=config.dataset.max_length // 2,
            model=model if not config.trainer.use_vllm else llm,
            lora_adapter_path=lora_adapter_path / "ditto",
            lora_config=config.lora,
            **config.resample
        )

        eval_collator = DITTODataCollator(
            tokenizer=tokenizer,
            train_dataset=eval_dataset,
            max_length=config.dataset.max_length,
            max_prompt_length=config.dataset.max_length // 2,
            model=model if not config.trainer.use_vllm else llm,
            lora_adapter_path=lora_adapter_path / "ditto",
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
            LoraCallback(model=model),
        ])

        trainer = DITTOTrainer(
            model=model,
            vllm_model=llm,
            adapter_name="ditto",
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
    finally:
        if dist.is_initialized():
            logger.info("Destroying process group...")
            dist.destroy_process_group()
    
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    