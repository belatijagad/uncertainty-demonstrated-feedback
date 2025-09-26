import os
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

import wandb
import hydra
from tqdm import tqdm
from typing import Any
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict, Dataset, load_dataset
from transformers.pipelines import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.trainers import DPOTrainer
from alignment.callbacks import ResampleCallback, LoggingCallback, WandbCallback
from alignment.collators import DITTODataCollator
from alignment.utils import seed_everything

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)

def process_dataset(dataset: DatasetDict, config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    train_split = dataset["train"] if "train" in dataset else dataset["val"]
    eval_split = dataset["val"] if "val" in dataset else dataset["test"]
        
    num_authors = config.get("num_authors", 5)
    train_samples_per_author = config.get("train_samples_per_author", 100)
    eval_samples_per_author = config.get("eval_samples_per_author", 20)
    
    logger.info(f"Processing dataset: selecting {num_authors} authors with "
                f"{train_samples_per_author} train and {eval_samples_per_author} eval samples each")
    
    train_author_data = defaultdict(list)
    for example in train_split:
        author = example["author_id"]
        train_author_data[author].append(example)
    
    eval_author_data = defaultdict(list)
    for example in eval_split:
        author = example["author_id"]
        eval_author_data[author].append(example)
    
    train_authors = set(train_author_data.keys())
    eval_authors = set(eval_author_data.keys())
    common_authors = list(train_authors.intersection(eval_authors))
    
    random.shuffle(common_authors)
    selected_authors = common_authors[:num_authors]
    
    logger.info(f"Selected authors: {selected_authors}")
    
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

def generate_rejected_responses(
    dataset: Dataset,
    ref_policy: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: DictConfig
) -> Dataset:
    """
    Generates rejected responses using a pipeline and KeyDataset for efficient,
    streaming batch processing.
    """
    logger.info("Generating rejected responses...")

    generator = pipeline(
        "text-generation",
        model=ref_policy,
        tokenizer=tokenizer,
        return_full_text=False,
        # TODO: temporary fix, changing to "mps" will give `probability tensor contains either `inf`, `nan` or element < 0`
        device="cuda",
    )
    
    prompt_dataset = KeyDataset(dataset, "prompt")
    generated_texts = []

    for response_list in tqdm(generator(prompt_dataset,
                                        batch_size=config.dataset.get("batch_size", 8),
                                        max_new_tokens=config.dataset.max_length // 2,
                                        do_sample=False,
                                        # Since sampling is not used, temperature and top_p is useless
                                        # temperature=0.8,
                                        # top_p=0.9,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,),
                              total=len(dataset), desc="Generating rejected responses",
                              leave=False):
        
        text = response_list[0]['generated_text']
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        generated_texts.append(text)
        
    if "rejected" in dataset.column_names:
        dataset = dataset.remove_columns("rejected")
    updated_dataset = dataset.add_column("rejected", generated_texts)

    logger.info(f"Generated {len(updated_dataset)} rejected responses")
    return updated_dataset

@hydra.main(version_base=None, config_path="../configs/ditto", config_name="default")
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
        lora_config_dict = OmegaConf.to_container(config.lora, resolve=True)
        lora_config = LoraConfig(**lora_config_dict, task_type=TaskType.CAUSAL_LM)
        policy = get_peft_model(policy, lora_config, adapter_name="ditto")
        policy.set_adapter("ditto")

    ref_policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    logger.info("Reference policy loaded successfully.")

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
    eval_dataset = generate_rejected_responses(eval_dataset, ref_policy, tokenizer, config)
        
    data_collator = DITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=policy,
        **config.resample
    )

    eval_collator = DITTODataCollator(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
        model=policy,
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
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=False, 
        collate_fn=eval_collator, 
        num_workers=config.dataset.num_workers
    )
    
    optimizer = AdamW(policy.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    
    callbacks.extend([
        ResampleCallback(collator=data_collator, model=policy, resample_rate=config.resample.get("resample_rate", 20)), 
        LoggingCallback(logging_steps=config.trainer.get("logging_steps", 500))
    ])

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
