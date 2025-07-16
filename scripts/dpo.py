import os
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from alignment.train.callbacks import JsonLoggingCallback
from alignment.train.trainers import DPOTrainer
from alignment.data.data_collator import OfflineDPODataCollator
from alignment.utils import seed_everything
from hydra.utils import get_original_cwd

@hydra.main(version_base=None, config_path="../configs", config_name="pythia160m_dpo")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    seed_everything(config.seed)
    
    model_path = config.model.name_or_path
    potential_local_path = os.path.join(get_original_cwd(), model_path)
    
    if os.path.isdir(potential_local_path):
        print(f"Loading model from local directory: {potential_local_path}")
        model_load_path = potential_local_path
    else:
        print(f"Loading model from Hugging Face Hub: {model_path}")
        model_load_path = model_path
    
    policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    
    subset_size = config.dataset.get("subset_size", None)
    if subset_size is not None:
        print(f"Using a subset of the data: {subset_size}")
        if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
            num_samples = int(len(full_dataset) * subset_size)
        elif isinstance(subset_size, int):
            num_samples = subset_size
        else:
            raise ValueError(f"Invalid subset_size: {subset_size}. Must be a float < 1.0 or an int.")
        full_dataset = full_dataset.shuffle(seed=config.seed).select(range(num_samples))

    dataset_splited = full_dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = dataset_splited["train"]
    eval_dataset = dataset_splited["test"]
    
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
    callbacks = [JsonLoggingCallback(log_dir=config.run_dir)]
    
    trainer = DPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        config=config,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    trainer.train()
    trainer.evaluate()
    trainer.save()
    
if __name__ == "__main__":
    main()