import random
from typing import Union, Any

import numpy as np
from tqdm import tqdm

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

def seed_everything(seed: int=42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
    
def process_data(full_dataset: Dataset, config: dict[str, Any], seed: int, logger) -> tuple[Dataset, Dataset, Dataset]:
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

    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=seed)
    return dataset_split["train"], dataset_split["val"], dataset_split["test"]
    
def generate_outputs(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataloader: DataLoader, config: dict) -> list[str]:
    model.eval()
    generated_outputs = []
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating outputs")):
            prompt_input_ids = batch["prompt_input_ids"].to(model.device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(model.device)
            
            prompts = tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=True)
            
            generated_ids = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **config.generation,
            )
            
            new_tokens = generated_ids[:, prompt_input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            for prompt, generated_text in zip(prompts, generated_texts, strict=True):
                generated_outputs.append({
                    "prompt": prompt.strip(),
                    "generated_response": generated_text.strip(),
                    "batch_idx": batch_idx
                })
    
    return generated_outputs
