import random
from contextlib import nullcontext
from typing import Union, Any, Iterable, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.pipelines import pipeline

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


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
    
def batched_generate(
    prompts: Iterable[str],
    max_new_tokens: int,
    model: Union[PreTrainedModel | LLM],
    tokenizer: Optional[PreTrainedTokenizer],
    device: Optional[str],
    lora_request: Optional[LoRARequest] = None,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    disable_peft_adapter: bool = False,
    adapter_name: Optional[str] = None,
) -> list[list[str]]:
    prompts = list(prompts)

    if isinstance(model, LLM):
        model.wake_up()
        params = SamplingParams(
            n=num_return_sequences,
            max_tokens=max_new_tokens,
            temperature=1.0 if do_sample else 0.0,
        )
        outputs = model.generate(prompts, params, lora_request=lora_request)
        model.sleep(level=2)
        return [[gen.text for gen in out.outputs] for out in outputs]

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=device,
    )

    if disable_peft_adapter and hasattr(model, "disable_adapter") and adapter_name:
        model.set_adapter(adapter_name)
        adapter_ctx = model.disable_adapter()
    else:
        adapter_ctx = nullcontext()
        
    with adapter_ctx, torch.inference_mode():
        results = generator(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [[gen["generated_text"] for gen in maybe_list] for maybe_list in results]