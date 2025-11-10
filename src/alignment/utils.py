import random
from contextlib import nullcontext
from typing import Union, Any, Iterable, Optional

import torch
import numpy as np
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModelForCausalLM
from transformers.pipelines import pipeline
from transformers import PreTrainedTokenizer

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
    model: PeftModelForCausalLM | LLM,
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

def generate_rejected_responses(
    dataset: Dataset,
    model: PeftModelForCausalLM | LLM,
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    logger,
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
    )

    logger.info(f"Raw 'generations' output (first 2): {generations[:2]}")

    generated_texts = [
        text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token
        for texts in generations
        for text in texts[:1]
    ]

    logger.info(f"Processed 'generated_texts' (first 2): {generated_texts[:2]}")
            
    if "rejected" in dataset.column_names:
        dataset = dataset.remove_columns("rejected")
    updated_dataset = dataset.add_column("rejected", generated_texts)

    logger.info(f"Generated {len(updated_dataset)} rejected responses")
    return updated_dataset

def build_model_card(config: DictConfig) -> str:
    """Assemble a lightweight model card using config metadata."""
    base_model = getattr(config.model, "name_or_path", "unknown")
    dataset_name = getattr(config.dataset, "name_or_path", "unknown")
    license_name = config.model.get("license", "apache-2.0") if hasattr(config.model, "get") else "apache-2.0"
    repo_id = config.trainer.get("repo_id", "model") if hasattr(config.trainer, "get") else "model"

    metadata = {
        "license": license_name,
        "base_model": base_model,
        "datasets": [dataset_name] if dataset_name else [],
        "tags": list(config.get("model_card_tags", [])) or ["alignment", "ditto"],
    }

    header_lines = ["---"]
    for key, value in metadata.items():
        if not value:
            continue
        if isinstance(value, list):
            header_lines.append(f"{key}:")
            for item in value:
                header_lines.append(f"  - {item}")
        else:
            header_lines.append(f"{key}: {value}")
    header_lines.append("---")

    body = [
        f"# {repo_id}",
        "",
        "## Model Summary",
        f"- Base model: `{base_model}`",
        f"- Dataset: `{dataset_name}`",
        "- Training objective: DITTO preference alignment",
        "",
        "## Intended Use",
        "This model is intended for research on preference alignment. Update this section with concrete guidance before sharing broadly.",
        "",
        "## Limitations",
        "Document evaluation metrics, known failure modes, and ethical considerations here before public release.",
    ]

    return "\n".join(header_lines + [""] + body)

