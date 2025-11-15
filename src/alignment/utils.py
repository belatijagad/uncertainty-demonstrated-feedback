import random
from contextlib import nullcontext
from typing import Union, Any, Iterable, Optional

import torch
import numpy as np
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModelForCausalLM
from transformers import PreTrainedTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


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
    model: PeftModelForCausalLM | LLM,
    tokenizer: Optional[PreTrainedTokenizer],
    device: str = "cpu",
    lora_request: Optional[LoRARequest] = None,
    disable_peft_adapter: bool = False,
    adapter_name: Optional[str] = None,
    gen_kwargs: dict | None = None,
) -> list[dict]:
    """
    returns [ # For each batch
        [ # For each generated sample (same text, just different chosen)
            {"generated_text": ..., "generated_token_ids": ..., "logits": ...}, # sample 1
            {"generated_text": ..., "generated_token_ids": ..., "logits": ...}, # sample 2
            ...
        ], # batch 1
        ..., # batch 2
    ]
    """
    prompts = list(prompts)
    gen_kwargs = gen_kwargs or {}
    num_return_seqs = gen_kwargs.get("num_return_sequences", 1)
    max_return_tokens = gen_kwargs.get("max_return_tokens", 512)
    do_sample = gen_kwargs.get("do_sample", True)

    results = []
    def _append_batch(samples: list[dict]) -> None:
        results.append(samples[0] if num_return_seqs == 1 else samples)

    def _format_sample(text, token_ids, logprobs):
        return {
            "generated_text": text,
            "generated_token_ids": token_ids,
            "logprobs": logprobs,
        }

    if isinstance(model, LLM):
        model.wake_up()
        params = SamplingParams(
            n=num_return_seqs,
            max_tokens=max_return_tokens,
            temperature=0.7 if do_sample else 0.0,
        )
        outputs = model.generate(prompts, params, lora_request=lora_request)
        model.sleep(level=2)

        for batch in outputs: # Iterate through batch
            batch_samples = [
                _format_sample(
                    generation.text,
                    generation.token_ids,
                    generation.logprobs,
                )
                for generation in batch.outputs
            ]
            _append_batch(batch_samples)
    else:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        model_inputs = {k: v for k, v in inputs.items()}

        if disable_peft_adapter and hasattr(model, "disable_adapter") and adapter_name:
            model.set_adapter(adapter_name)
            adapter_ctx = model.disable_adapter()
        else:
            adapter_ctx = nullcontext()
            
        with adapter_ctx, torch.inference_mode():
            outputs = model.generate(**model_inputs, **gen_kwargs)

        logits = torch.stack(outputs.logits, dim=1).cpu() if hasattr(outputs, "logits") and outputs.logits else None
        if logits is None and hasattr(outputs, "scores") and outputs.scores:
            logits = torch.stack(outputs.scores, dim=1).cpu()
        logprobs = torch.log_softmax(logits, dim=-1) if logits is not None else None
        sequences = outputs.sequences.cpu()

        total_sequences = sequences.size(0)
        for offset in range(0, total_sequences, num_return_seqs):
            seq_chunk = sequences[offset: offset + num_return_seqs]
            logprob_chunk = (
                logprobs[offset: offset + num_return_seqs]
                if logprobs is not None
                else [None] * len(seq_chunk)
            )

            batch_samples = [
                _format_sample(
                    tokenizer.decode(seq, skip_special_tokens=True),
                    seq,
                    logprob_chunk[idx] if logprobs is not None else None,
                )
                for idx, seq in enumerate(seq_chunk)
            ]

            _append_batch(batch_samples)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

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
    gen_kwargs = {
        "num_return_sequences": 1,
        "do_sample": False,
        "max_new_tokens": config.dataset.max_length // 2,
    }

    generations = batched_generate(
        dataset["prompt"],
        model=model,
        tokenizer=tokenizer,
        device=config.model.device,
        disable_peft_adapter=config.model.disable_adapter_during_eval,
        gen_kwargs=gen_kwargs,
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

