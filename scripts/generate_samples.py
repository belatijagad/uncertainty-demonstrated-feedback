import json
import logging
import random
from pathlib import Path
from typing import Iterable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.utils import seed_everything, batched_generate

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model, tokenizer, and attach the requested LoRA adapter."""
    torch_dtype = torch.bfloat16 if config.model.get("use_bf16", True) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.model.get("attn_implementation"),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_name = config.model.get("adapter_name")
    adapter_path = config.model.get("adapter_path")
    adapter_subfolder = config.model.get("adapter_subfolder")
    if adapter_name and adapter_path:
        load_path: str | None = None

        local_path = Path(adapter_path)
        if not local_path.is_absolute():
            local_path = Path(get_original_cwd()) / local_path

        if local_path.exists():
            load_path = str(local_path)
        else:
            load_path = adapter_path

        try:
            logger.info(
                "Loading LoRA adapter '%s' from %s%s",
                adapter_name,
                load_path,
                f" (subfolder='{adapter_subfolder}')" if adapter_subfolder else "",
            )
            model = PeftModel.from_pretrained(
                model,
                load_path,
                adapter_name=adapter_name,
                subfolder=adapter_subfolder,
            )
            model.set_adapter(adapter_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load LoRA adapter '%s': %s", adapter_name, exc)

    model.eval()

    return model, tokenizer


def process_dataset(dataset: DatasetDict, config: DictConfig) -> tuple[Dataset, Dataset]:
    train_split = dataset["train"] if "train" in dataset else dataset["val"]
    eval_split = dataset["val"] if "val" in dataset else dataset["test"]

    author_id = config.get("author_id")
    if author_id is None:
        raise ValueError("Configuration error: `config.dataset.author_id` must be specified.")

    train_samples_per_author = config.get("train_samples_per_author", 100)
    eval_samples_per_author = config.get("eval_samples_per_author", 20)

    logger.info(
        "Selecting prompts for author '%s' (train=%d, eval=%d)",
        author_id,
        train_samples_per_author,
        eval_samples_per_author,
    )

    train_by_author: dict[int, list[dict]] = {}
    for example in train_split:
        train_by_author.setdefault(example["author_id"], []).append(example)

    eval_by_author: dict[int, list[dict]] = {}
    for example in eval_split:
        eval_by_author.setdefault(example["author_id"], []).append(example)

    if author_id not in train_by_author or author_id not in eval_by_author:
        raise ValueError(f"Specified author_id '{author_id}' not present in both train and eval splits.")

    train_examples = list(train_by_author[author_id])
    eval_examples = list(eval_by_author[author_id])

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    train_examples = train_examples[:min(len(train_examples), train_samples_per_author)]
    eval_examples = eval_examples[:min(len(eval_examples), eval_samples_per_author)]

    logger.info(
        "Prepared %d training prompts and %d evaluation prompts for author '%s'",
        len(train_examples),
        len(eval_examples),
        author_id,
    )

    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples)


def select_prompts(train_dataset: Dataset, eval_dataset: Dataset, config: DictConfig) -> list[str]:
    source = config.dataset.get("split", "eval")
    if source not in {"train", "eval"}:
        raise ValueError("config.dataset.split must be either 'train' or 'eval'.")

    selected_dataset = train_dataset if source == "train" else eval_dataset
    prompts: Iterable[str] = selected_dataset["prompt"]

    max_prompts = config.dataset.get("max_prompts")
    if max_prompts is not None:
        prompts = list(prompts)[:max_prompts]
    else:
        prompts = list(prompts)

    logger.info("Generating completions for %d prompts from the %s split.", len(prompts), source)
    return prompts


def generate_responses(
    prompts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: DictConfig,
) -> list[dict[str, object]]:
    if not prompts:
        return []

    generation_cfg = config.generation
    max_new_tokens = generation_cfg.get("max_new_tokens", 512)
    do_sample = generation_cfg.get("do_sample", False)
    num_return_sequences = generation_cfg.get("num_return_sequences", 1)

    logger.info(
        "Running generation with max_new_tokens=%d, do_sample=%s, num_return_sequences=%d",
        max_new_tokens,
        do_sample,
        num_return_sequences,
    )

    adapter_name = config.model.get("adapter_name")
    disable_adapter = config.model.get("disable_adapter_during_generation", False)

    device = config.model.get("device")
    generations = batched_generate(
        prompts,
        max_new_tokens=max_new_tokens,
        model=model,
        tokenizer=tokenizer,
        device=device,
        lora_request=None,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        disable_peft_adapter=disable_adapter,
        adapter_name=adapter_name,
    )

    responses: list[dict[str, object]] = []
    for prompt, texts in zip(prompts, generations, strict=True):
        responses.append({"prompt": prompt, "generations": texts})

    return responses


def save_outputs(results: list[dict[str, object]], config: DictConfig) -> Path:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_cfg = config.get("output") or {}
    filename = output_cfg.get("filename", "generated_samples.json")
    output_path = output_dir / filename

    payload = {
        "num_prompts": len(results),
        "model": config.model.name_or_path,
        "adapter": config.model.get("adapter_name"),
        "generation": OmegaConf.to_container(config.generation, resolve=True),
        "samples": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved generated samples to %s", output_path)
    return output_path


@hydra.main(version_base=None, config_path="../configs", config_name="generate_samples")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    seed_everything(config.seed)

    model, tokenizer = setup_model_and_tokenizer(config)

    dataset_dict = load_dataset(config.dataset.name_or_path)
    train_dataset, eval_dataset = process_dataset(dataset_dict, config.dataset)
    prompts = select_prompts(train_dataset, eval_dataset, config)

    results = generate_responses(prompts, model, tokenizer, config)
    save_outputs(results, config)

    logger.info("Generation complete for %d prompts.", len(results))


if __name__ == "__main__":
    main()