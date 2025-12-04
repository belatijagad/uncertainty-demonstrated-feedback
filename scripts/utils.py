import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clone_adapter(model: PeftModel, src_name: str, tgt_name: str) -> None:
    model.add_adapter(tgt_name, model.peft_config[src_name])
    src_weights = get_peft_model_state_dict(model, adapter_name=src_name)
    tgt_weights = {k.replace(src_name, tgt_name): v for k, v in src_weights.items()}
    set_peft_model_state_dict(model, tgt_weights, adapter_name=tgt_name)


def apply_chat_template(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """
    Formats the example for SFT (creating a 'text' column)
    AND ensures 'prompt'/'chosen' are ready for DITTO/DPO.
    """
    p_content = example["prompt"]
    c_content = example["chosen"]

    # Convert strings to list of dicts if necessary
    prompt_messages = (
        [{"role": "user", "content": p_content}]
        if isinstance(p_content, str)
        else p_content
    )
    chosen_messages = (
        [{"role": "assistant", "content": c_content}]
        if isinstance(c_content, str)
        else c_content
    )

    messages = prompt_messages + chosen_messages

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    example["prompt"] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    if isinstance(chosen_messages, list):
        example["chosen"] = chosen_messages[0]["content"]

    # TRL's SFTTrainer expects a `completion` column (prompt+completion schema).
    example.setdefault("completion", example["chosen"])

    return example


def generate_model_outputs(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    gen_kwargs: dict[str, Any],
) -> tuple[list[list[str]], torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    if isinstance(gen_kwargs, DictConfig):
        gen_kwargs = OmegaConf.to_container(gen_kwargs, resolve=True)
    else:
        gen_kwargs = gen_kwargs.copy()

    if gen_kwargs.get("eos_token_id") is None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if gen_kwargs.get("pad_token_id") is None:
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        else:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
    if gen_kwargs.get("pad_token_id") is None:
        gen_kwargs["pad_token_id"] = 0

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs,
        )
    
    raw_logits = torch.stack(outputs.scores, dim=-1).cpu()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        beam_indices=getattr(outputs, "beam_indices", None),
        normalize_logits=False,
    ).cpu()

    decoded_text = []
    for seq in outputs.sequences:
        full_decode = tokenizer.decode(seq, skip_special_tokens=False)
        if tokenizer.pad_token:
            full_decode = full_decode.replace(tokenizer.pad_token, "")
        decoded_text.append(full_decode)

    batch_size = len(prompts)
    num_return_sequences = gen_kwargs.get("num_return_sequences", 1)

    scores_view = transition_scores.contiguous().view(
        batch_size, num_return_sequences, -1
    )
    sequences_view = (
        outputs.sequences.cpu()
        .contiguous()
        .view(batch_size, num_return_sequences, outputs.sequences.size(-1))
    )
    logits_view = raw_logits.contiguous().view(
        batch_size, num_return_sequences, -1, raw_logits.size(-1)
    )
    text_chunks = [
        decoded_text[i : i + num_return_sequences]
        for i in range(0, len(decoded_text), num_return_sequences)
    ]

    return text_chunks, sequences_view, scores_view, logits_view

def process_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    config: dict[str, Any],
    logger,
) -> tuple[Dataset, Dataset]:
    train_split = dataset["train"] if "train" in dataset else dataset["val"]
    eval_split = dataset["val"] if "val" in dataset else dataset["test"]

    author_id = config["author_id"]

    train_samples_per_author = config.get("train_samples_per_author", 100)
    eval_samples_per_author = config.get("eval_samples_per_author", 20)

    logger.info(
        f"Processing dataset: selecting author '{author_id}' with "
        f"up to {train_samples_per_author} train and {eval_samples_per_author} eval samples."
    )

    train_author_data = defaultdict(list)
    for example in train_split:
        author = example["author_id"]
        train_author_data[author].append(example)

    eval_author_data = defaultdict(list)
    for example in eval_split:
        author = example["author_id"]
        eval_author_data[author].append(example)

    assert author_id in train_author_data

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

    logger.info(
        f"Final dataset sizes: train={len(train_dataset)}, eval={len(eval_dataset)}"
    )

    return train_dataset, eval_dataset


def generate_rejected_responses(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: DictConfig,
    logger,
) -> Dataset:
    """
    Generates rejected responses using a pipeline and KeyDataset for efficient,
    streaming batch processing.
    """
    logger.info("Generating rejected responses...")

    gen_kwargs = {"num_return_sequences": 1, "do_sample": False, "repetition_penalty": 0.8}

    prompts = list(dataset["prompt"])
    text_chunks, _, _, _ = generate_model_outputs(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        gen_kwargs=gen_kwargs,
    )

    logger.info(f"Raw generation output (first 2 prompts): {text_chunks[:2]}")

    generated_texts: list[str] = []
    for samples in text_chunks:
        if not samples:
            continue
        text = samples[0]
        if not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token
        generated_texts.append(text)

    logger.info(f"Processed 'generated_texts' (first 2): {generated_texts[:2]}")

    if "rejected" in dataset.column_names:
        dataset = dataset.remove_columns("rejected")
    updated_dataset = dataset.add_column("rejected", generated_texts)

    logger.info(f"Generated {len(updated_dataset)} rejected responses")
    return updated_dataset
