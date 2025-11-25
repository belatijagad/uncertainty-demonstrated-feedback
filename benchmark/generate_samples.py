import os
import sys
import random
import logging
import importlib
from pathlib import Path
from typing import Any, Literal

import hydra
import pandas as pd
from peft import PeftModel
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

utils_module = importlib.import_module("scripts.utils")
seed_everything = utils_module.seed_everything
generate_model_outputs = utils_module.generate_model_outputs

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)

def process_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    train_split = dataset["train"]

    author_id = dataset_kwargs["author_id"]
    n_train = dataset_kwargs["num_train_samples"]
    n_eval = dataset_kwargs["num_eval_samples"]

    author_rows: list[dict[str, Any]] = [
        example for example in train_split if example.get("author_id") == author_id
    ]

    assert author_rows

    random.shuffle(author_rows)

    examples = author_rows[:n_train]
    eval_candidates = author_rows[n_train:n_train+n_eval]

    prompts: list[str] = []
    for example in eval_candidates[:n_eval]:
        prompts.append(str(example.get("prompt")))

    return examples, prompts


def generate_results(
    model: PreTrainedModel | PeftModel, 
    tokenizer: PreTrainedTokenizer, 
    prompts: list[str],
    examples: list[dict[str, Any]],
    gen_kwargs: dict[str, Any],
    method_name: Literal["zero_shot", "few_shot", "sft", "ditto"],
) -> None:
    responses_dict = {
        "prompt": [],
        "completion": [],
    }

    prompts_to_use = []

    if method_name == "zero_shot":
        prompts_to_use = prompts
    elif method_name == "few_shot":
        formatted_examples: list[str] = []

        for example in examples:
            prompt_text = example.get("prompt")
            completion_text = example.get("chosen")
            formatted_examples.append(
                f"Prompt:\n{prompt_text}\n\nCompletion:\n{completion_text}"
            )

        few_shot_context = "\n\n".join(formatted_examples)
        prompts_to_use = [
            f"{few_shot_context}\n\nPrompt:\n{prompt}\n\nCompletion:"
            for prompt in prompts
        ]
    elif method_name in ["sft", "ditto"]:
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts_to_use.append(formatted)

    text_chunks, _, _ = generate_model_outputs(
        prompts_to_use,
        model,
        tokenizer,
        gen_kwargs=gen_kwargs,
    )

    for prompt_text, generations in zip(prompts_to_use, text_chunks, strict=True):
        responses_dict["prompt"].append(prompt_text)
        completion_text = generations[0] if generations else ""
        responses_dict["completion"].append(completion_text)

    responses = pd.DataFrame.from_dict(responses_dict)
    responses.to_csv(f"{method_name}.csv", index=False)

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    seed_everything(config.seed)

    relative_checkpoint_dir = os.path.join(
        config.checkpoints.base_dir, 
        config.checkpoints.run_name
    )
    checkpoint_dir = hydra.utils.to_absolute_path(relative_checkpoint_dir)

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    generation_config = OmegaConf.to_container(config.gen_kwargs, resolve=True)

    logger.info(f"Starting generation for model {config.model.name_or_path} on dataset {config.dataset.name_or_path}")

    # Load data
    dataset = load_dataset(dataset_config["name_or_path"])
    examples, prompts = process_dataset(dataset, dataset_config)

    # Generate zero-shot and few-shot completions
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="zero_shot")
    logger.info("-> Finished generating zero shot generations.")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="zero_shot")
    logger.info("-> Finished generating few shot generations.")

    # Generate SFT and DITTO completions
    model = PeftModel.from_pretrained(
        model=model, 
        model_id=checkpoint_dir, 
        adapter_name="sft", 
        is_trainable=False,
    ).to(config["device"])
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="sft")
    logger.info("-> Finished generating SFT generations.")

    model.set_adapter("ditto")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="ditto")
    logger.info("-> Finished generating DITTO generations.")

if __name__ == "__main__":
    main()