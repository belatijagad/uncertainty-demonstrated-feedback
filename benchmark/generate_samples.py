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
    base_dir: str,
) -> None:
    responses_dict = {
        "prompt": [],
        "completion": [],
    }

    model_inputs = []
    csv_prompts = []
    decoded_prefixes = [] 

    # --- 1. Prepare Inputs (Same as before) ---
    
    if method_name == "zero_shot":
        model_inputs = prompts
        csv_prompts = prompts
        
    elif method_name == "few_shot":
        # ... (Same few-shot construction logic) ...
        formatted_examples = []
        for example in examples:
            formatted_examples.append(f"Prompt:\n{example.get('prompt')}\n\nCompletion:\n{example.get('chosen')}")
        few_shot_context = "\n\n".join(formatted_examples)
        
        model_inputs = [f"{few_shot_context}\n\nPrompt:\n{p}\n\nCompletion:" for p in prompts]
        csv_prompts = prompts

    elif method_name in ["sft", "ditto"]:
        csv_prompts = prompts 
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs.append(formatted)

    # Calculate prefixes for stripping later
    for inp in model_inputs:
        temp_ids = tokenizer(inp, add_special_tokens=False)["input_ids"]
        prefix = tokenizer.decode(temp_ids, skip_special_tokens=True)
        decoded_prefixes.append(prefix)

    # --- 2. Generate with CLIENT-SIDE BATCHING (The Fix) ---
    
    # Extract batch size (default to 1 to be safe on 8B models)
    # We use .pop() so we don't pass 'batch_size' into the HF generate function
    batch_size = gen_kwargs.pop("batch_size", 1) 
    
    all_text_chunks = []

    # Loop through the inputs in steps of 'batch_size'
    for i in range(0, len(model_inputs), batch_size):
        # Slice the list
        batch_inputs = model_inputs[i : i + batch_size]
        
        # Call your UNCHANGED utility function with the smaller list
        batch_chunks, _, _ = generate_model_outputs(
            batch_inputs,
            model,
            tokenizer,
            gen_kwargs=gen_kwargs,
        )
        
        # Accumulate the text results
        all_text_chunks.extend(batch_chunks)
        
    # --- 3. Save & Cleanup (Use all_text_chunks) ---
    
    for raw_prompt, prefix, generations in zip(csv_prompts, decoded_prefixes, all_text_chunks, strict=True):
        responses_dict["prompt"].append(raw_prompt)
        
        full_generation = generations[0] if generations else ""
        
        if full_generation.startswith(prefix):
            clean_completion = full_generation[len(prefix):]
        else:
            clean_completion = full_generation

        responses_dict["completion"].append(clean_completion.strip())

    # --- 4. Write ---
    responses = pd.DataFrame.from_dict(responses_dict)
    os.makedirs(base_dir, exist_ok=True)
    responses.to_csv(f"{base_dir}/{method_name}.csv", index=False)

@hydra.main(version_base=None, config_path="../configs", config_name="generation")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    seed_everything(config.seed)

    relative_checkpoint_dir = os.path.join(
        config.checkpoints.base_dir, 
        config.checkpoints.run_name,
        "ditto"
    )
    checkpoint_dir = hydra.utils.to_absolute_path(relative_checkpoint_dir)
    output_dir = hydra.utils.to_absolute_path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    generation_config = OmegaConf.to_container(config.gen_kwargs, resolve=True)

    logger.info(f"Starting generation for model {config.model.name_or_path} on dataset {config.dataset.name_or_path}")

    # Load data
    dataset = load_dataset(dataset_config["name_or_path"])
    examples, prompts = process_dataset(dataset, dataset_config)

    # Generate zero-shot and few-shot completions
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path).to(config.model["device"])
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="zero_shot", base_dir=output_dir)
    logger.info("-> Finished generating zero shot generations.")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="few_shot", base_dir=output_dir)
    logger.info("-> Finished generating few shot generations.")

    # Generate SFT and DITTO completions
    model = PeftModel.from_pretrained(
        model=model, 
        model_id=checkpoint_dir + "/ref_model", 
        adapter_name="ref_model", 
        is_trainable=False,
    ).to(config.model["device"])
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="sft", base_dir=output_dir)
    logger.info("-> Finished generating SFT generations.")

    model.load_adapter(checkpoint_dir + "/policy_model", adapter_name="policy_model")
    model.set_adapter("policy_model")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="ditto", base_dir=output_dir)
    logger.info("-> Finished generating DITTO generations.")

if __name__ == "__main__":
    main()