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

import random
from typing import Any
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

def process_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    
    author_id = dataset_kwargs["author_id"]
    n_train = dataset_kwargs["num_train_samples"]
    n_eval = dataset_kwargs["num_eval_samples"]
    target_eval_split = dataset_kwargs["eval_split"]

    train_data = dataset["train"]
    train_rows: list[dict[str, Any]] = [
        example for example in train_data if example.get("author_id") == author_id
    ]
    
    assert train_rows, f"No rows found for author {author_id} in 'train' split."
    random.shuffle(train_rows)
    
    examples = train_rows[:n_train]

    eval_candidates_rows = []

    if target_eval_split == "train":
        eval_candidates_rows = train_rows[n_train:]
    else:
        if target_eval_split not in dataset:
            raise ValueError(f"Split '{target_eval_split}' not found in dataset.")
            
        eval_data = dataset[target_eval_split]
        eval_rows: list[dict[str, Any]] = [
            example for example in eval_data if example.get("author_id") == author_id
        ]
        
        assert eval_rows, f"No rows found for author {author_id} in '{target_eval_split}' split."
        random.shuffle(eval_rows)
        eval_candidates_rows = eval_rows

    final_eval_selection = eval_candidates_rows[:n_eval]

    prompts: list[str] = []
    for example in final_eval_selection:
        prompts.append(str(example.get("prompt")))

    return examples, prompts

def generate_examples(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset, 
    dataset_kwargs: dict[str, Any],
    base_dir: str,
) -> None:
    assert dataset_kwargs["eval_split"] != "train", "Doesn't support `train` split currently."
    example_dataset = dataset[dataset_kwargs["eval_split"]].to_pandas()
    example_dataset = example_dataset.loc[example_dataset.author_id == dataset_kwargs["author_id"]]
    os.makedirs(base_dir+"/examples", exist_ok=True)
    example_dataset.to_csv(base_dir + "/examples/" + f"{dataset_kwargs["name"]}_{dataset_kwargs["author_id"]}.csv", index=False)


def generate_results(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompts: list[str],
    examples: list[dict[str, Any]],
    gen_kwargs: dict[str, Any],
    method_name: Literal["zero-shot", "few-shot", "sft", "ditto"],
    base_dir: str,
) -> None:
    responses_dict = {
        "prompt": [],
        "completion": [],
    }

    model_inputs = []
    csv_prompts = []
    decoded_prefixes = [] 

    gen_kwargs = gen_kwargs.copy()
    batch_size = gen_kwargs.pop("batch_size", 1) 

    if method_name == "zero-shot":
        model_inputs = prompts
        csv_prompts = prompts
        
    elif method_name == "few-shot":
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

    for inp in model_inputs:
        temp_ids = tokenizer(inp, add_special_tokens=False)["input_ids"]
        prefix = tokenizer.decode(temp_ids, skip_special_tokens=True)
        decoded_prefixes.append(prefix)

    all_text_chunks = []

    for i in range(0, len(model_inputs), batch_size):
        batch_inputs = model_inputs[i : i + batch_size]
        
        batch_chunks, _, _ = generate_model_outputs(
            batch_inputs,
            model,
            tokenizer,
            gen_kwargs=gen_kwargs,
        )
        all_text_chunks.extend(batch_chunks)
            
    for raw_prompt, prefix, generations in zip(csv_prompts, decoded_prefixes, all_text_chunks, strict=True):
        clean_prompt = raw_prompt.replace('\n', '\\n')
        responses_dict["prompt"].append(clean_prompt)
        
        full_generation = generations[0] if generations else ""
        
        if full_generation.startswith(prefix):
            clean_completion = full_generation[len(prefix):]
        else:
            clean_completion = full_generation

        clean_completion = clean_completion.strip().replace('\n', '\\n')

        responses_dict["completion"].append(clean_completion)

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
    )
    checkpoint_dir = hydra.utils.to_absolute_path(relative_checkpoint_dir)
    output_dir = hydra.utils.to_absolute_path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    generation_config = OmegaConf.to_container(config.gen_kwargs, resolve=True)

    logger.info(f"Starting generation for model {config.model.name_or_path} on dataset {config.dataset.name_or_path}")

    # Load data
    dataset = load_dataset(dataset_config["name_or_path"])
    examples, prompts = process_dataset(dataset, dataset_config)

    # Generate examples
    generate_examples(dataset, dataset_config, base_dir=str(Path(output_dir).parent))

    # Generate zero-shot and few-shot completions
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path).to(config.model["device"])
    tokenizer = AutoTokenizer.from_pretrained(relative_checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if config.estimator in ["None", None]:
        generate_results(model, tokenizer, prompts, examples, generation_config, method_name="zero-shot", base_dir=output_dir)
        logger.info("-> Finished generating zero shot generations.")
        generate_results(model, tokenizer, prompts, examples, generation_config, method_name="few-shot", base_dir=output_dir)
        logger.info("-> Finished generating few shot generations.")

    # Generate SFT and DITTO completions
    model = PeftModel.from_pretrained(
        model=model, 
        model_id=checkpoint_dir + "/ref_model", 
        adapter_name="ref_model", 
        is_trainable=False,
    ).to(config.model["device"])
    if config.estimator in ["None", None]:
        generate_results(model, tokenizer, prompts, examples, generation_config, method_name="sft", base_dir=output_dir)
        logger.info("-> Finished generating SFT generations.")

    model.load_adapter(checkpoint_dir + "/policy_model", adapter_name="policy_model")
    model.set_adapter("policy_model")
    generate_results(model, tokenizer, prompts, examples, generation_config, method_name="ditto", base_dir=output_dir)
    logger.info("-> Finished generating DITTO generations.")

if __name__ == "__main__":
    # TODO: make sweep version to inference models, dataset and authorid, and estimator
    main()
