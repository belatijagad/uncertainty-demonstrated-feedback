import os
import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.collators import OfflineDPODataCollator
from alignment.utils import seed_everything, process_data, generate_outputs

logger = logging.getLogger(__name__)

def generate_few_shot(samples: list[tuple[str, str]]):
    prompt = "Below are a few writing samples."
    for i, (sample_prompt, sample_response) in enumerate(samples, start=1):
        prompt += f"\n\n### Example {i}\n{sample_prompt}\n{sample_response}"
    prompt += "\n\nRespond to the following prompt in the same way as the writing samples."
    return prompt

def save_outputs(outputs: list[str], output_path: str, config: DictConfig) -> None:
    save_data = {"num_samples": len(outputs), "outputs": outputs}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(outputs)} generated outputs to {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="generate_samples")
def main(config: DictConfig):
    load_dotenv()
    OmegaConf.resolve(config)
    seed_everything(config.seed)

    model_path = config.model.name_or_path
    potential_local_path = os.path.join(get_original_cwd(), model_path)
    
    if os.path.isdir(potential_local_path):
        logger.info(f"Loading model from local directory: {potential_local_path}")
        model_load_path = potential_local_path
    else:
        logger.info(f"Loading model from Hugging Face Hub: {model_path}")
        model_load_path = model_path

    torch_dtype = torch.bfloat16 if config.model.use_bf16 else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(model_load_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    logger.info("Policy loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded successfully.")

    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    logger.info("Dataset loaded successfully.")

    train_dataset, _, test_dataset = process_data(full_dataset, config.dataset, seed=config.seed, logger=logger)

    if config.generation.is_few_shot:
        logger.info("Applying few-shot prompting based on authors")
        
        train_by_author = {}
        for example in train_dataset:
            author = example['author']
            if author not in train_by_author:
                train_by_author[author] = []
            train_by_author[author].append((example['prompt'], example['accepted']))
        
        modified_test_data = []
        for example in test_dataset:
            author = example['author']
            
            if author in train_by_author:
                author_examples = train_by_author[author]
                
                max_examples = getattr(config.generation, 'max_few_shot_examples', 3)
                selected_examples = author_examples[:max_examples]
                
                few_shot_prompt = generate_few_shot(selected_examples)
                modified_prompt = f"{few_shot_prompt}\n\n### New Prompt\n{example['prompt']}"
                
                modified_example = example.copy()
                modified_example['prompt'] = modified_prompt
                modified_test_data.append(modified_example)
                
                logger.debug(f"Applied {len(selected_examples)} few-shot examples for author {author}")
            else:
                logger.warning(f"No training examples found for author {author}, using original prompt")
                modified_test_data.append(example)
        
        test_dataset = Dataset.from_list(modified_test_data)
        logger.info(f"Applied few-shot prompting to {len(test_dataset)} test examples")

    logger.info(f"Generating output for {len(test_dataset)} examples")

    data_collator = OfflineDPODataCollator(
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        max_prompt_length=config.dataset.max_length // 2,
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=False, 
        collate_fn=data_collator, 
        num_workers=config.dataset.num_workers
    )

    logger.info("Starting generation...")
    generated_outputs = generate_outputs(model, tokenizer, test_dataloader, config.generation)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_filename = f"{model.name_or_path.replace('/', '_')}_generated_samples.json"
    output_path = os.path.join(output_dir, output_filename)

    save_outputs(generated_outputs, output_path)

    logger.info("Generation complete!")
    logger.info(f"Total samples generated: {len(generated_outputs)}")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()