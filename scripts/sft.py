import logging

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset

from alignment.trainers import SFTTrainer
from alignment.collators.sft import SFTDataCollator
from alignment.callbacks import LoggingCallback

logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(config: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads and configures the model and tokenizer."""
    logger.info("--- Setting up model and tokenizer ---")
    
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    CHATML_TEMPLATE = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    })
    
    tokenizer.chat_template = CHATML_TEMPLATE
    model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer

@hydra.main(version_base=None, config_path="../configs", config_name="pythia160m_sft")
def main(config: DictConfig):
    logger.info("--- Starting Supervised Fine-Tuning (SFT) ---")

    model, tokenizer = setup_model_and_tokenizer(config)
    
    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    
    subset_size = config.dataset.get("subset_size", None)
    if subset_size is not None:
        logger.info(f"Using a subset of the data: {subset_size}")
        
        if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
            num_samples = int(len(full_dataset) * subset_size)
        elif isinstance(subset_size, int):
            num_samples = subset_size
        else:
            raise ValueError(f"Invalid subset_size: {subset_size}. Must be a float <= 1.0 or an int.")

        full_dataset = full_dataset.shuffle(seed=config.seed).select(range(num_samples))

    dataset_splited = full_dataset.train_test_split(test_size=0.05, seed=config.seed)
    
    data_collator = SFTDataCollator(tokenizer, max_length=config.dataset.max_length)

    train_dataloader = DataLoader(
        dataset_splited["train"],
        batch_size=config.dataset.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=config.dataset.num_workers
    )
    eval_dataloader = DataLoader(
        dataset_splited["test"],
        batch_size=config.dataset.batch_size,
        collate_fn=data_collator,
        num_workers=config.dataset.num_workers
    )
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    callbacks = []    
    logging_callback = LoggingCallback(logging_steps=config.get("logging_steps", 500))
    callbacks.append(logging_callback)
    
    trainer = SFTTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        callbacks=callbacks
    )

    trainer.train()
    trainer.evaluate()
    trainer.save()

if __name__ == "__main__":
    main()