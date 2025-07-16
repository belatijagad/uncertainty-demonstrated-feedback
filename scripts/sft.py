import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset

from alignment.train.callbacks import JsonLoggingCallback
from alignment.train.trainers import SFTTrainer
from alignment.data.data_collator import SFTDataCollator
from alignment.data.utils import format_sft_prompt

def setup_model_and_tokenizer(config: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads and configures the model and tokenizer."""
    print("--- Setting up model and tokenizer ---")
    
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
    print("--- Starting Supervised Fine-Tuning (SFT) ---")

    model, tokenizer = setup_model_and_tokenizer(config)
    
    full_dataset = load_dataset(config.dataset.name_or_path, split=config.dataset.split)
    processed_dataset = full_dataset.map(format_sft_prompt)
    
    subset_size = config.dataset.get("subset_size", None)
    if subset_size is not None:
        print(f"Using a subset of the data: {subset_size}")
        
        if isinstance(subset_size, float) and 0 < subset_size <= 1.0:
            num_samples = int(len(full_dataset) * subset_size)
        elif isinstance(subset_size, int):
            num_samples = subset_size
        else:
            raise ValueError(f"Invalid subset_size: {subset_size}. Must be a float <= 1.0 or an int.")

        processed_dataset = processed_dataset.shuffle(seed=config.seed).select(range(num_samples))

    dataset_splited = processed_dataset.train_test_split(test_size=0.05, seed=config.seed)
    
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

    log_dir = config.run_dir
    callbacks = [JsonLoggingCallback(log_dir=log_dir)]
    
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