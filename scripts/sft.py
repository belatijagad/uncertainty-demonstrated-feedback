import logging
import random
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from alignment.trainers import SFTTrainer
from alignment.collators.sft import SFTDataCollator
from alignment.callbacks import LoggingCallback
from alignment.utils import seed_everything

logger = logging.getLogger(__name__)

CHATML_TEMPLATE = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def setup_model_and_tokenizer(config: DictConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads and configures the model and tokenizer."""
    logger.info("--- Setting up model and tokenizer ---")
    
    torch_dtype = torch.bfloat16 if config.model.get("use_bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=config.model.get("attn_implementation"),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    })
    
    tokenizer.chat_template = CHATML_TEMPLATE
    model.resize_token_embeddings(len(tokenizer))
        
    return model, tokenizer


def process_dataset(dataset: DatasetDict, config: DictConfig) -> tuple[Dataset, Dataset]:
    train_split = dataset["train"] if "train" in dataset else dataset["val"]
    eval_split = dataset["val"] if "val" in dataset else dataset["test"]

    author_id = config.get("author_id")
    if author_id is None:
        raise ValueError("Configuration error: `config.dataset.author_id` must be specified for SFT.")

    train_samples_per_author = config.get("train_samples_per_author", 100)
    eval_samples_per_author = config.get("eval_samples_per_author", 20)

    logger.info(
        "Processing dataset for author '%s' (train=%d, eval=%d)",
        author_id,
        train_samples_per_author,
        eval_samples_per_author,
    )

    train_by_author = {}
    for example in train_split:
        train_by_author.setdefault(example["author_id"], []).append(example)

    eval_by_author = {}
    for example in eval_split:
        eval_by_author.setdefault(example["author_id"], []).append(example)

    if author_id not in train_by_author or author_id not in eval_by_author:
        raise ValueError(f"Specified author_id '{author_id}' not found in both train and eval splits.")

    train_examples = list(train_by_author[author_id])
    eval_examples = list(eval_by_author[author_id])

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    train_examples = train_examples[:min(len(train_examples), train_samples_per_author)]
    eval_examples = eval_examples[:min(len(eval_examples), eval_samples_per_author)]

    logger.info(
        "Selected %d training and %d evaluation samples for author '%s'",
        len(train_examples),
        len(eval_examples),
        author_id,
    )

    return Dataset.from_list(train_examples), Dataset.from_list(eval_examples)


def add_messages_column(dataset: Dataset) -> Dataset:
    def _to_messages(example: dict) -> dict:
        prompt = example.get("prompt")
        chosen = example.get("chosen") or example.get("accepted")

        if prompt is None or chosen is None:
            raise ValueError("Dataset examples must contain 'prompt' and 'chosen' fields for SFT.")

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]

        return {"messages": messages}

    return dataset.map(_to_messages)

@hydra.main(version_base=None, config_path="../configs/sft", config_name="default")
def main(config: DictConfig):
    logger.info("--- Starting Supervised Fine-Tuning (SFT) ---")

    OmegaConf.resolve(config)
    seed_everything(config.seed)

    adapter_name = config.model.get("adapter_name", "sft")
    project_root = Path(get_original_cwd())
    lora_adapter_path = project_root / "src" / "alignment" / "temp" / adapter_name
    lora_adapter_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(config)

    lora_config_dict = OmegaConf.to_container(config.lora, resolve=True)
    lora_config = LoraConfig(**lora_config_dict, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    model.save_pretrained(str(lora_adapter_path), adapter_name=adapter_name)
    
    dataset_dict = load_dataset(config.dataset.name_or_path)
    train_dataset, eval_dataset = process_dataset(dataset_dict, config.dataset)

    train_dataset = add_messages_column(train_dataset)
    eval_dataset = add_messages_column(eval_dataset)

    data_collator = SFTDataCollator(tokenizer, max_length=config.dataset.max_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=config.dataset.num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        collate_fn=data_collator,
        num_workers=config.dataset.num_workers
    )
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    trainer_config = OmegaConf.create(OmegaConf.to_container(config.trainer, resolve=True))
    trainer_config.epochs = trainer_config.get("epochs", config.get("epochs", 1))
    scheduler_cfg = config.get("scheduler")
    warmup_steps = trainer_config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = scheduler_cfg.get("warmup_steps", 0) if scheduler_cfg else 0
        trainer_config.warmup_steps = warmup_steps
    trainer_config.max_length = trainer_config.get("max_length", config.dataset.max_length)
    trainer_config.gradient_accumulation_steps = trainer_config.get(
        "gradient_accumulation_steps",
        config.get("gradient_accumulation_steps", 1),
    )

    callbacks = []
    logging_steps = trainer_config.get("logging_steps", 500)
    callbacks.append(LoggingCallback(logging_steps=logging_steps))

    device = config.model.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = SFTTrainer(
        model=model,
        adapter_name=adapter_name,
        config=trainer_config,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        lora_save_path=lora_adapter_path,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    if eval_metrics:
        logger.info("Evaluation metrics: %s", eval_metrics)

    output_dir = Path(config.model.get("output_path", "outputs/sft_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(output_dir=str(output_dir))

if __name__ == "__main__":
    main()