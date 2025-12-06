import gc
import os
import sys
import logging
from pathlib import Path
from typing import cast

import hydra
from omegaconf import OmegaConf
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from datasets import DatasetDict
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, SFTConfig, SFTTrainer
from huggingface_hub import repo_exists, file_exists

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.trainer import DITTOTrainer
from scripts.callback import (
    EarlyStoppingCallback,
    ResampleCallback,
)
from scripts.collator import DITTOCollator
from scripts.utils import (
    clone_adapter,
    seed_everything,
)
from scripts.estimator import ESTIMATOR_MAP



logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="ditto")
def main(config: DictConfig):
    load_dotenv()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    seed_everything(config["seed"])

    run_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    # Prepare model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model["name_or_path"],
        attn_implementation=config.model["attn_implementation"],
        dtype=torch.bfloat16 if config.model["use_bf16"] else torch.float16,
        device_map="auto",
    )
    lora_config = LoraConfig(**OmegaConf.to_container(config.lora, resolve=True), task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ref_model")
    model.set_adapter("ref_model")
    tokenizer = AutoTokenizer.from_pretrained(config.model["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset; I love functional programming.
    dataset: DatasetDict = (
        load_dataset(config.dataset["name_or_path"])["train"]
            .filter(lambda x: x["author_id"] == config.dataset.author_id)
            .map(
                lambda x: {
                    "prompt": tokenizer.apply_chat_template(x["prompt"]),
                    "chosen": tokenizer.apply_chat_template(x["prompt"] + x["chosen"]),
                }
            )
            .shuffle(seed=config.seed)[:config.dataset.train_samples_per_author]
    )

    if enable_wandb := config.wandb["enabled"]:
        config.wandb.__delattr__("enabled")
        wandb.init(**config.wandb)

    # Train SFT
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=run_dir,
            report_to="wandb" if enable_wandb else "none",
            chat_template_path=config.model["name_or_path"],
            dataset_text_field="text",
            **config.training_args.sft,
            **config.training_args.general,
        ),
        optimizer_cls_and_kwargs=(AdamW, config.optim_args.sft),
        callbacks=[EarlyStoppingCallback(threshold=1.0)],
    )
    trainer.train()

    del trainer
    gc.collect()

    # Test all uncertainty methods at once
    repo_username = "belati"
    repo_name_base = f"{config.model.name}_{config.dataset.name}_{config.dataset.author_id}"
    full_repo_id = f"{repo_username}/{repo_name_base}"

    tokenizer.padding_side = "left"
    for name, estimator in ESTIMATOR_MAP.items():
        logger.info(f"Preparing DITTO training for method: {name}")
        
        adapter_name = f"{name}_policy_model"

        if repo_exists(full_repo_id):
            if file_exists(full_repo_id, filename="adapter_config.json"):
                logger.info(f"=>> Method {name} already exists in {full_repo_id}; skipping...")
                continue
        
        # Clone SFT weights to new Adapter
        clone_adapter(cast(PeftModel, model), "ref_model", adapter_name)
        model.set_adapter(adapter_name)

        data_collator = DITTOCollator(
            **config.sampler,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            estimator=estimator,
        )
        
        dpo_trainer = DITTOTrainer(
            model=model,
            args=DPOConfig(
                output_dir=str(Path(run_dir) / name),
                report_to="wandb" if enable_wandb else "none",
                model_adapter_name=adapter_name,
                ref_adapter_name="ref_model",
                push_to_hub=config.push_to_hub,
                hub_model_id=full_repo_id,
                remove_unused_columns=False,
                **config.training_args.dpo,
                **config.training_args.general,
            ),
            optimizer_cls_and_kwargs=(AdamW, config.optim_args.dpo),
            processing_class=tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[
                ResampleCallback(
                    model, tokenizer, dataset, data_collator, config.sampler
                ),
            ],
        )
        
        dpo_trainer.train()
        dpo_trainer.save_model()
        
        # Cleanup
        del dpo_trainer
        model.delete_adapter(adapter_name)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Switch back to ref for next cloning
        model.set_adapter("ref_model")

if __name__ == "__main__":
    main()
