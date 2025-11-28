import gc
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
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, SFTConfig, SFTTrainer

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
    apply_chat_template,
    clone_adapter,
    generate_rejected_responses,
    process_dataset,
    seed_everything,
)
from scripts.estimator import RandomEstimator, MSP



logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="ditto")
def main(config: DictConfig):
    load_dotenv()
    seed_everything(config["seed"])

    run_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    # Prepare model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model["name_or_path"],
        attn_implementation=config.model["attn_implementation"],
        dtype=torch.bfloat16 if config.model["use_bf16"] else torch.float32,
    ).to(config.model["device"])
    lora_config = LoraConfig(**OmegaConf.to_container(config.lora, resolve=True), task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ref_model")
    model.set_adapter("ref_model")
    tokenizer = AutoTokenizer.from_pretrained(config.model["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset
    raw_datasets = load_dataset(config.dataset["name_or_path"])
    train_dataset, eval_dataset = process_dataset(raw_datasets, config.dataset, logger)
    train_dataset = train_dataset.add_column(
        "example_id", list(range(len(train_dataset)))
    )
    eval_dataset = eval_dataset.add_column("example_id", list(range(len(eval_dataset))))
    eval_dataset = generate_rejected_responses(
        eval_dataset, model, tokenizer, config.dataset, logger
    )

    def _format_example(example):
        return apply_chat_template(example, tokenizer)

    sft_train_dataset = train_dataset.map(
        _format_example,
        num_proc=0,
        desc="Applying chat template to train split",
    )
    sft_eval_dataset = eval_dataset.map(
        _format_example,
        num_proc=0,
        desc="Applying chat template to eval split",
    )

    if enable_wabdb := config.wandb["enabled"]:
        config.wandb.__delattr__("enabled")
        wandb.init(**config.wandb)

    # Train SFT
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=SFTConfig(
            output_dir=run_dir,
            report_to="wandb" if enable_wabdb else "none",
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

    # Copy adapter weights for DPO
    # TODO: how about merging the SFT weights, then reinstatiate LoRA weights on DPO?
    # https://huggingface.co/docs/peft/en/developer_guides/lora#merge-lora-weights-into-the-base-model
    clone_adapter(cast(PeftModel, model), "ref_model", "policy_model")
    model.set_adapter("policy_model")

    # Train DPO
    # TODO: make a cleaner implementation of mapping estimators
    estimator_map = {
        None: None,
        "None": None,
        "msp": MSP(),
        "random": RandomEstimator(),
    }
    data_collator = DITTOCollator(
        **config.sampler,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        estimator=estimator_map[config.estimator],
    )
    tokenizer.padding_side = "left"
    trainer = DITTOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=run_dir,
            report_to="wandb" if enable_wabdb else "none",
            model_adapter_name="policy_model",
            ref_adapter_name="ref_model",
            push_to_hub=config.push_to_hub,
            hub_model_id=f"{config.model.name}_{config.estimator}_ditto_{config.dataset.name}_{config.dataset.author_id}",
            **config.training_args.dpo,
            **config.training_args.general,
        ),
        # use_cache=not config.training_args.dpo["gradient_checkpointing"],
        optimizer_cls_and_kwargs=(AdamW, config.optim_args.dpo),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            ResampleCallback(
                model, tokenizer, train_dataset, data_collator, config.sampler
            ),
        ],
    )
    trainer.train()

    trainer.save_model()

if __name__ == "__main__":
    main()
