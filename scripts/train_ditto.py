import logging
from typing import cast

import hydra
import torch
import wandb
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_scheduler,
)
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from .callback import EarlyStoppingCallback, ResampleCallback
from .collator import DITTOCollator
from .utils import (
    apply_chat_template,
    clone_adapter,
    generate_rejected_responses,
    process_dataset,
    seed_everything,
)

logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True


@hydra.main(version_base=None, config_path="../configs", config_name="ditto")
def main(config: DictConfig):
    load_dotenv()
    seed_everything(config.seed)

    model_cfg = config.model
    dataset_cfg = config.dataset
    sampler_cfg = config.sampler
    wandb_cfg = config.wandb
    training_cfg = config.training_args
    lora_cfg = config.lora

    output_dir = config.get("output_dir", "outputs")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        attn_implementation=model_cfg.get("attn_implementation"),
        dtype=torch.bfloat16 if model_cfg.get("use_bf16", False) else torch.float32,
    ).to(model_cfg["device"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])

    raw_datasets = load_dataset(dataset_cfg["name_or_path"])
    train_dataset, eval_dataset = process_dataset(raw_datasets, dataset_cfg, logger)
    eval_dataset = generate_rejected_responses(
        eval_dataset, model, tokenizer, dataset_cfg, logger
    )
    assert isinstance(train_dataset, Dataset) and isinstance(eval_dataset, Dataset)

    lora_config = LoraConfig(**lora_cfg, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ref_model")
    model.set_adapter("ref_model")
    assert isinstance(model, PreTrainedModel)

    collator_kwargs = {
        key: sampler_cfg[key]
        for key in (
            "frac_expert",
            "frac_replay",
            "frac_noisy",
            "rescale_batch",
            "bootstrap_count",
        )
        if key in sampler_cfg
    }
    data_collator = DITTOCollator(
        tokenizer=tokenizer,
        **collator_kwargs,
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

    sft_args = SFTConfig(
        output_dir=output_dir,
        report_to="wandb" if wandb_cfg.get("enabled", False) else "none",
        chat_template_path=model_cfg["name_or_path"],
        dataset_text_field="text",
        **training_cfg,
    )
    sft_optimizer = AdamW(model.parameters(), lr=sft_args.learning_rate)
    sft_scheduler = get_scheduler(
        name="linear",
        optimizer=sft_optimizer,
        num_warmup_steps=sft_args.warmup_steps,
        num_training_steps=sft_args.max_steps,
    )

    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_args,
        optimizers=(sft_optimizer, sft_scheduler),
        callbacks=[EarlyStoppingCallback(threshold=1.0)],
    )
    sft_trainer.train()

    clone_adapter(cast(PeftModel, model), "ref_model", "policy_model")
    model.set_adapter("policy_model")

    dpo_args = DPOConfig(
        output_dir=output_dir,
        report_to="wandb" if wandb_cfg.get("enabled", False) else "none",
        model_adapter_name="policy_model",
        ref_adapter_name="ref_model",
        **training_cfg,
    )
    dpo_optimizer = AdamW(model.parameters(), lr=dpo_args.learning_rate)
    dpo_scheduler = get_scheduler(
        name="linear",
        optimizer=dpo_optimizer,
        num_warmup_steps=dpo_args.warmup_steps,
        num_training_steps=dpo_args.max_steps,
    )

    if wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg.get("project", "ditto"),
            config=wandb_cfg,
            name=wandb_cfg.get("name"),
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
            notes=wandb_cfg.get("notes"),
        )

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        optimizers=(dpo_optimizer, dpo_scheduler),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=lora_config,
        callbacks=[
            ResampleCallback(
                model, tokenizer, train_dataset, data_collator, sampler_cfg
            )
        ],
    )
    dpo_trainer.train()


if __name__ == "__main__":
    main()
