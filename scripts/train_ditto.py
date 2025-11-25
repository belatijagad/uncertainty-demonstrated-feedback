import gc
import sys
import logging
from pathlib import Path
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



logger = logging.getLogger(__name__)
logging.getLogger("transformers.pipelines").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../configs", config_name="ditto")
def main(config: DictConfig):
    load_dotenv()
    seed_everything(config.seed)

    # Prepare model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model["name_or_path"],
        attn_implementation=config.model["attn_implementation"],
        dtype=torch.bfloat16 if config.model["use_bf16"] else torch.float32,
    ).to(config.model["device"])
    lora_config = LoraConfig(**config.lora, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="ref_model")
    model.set_adapter("ref_model")
    tokenizer = AutoTokenizer.from_pretrained(config.model["name_or_path"], padding_side="left")
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

    if enable_wabdb := config.wandb.pop("enabled"):
        wandb.init(**config.wandb)

    # Train SFT
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=SFTConfig(
            output_dir=config.output_dir + "/sft",
            report_to="wandb" if enable_wabdb else "none",
            chat_template_path=config.model["name_or_path"],
            dataset_text_field="text",
            optimizer_cls_and_kwargs=(),
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
    data_collator = DITTOCollator(
        **config.sampler,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
    )

    dpo_trainer = DITTOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=config.output_dir + "/dpo",
            report_to="wandb" if enable_wabdb else "none",
            model_adapter_name="policy_model",
            ref_adapter_name="ref_model",
            optimizer_cls_and_kwargs=(AdamW, config.optim_args.dpo),
            **config.training_args.dpo,
            **config.training_args.general,
        ),
        optimizer_cls_and_kwargs=(AdamW, config.optim_args.dpo),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # TODO: how to make the eval?
        data_collator=data_collator,
        callbacks=[
            ResampleCallback(
                model, tokenizer, train_dataset, data_collator, config.sampler
            ),
        ],
    )
    dpo_trainer.train()


if __name__ == "__main__":
    main()
