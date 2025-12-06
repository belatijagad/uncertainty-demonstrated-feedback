import random
from typing import Any

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clone_adapter(model: PeftModel, src_name: str, tgt_name: str) -> None:
    model.add_adapter(tgt_name, model.peft_config[src_name])
    src_weights = get_peft_model_state_dict(model, adapter_name=src_name)
    tgt_weights = {k.replace(src_name, tgt_name): v for k, v in src_weights.items()}
    set_peft_model_state_dict(model, tgt_weights, adapter_name=tgt_name)

def generate_model_outputs(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    gen_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=False 
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], # Explicitly pass mask for left-padding safety
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **gen_kwargs,
        )
    
    raw_logits = torch.stack(outputs.scores, dim=1).cpu()

    transition_scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=False,
    ).cpu()

    if torch.cuda.is_available():
        del inputs
        torch.cuda.empty_cache()

    batch_size = len(prompts)
    num_return_sequences = gen_kwargs.get("num_return_sequences", 1)

    prompt_input_ids = outputs.sequences[:, :prompt_len].cpu()

    generated_sequences = outputs.sequences[:, prompt_len:]
    generated_input_ids = (
        generated_sequences.cpu()
        .contiguous()
        .view(batch_size, num_return_sequences, -1)
    )

    scores_view = transition_scores.contiguous().view(
        batch_size, num_return_sequences, -1
    )

    logits_view = raw_logits.contiguous().view(
        batch_size, num_return_sequences, -1, raw_logits.size(-1)
    )

    return prompt_input_ids, generated_input_ids, scores_view, logits_view
