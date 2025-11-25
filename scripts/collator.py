# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.

import random
import torch
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.dpo_trainer import DataCollatorForPreference

from scripts.estimator import BaseEstimator
from scripts.utils import generate_model_outputs

@dataclass
class DITTOCollator(DataCollatorForPreference):
    """
    DITTO DataCollator that handles dynamic sampling of generated responses 
    (Expert, Replay, Noisy) and correctly formats them for DPO training.
    """
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 2

    cache: dict[int, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict, init=False, repr=False
    )
    last_sampled_step: int = field(default=0, init=False)

    def __init__(self, *args, **kwargs):
        self.frac_expert = kwargs.pop("frac_expert", 0.7)
        self.frac_replay = kwargs.pop("frac_replay", 0.2)
        self.frac_noisy = kwargs.pop("frac_noisy", 0.1)
        self.rescale_batch = kwargs.pop("rescale_batch", 2)
        self.tokenizer = kwargs.pop("tokenizer")
        self.resample_rate = kwargs.pop("resample_rate")
        self.gen_kwargs = kwargs.pop("gen_kwargs")
        
        super().__init__(*args, **kwargs)
        
        self.cache = {}
        self.last_sampled_step = 0

    def resample(
        self,
        step: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset | IterableDataset,
        estimator: BaseEstimator | None = None,
    ) -> None:
        """
        Generates new responses from the model and updates the cache.
        """
        self.last_sampled_step = step
        self.cache.setdefault(step, {})

        estimator = estimator or BaseEstimator()
        prompts = list(dataset["prompt"])
        
        text_chunks, sequences_view, scores_view = generate_model_outputs(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=self.gen_kwargs,
        )

        for prompt, texts, scores, seq_chunk in zip(
            prompts, text_chunks, scores_view, sequences_view, strict=True
        ):
            cache_slot = self.cache[step].setdefault(prompt, [])

            for text, seq, score_seq in zip(texts, seq_chunk, scores, strict=True):
                if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token

                cache_slot.append(
                    {
                        "generated_text": text,
                        "score": estimator(text, seq, score_seq),
                    }
                )
        
        print(f"Resampling complete for step {step}. Cache updated.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_noisy_pairs(self, prompt: str, step_a: int) -> list[tuple]:
        """
        Generates 'noisy' preference pairs by comparing scores of different generations.
        """
        noisy_pairs = []
        if prompt not in self.cache.get(step_a, {}):
            return noisy_pairs

        current_entries = self.cache[step_a][prompt]
        
        # Look at all previous steps
        for step_b in range(step_a):
            past_entries = self.cache.get(step_b, {}).get(prompt)
            if not past_entries:
                continue

            for newer in current_entries:
                for older in past_entries:
                    if newer["generated_text"] == older["generated_text"]:
                        continue

                    newer_score = float(newer["score"])
                    older_score = float(older["score"])

                    # Score-based selection: Better score is 'chosen'
                    if newer_score != older_score:
                        chosen, rejected = (
                            (newer, older)
                            if newer_score > older_score
                            else (older, newer)
                        )
                        # Extract just the text
                        noisy_pairs.append((prompt, chosen["generated_text"], rejected["generated_text"]))
        
        return noisy_pairs

    def _build_dpo_batch_element(self, prompt: str, chosen: str, rejected: str) -> dict[str, Any]:
        """
        Manually tokenizes and creates LABELS for the DPO trainer.
        Sets the prompt labels to -100 so the model doesn't calculate loss on the prompt.
        """
        # 1. Tokenize components
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_ids = self.tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_ids = self.tokenizer(rejected, add_special_tokens=False)["input_ids"]

        # 2. Add EOS if missing
        if self.tokenizer.eos_token_id is not None:
             if not chosen_ids or chosen_ids[-1] != self.tokenizer.eos_token_id:
                 chosen_ids.append(self.tokenizer.eos_token_id)
             if not rejected_ids or rejected_ids[-1] != self.tokenizer.eos_token_id:
                 rejected_ids.append(self.tokenizer.eos_token_id)

        # 3. Concatenate (Prompt + Response)
        full_chosen_ids = prompt_ids + chosen_ids
        full_rejected_ids = prompt_ids + rejected_ids

        # 4. Create Labels (Masking the prompt)
        prompt_len = len(prompt_ids)
        chosen_labels = [-100] * prompt_len + chosen_ids
        rejected_labels = [-100] * prompt_len + rejected_ids

        # 5. Return dict compatible with parent torch_call
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": [1] * len(prompt_ids),
            
            "chosen_input_ids": full_chosen_ids,
            "chosen_attention_mask": [1] * len(full_chosen_ids),
            "chosen_labels": chosen_labels,
            
            "rejected_input_ids": full_rejected_ids,
            "rejected_attention_mask": [1] * len(full_rejected_ids),
            "rejected_labels": rejected_labels,
        }

    def _get_sampled_triplets(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Decodes input IDs, queries the cache for DITTO samples, and re-tokenizes the result.
        """
        new_batch_examples = []

        for example in examples:
            prompt_text = self.tokenizer.decode(example["prompt_input_ids"], skip_special_tokens=True)
            
            # Careful: chosen_input_ids often includes the prompt. We assume prompt is consistent.
            full_chosen_text = self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=True)
            # Strip prompt roughly if needed, or just use full text as the "chosen" response string
            # For simplicity, we use the decoded text as is, assuming the model handles it.
            chosen_text = full_chosen_text[len(prompt_text):] if full_chosen_text.startswith(prompt_text) else full_chosen_text

            expert_samples = []
            replay_samples = []
            noisy_samples = []

            # Expert pairs
            if (self.last_sampled_step in self.cache and prompt_text in self.cache[self.last_sampled_step]):
                for rejected_data in self.cache[self.last_sampled_step][prompt_text]:
                    expert_samples.append((prompt_text, chosen_text, rejected_data["generated_text"]))

            # Iterate history
            for step_a in self.cache.keys():
                if prompt_text not in self.cache.get(step_a, {}): continue
                
                # Replay pairs
                if step_a < self.last_sampled_step:
                    for rejected_data in self.cache[step_a][prompt_text]:
                        replay_samples.append((prompt_text, chosen_text, rejected_data["generated_text"]))

                # Noisy pairs
                noisy_samples.extend(self._get_noisy_pairs(prompt_text, step_a))

            len_superbatch = len(examples) * self.rescale_batch
            
            def safe_sample(source, frac):
                count = min(len(source), int(len_superbatch * frac))
                return random.sample(source, count) if count > 0 else []

            selected_triplets = (
                safe_sample(expert_samples, self.frac_expert) +
                safe_sample(replay_samples, self.frac_replay) +
                safe_sample(noisy_samples, self.frac_noisy)
            )

            if not selected_triplets:
                new_batch_examples.append(example)
                continue

            for p_txt, c_txt, r_txt in selected_triplets:
                batch_element = self._build_dpo_batch_element(p_txt, c_txt, r_txt)
                new_batch_examples.append(batch_element)

        return new_batch_examples

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # If cache is empty, just behave like a normal DPO collator
        if not self.cache:
            return super().torch_call(examples)
        
        # Get DITTO samples (returns a list of dicts with IDs and Labels)
        sampled_batch = self._get_sampled_triplets(examples)
        
        # If sampling failed completely, fall back to original examples
        if not sampled_batch:
            return super().torch_call(examples)
            
        return super().torch_call(sampled_batch)