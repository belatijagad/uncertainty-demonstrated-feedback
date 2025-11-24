# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.dpo_trainer import DataCollatorForPreference

from scripts.estimator import BaseEstimator
from scripts.utils import generate_model_outputs

@dataclass(init=False)
class DITTOCollator(DataCollatorForPreference):
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 2
    bootstrap_count: int = 10

    cache: dict[int, dict[str, list[dict[str, Any]]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __init__(
        self,
        *args,
        frac_expert: float = 0.7,
        frac_replay: float = 0.2,
        frac_noisy: float = 0.1,
        rescale_batch: int = 2,
        bootstrap_count: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.frac_expert = frac_expert
        self.frac_replay = frac_replay
        self.frac_noisy = frac_noisy
        self.rescale_batch = rescale_batch
        self.bootstrap_count = bootstrap_count
        self.cache = {}
        self.last_sampled_step = 0

    def resample(
        self,
        step: int,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset | IterableDataset,
        config: dict[str, Any],
        estimator: BaseEstimator | None = None,
    ) -> None:
        self.last_sampled_step = step
        self.cache.setdefault(step, {})
        estimator = estimator or BaseEstimator()

        prompts = list(dataset["prompt"])
        num_return_seqs = config.get("num_return_seqs", 1)
        gen_kwargs = {"num_return_sequences": num_return_seqs}
        gen_kwargs.update(config.get("gen_kwargs", {}))

        text_chunks, sequences_view, scores_view = generate_model_outputs(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
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

    def _get_noisy_pairs(self, prompt: str, step_a: int) -> list[tuple]:
        """
        Generates 'noisy' preference pairs by comparing newer generations
        (from step_a) with older ones. By default, the newer one is 'chosen'.
        """
        noisy_pairs = []
        if prompt not in self.cache.get(step_a, {}):
            return noisy_pairs

        current_entries = self.cache[step_a][prompt]
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

                    if newer_score != older_score:
                        chosen, rejected = (
                            (newer, older)
                            if newer_score > older_score
                            else (older, newer)
                        )
                    else:
                        chosen, rejected = newer, older

                    noisy_pairs.append((prompt, chosen, rejected))
        return noisy_pairs

    def _get_sampled_triplets(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> list[list[int] | Any | dict[str, Any]] | None:
        """Contains the core DITTO sampling logic."""
        expert_samples, replay_samples, noisy_samples = [], [], []

        for example in examples:
            assert isinstance(example, dict)
            prompt = example.get("ditto_prompt") or example.get("prompt")
            chosen = example.get("ditto_chosen") or example.get("chosen")

            if prompt is None or chosen is None:
                return None

            # 1. Expert pairs (gold chosen vs. latest rejected)
            if (
                self.last_sampled_step in self.cache
                and prompt in self.cache[self.last_sampled_step]
            ):
                for rejected in self.cache[self.last_sampled_step][prompt]:
                    expert_samples.append((prompt, chosen, rejected))

            # 2. Replay and Noisy pairs
            for step_a in self.cache.keys():
                if prompt not in self.cache.get(step_a, {}):
                    continue

                # Replay pairs (gold chosen vs. past rejected)
                if step_a < self.last_sampled_step:
                    for rejected_past in self.cache[step_a][prompt]:
                        replay_samples.append((prompt, chosen, rejected_past))

                # Noisy pairs (past rejected vs. older past rejected)
                noisy_samples.extend(self._get_noisy_pairs(prompt, step_a))

        len_superbatch = len(examples) * self.rescale_batch
        noisy_subsample = random.sample(
            noisy_samples,
            min(len(noisy_samples), round(len_superbatch * self.frac_noisy)),
        )
        expert_subsample = random.sample(
            expert_samples,
            min(len(expert_samples), round(len_superbatch * self.frac_expert)),
        )
        replay_subsample = random.sample(
            replay_samples,
            min(len(replay_samples), round(len_superbatch * self.frac_replay)),
        )

        return expert_subsample + noisy_subsample + replay_subsample

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        if not self.cache:
            return super().torch_call(examples)
        sampled_triplets = self._get_sampled_triplets(examples)
        if not sampled_triplets:
            return super().torch_call(examples)
        return super().torch_call(sampled_triplets)
