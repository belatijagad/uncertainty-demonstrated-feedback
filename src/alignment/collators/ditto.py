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
import logging
from tqdm import tqdm
from typing import Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, pipeline

from alignment.collators import BaseDPOCollator

logger = logging.getLogger(__name__)

@dataclass
class DITTODataCollator(BaseDPOCollator):
    """
    Implements the DITTO online data sampling strategy by inheriting tokenization
    and padding mechanics from the BaseDPOCollator.
    """
    train_dataset: Dataset = field(default=None)
    batch_size: int = 24
    model: PreTrainedModel = None
    
    # DITTO-specific parameters
    mode: str = "train"
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 1
    bootstrap_count: int = 10
    
    cache: dict[int, dict[str, list[str]]] = field(default_factory=dict, init=False, repr=False)
    last_sampled_step: int = field(default=0, init=False)

    def resample(self, step: int) -> None:
        """
        Generates new responses using Hugging Face's text generation pipeline
        and caches them for DITTO's dynamic batching.
        """

        if self.model is None:
            logger.error("DITTOCollator's model is None.")
            raise ValueError("Model is not defined.")

        logger.info(f"Resampling data at step {step}")
        self.last_sampled_step = step
        if step not in self.cache: self.cache[step] = {}

        if len(self.train_dataset) == 0 or len(self.train_dataset["prompt"]) == 0:
            logger.warning("Empty dataset provided for DITTO resampling")
            return

        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        prompts = sorted(set(self.train_dataset["prompt"]))

        self.model.eval()
        
        with torch.inference_mode():
            for prompt in tqdm(prompts, desc="Generating Samples", leave=False):
                if prompt not in self.cache[step]:
                    self.cache[step][prompt] = []
                
                responses = generator(
                    prompt,
                    max_new_tokens=self.max_length - self.max_prompt_length,
                    do_sample=True,
                    num_return_sequences=self.bootstrap_count,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True,
                )
                
                for response in responses:
                    generated_text = response['generated_text']
                    self.cache[step][prompt].append(generated_text + self.tokenizer.eos_token)
        
        self.model.train()

    def _get_noisy_pairs(self, prompt: str, step_a: int) -> list[tuple]:
        """
        Generates 'noisy' preference pairs by comparing newer generations
        (from step_a) with older ones. By default, the newer one is 'chosen'.
        """
        noisy_pairs = []
        # Compare against all previous steps
        for step_b in range(step_a):
            if prompt not in self.cache.get(step_b, {}): continue
            
            # The generation from step_a is newer than from step_b
            for newer_rejection in self.cache[step_a][prompt]:
                for older_rejection in self.cache[step_b][prompt]:
                    if newer_rejection != older_rejection:
                        # Default behavior: newer is chosen, older is rejected
                        noisy_pairs.append((prompt, newer_rejection, older_rejection))
        return noisy_pairs
        
    def _get_ditto_sampled_triplets(self, features: list[dict]) -> list[tuple]:
        """Contains the core DITTO sampling logic."""
        expert_samples, replay_samples, noisy_samples = [], [], []

        for feature in features:
            prompt, chosen = feature["prompt"], feature["chosen"]
            
            # 1. Expert pairs (gold chosen vs. latest rejected)
            if self.last_sampled_step in self.cache and prompt in self.cache[self.last_sampled_step]:
                for rejected in self.cache[self.last_sampled_step][prompt]:
                    expert_samples.append((prompt, chosen, rejected))

            # 2. Replay and Noisy pairs
            for step_a in self.cache.keys():
                if prompt not in self.cache.get(step_a, {}): continue
                
                # Replay pairs (gold chosen vs. past rejected)
                if step_a < self.last_sampled_step:
                    for rejected_past in self.cache[step_a][prompt]:
                        replay_samples.append((prompt, chosen, rejected_past))
                
                # Noisy pairs (past rejected vs. older past rejected)
                noisy_samples.extend(self._get_noisy_pairs(prompt, step_a))

        len_superbatch = len(features) * self.rescale_batch
        noisy_subsample = random.sample(noisy_samples, min(len(noisy_samples), round(len_superbatch * self.frac_noisy)))
        expert_subsample = random.sample(expert_samples, min(len(expert_samples), round(len_superbatch * self.frac_expert)))
        replay_subsample = random.sample(replay_samples, min(len(replay_samples), round(len_superbatch * self.frac_replay)))
        
        return expert_subsample + noisy_subsample + replay_subsample

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Constructs a batch by dynamically sampling, then delegates processing."""
        if self.mode == "eval":
            if not features:
                return {}

            tokenized_batch = []
            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]

                tokenized_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length)
                tokenized_chosen = self.tokenizer(prompt + chosen + self.tokenizer.eos_token, truncation=True, max_length=self.max_length)
                tokenized_rejected = self.tokenizer(prompt + rejected + self.tokenizer.eos_token, truncation=True, max_length=self.max_length)

                chosen_labels = tokenized_chosen["input_ids"][:]
                chosen_labels[:len(tokenized_prompt["input_ids"])] = [-100] * len(tokenized_prompt["input_ids"])
                
                rejected_labels = tokenized_rejected["input_ids"][:]
                rejected_labels[:len(tokenized_prompt["input_ids"])] = [-100] * len(tokenized_prompt["input_ids"])

                tokenized_batch.append({
                    "prompt_input_ids": tokenized_prompt["input_ids"],
                    "prompt_attention_mask": tokenized_prompt["attention_mask"],
                    "chosen_input_ids": tokenized_chosen["input_ids"],
                    "chosen_attention_mask": tokenized_chosen["attention_mask"],
                    "chosen_labels": chosen_labels,
                    "rejected_input_ids": tokenized_rejected["input_ids"],
                    "rejected_attention_mask": tokenized_rejected["attention_mask"],
                    "rejected_labels": rejected_labels,
                })

            # TODO: adjust `padding_side` to remove warning
            batch = {}
            for key in tokenized_batch[0].keys():
                if "labels" in key:
                    padding_value = -100
                elif "attention_mask" in key:
                    padding_value = 0
                else:
                    padding_value = self.tokenizer.pad_token_id
                
                sequences = [example[key] for example in tokenized_batch]
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
                
                batch[key] = torch.tensor(padded_sequences, dtype=torch.long)

            return batch

        final_triplets = self._get_ditto_sampled_triplets(features)
        
        if not final_triplets:
            logger.warning("DITTO sampling returned no triplets for this batch. Returning empty dict.")
            
            return {}

        return self.process_and_pad(final_triplets)
