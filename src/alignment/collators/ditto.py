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
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import PreTrainedModel

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
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 1
    bootstrap_count: int = 10
    
    cache: dict[int, dict[str, list[str]]] = field(default_factory=dict, init=False, repr=False)
    last_sampled_step: int = field(default=0, init=False)

    def _process_and_cache_generations(self, generated_ids, prompts, prompt_len, step):
        """
        Decodes and caches generated text.
        
        Cache Structure:
            The `self.cache` attribute is a nested dictionary with the following
            structure: `cache[step][prompt]`.
            - `step` (int): The current iteration or generation step.
            - `prompt` (str): The original input prompt text.
            - The value is a list of `response` tuples, storing each
              generated candidate.
        """

        # Decode only the newly generated part of the sequences
        response_ids = generated_ids[:, prompt_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        # Populate the cache with the generated text for each prompt
        response_idx = 0
        for prompt in prompts:
            if prompt not in self.cache[step]: self.cache[step][prompt] = []
            for _ in range(self.bootstrap_count):
                self.cache[step][prompt].append(responses[response_idx] + self.tokenizer.eos_token)
                response_idx += 1

    def resample(self, step: int) -> None:
        """
        Generates new responses using direct PyTorch model.generate calls
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

        self.model.eval()
        with torch.inference_mode():
            # 1. Get all unique prompts and sort them for a consistent order
            prompts = sorted(self.train_dataset["prompt"])
            
            # 2. Tokenize all prompts for generation
            tokenized_prompts = self.tokenizer(
                prompts, return_tensors="pt", padding=True, 
                padding_side='left', truncation=True, max_length=self.max_prompt_length
            )

            # 3. Create a DataLoader to process prompts in batches for memory efficiency
            dataset = TensorDataset(tokenized_prompts.input_ids, tokenized_prompts.attention_mask)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)

            for i, (input_ids_batch, attention_mask_batch) in enumerate(tqdm(dataloader, desc="Generating Samples")):
                input_ids_batch = input_ids_batch.to(self.model.device)
                attention_mask_batch = attention_mask_batch.to(self.model.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    max_new_tokens=self.max_length - self.max_prompt_length,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=self.bootstrap_count,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                start_index = i * self.batch_size
                end_index = start_index + len(input_ids_batch)
                prompts_in_batch = prompts[start_index:end_index]
                prompt_len = input_ids_batch.shape[1]
                
                self._process_and_cache_generations(generated_ids, prompts_in_batch, prompt_len, step)
            
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
        final_triplets = self._get_ditto_sampled_triplets(features)
        
        if not final_triplets:
            logger.warning("DITTO sampling returned no triplets for this batch. Returning empty dict.")
            
            return {}

        return self.process_and_pad(final_triplets)
