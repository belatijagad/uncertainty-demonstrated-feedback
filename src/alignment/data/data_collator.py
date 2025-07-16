# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang 2024
# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
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

import torch
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Any
from dataclasses import dataclass, field
from tqdm import tqdm
import random

@dataclass
class SFTDataCollator:
    """Data collator that applies the tokenizer's chat template for SFT."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(self, features: list[dict]) -> dict:
        formatted_texts = [self.tokenizer.apply_chat_template(f["messages"], tokenize=False, add_generation_prompt=False) for f in features]
        
        batch = self.tokenizer(
            formatted_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        batch["labels"] = batch["input_ids"].clone()
        for i, feature in enumerate(features):
            prompt_turns = feature["messages"][:-1]
            prompt_str = self.tokenizer.apply_chat_template(prompt_turns, tokenize=False, add_generation_prompt=True)
            prompt_len = len(self.tokenizer(prompt_str, add_special_tokens=True).input_ids)
            
            batch["labels"][i, :prompt_len] = -100

        batch["labels"][batch["attention_mask"] == 0] = -100
        
        return batch

@dataclass
class BaseDPOCollator:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """
    """Base class for DPO-style collators that handles all tokenization and padding."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    max_prompt_length: int
    label_pad_token_id: int = -100

    def process_and_pad(self, list_of_triplets: list[tuple]) -> dict[str, Any]:
        """Takes a list of (prompt, chosen, rejected) tuples and creates a padded batch."""
        tokenized_batch = [self._tokenize_row(p, c, r) for p, c, r in list_of_triplets]
        return self._collate_batch(tokenized_batch)

    def _build_tokenized_answer(self, prompt: str, full_answer: str) -> dict:
        """Helper to robustly find the split point between prompt and answer."""
        full_tok = self.tokenizer(full_answer, add_special_tokens=False)
        prompt_tok = self.tokenizer(prompt, add_special_tokens=False)
        
        start_idx = len(prompt_tok["input_ids"])
        # Handle tokenizer merging issues
        if prompt_tok["input_ids"] != full_tok["input_ids"][:start_idx]:
            start_idx -= 1
            
        return {
            "prompt_input_ids": full_tok["input_ids"][:start_idx],
            "input_ids": full_tok["input_ids"][start_idx:],
        }

    def _tokenize_row(self, prompt: str, chosen: str, rejected: str) -> dict:
        """
        Tokenizes a single data point for DPO, creating input IDs, attention
        masks, and labels for both the chosen and rejected sequences.

        This function handles the truncation of prompts and responses to
        ensure the final sequences fit within the model's `max_length`.

        Args:
            prompt (str): The conversational context or prompt.
            chosen (str): The preferred or "chosen" response text.
            rejected (str): The dispreferred or "rejected" response text.

        Returns:
            dict: A dictionary containing the tokenized and prepared inputs
                  for a single DPO example. The keys are:
                  - `chosen_input_ids`: Token IDs for the full chosen sequence.
                  - `chosen_attention_mask`: Attention mask for the chosen sequence.
                  - `chosen_labels`: Labels for the chosen sequence, with prompt tokens masked with -100.
                  - `rejected_input_ids`: Token IDs for the full rejected sequence.
                  - `rejected_attention_mask`: Attention mask for the rejected sequence.
                  - `rejected_labels`: Labels for the rejected sequence, with prompt tokens masked with -100.
        """
        batch_element = {}
        
        prompt_tokens = {"prompt_input_ids": self.tokenizer(prompt, add_special_tokens=False)["input_ids"]}
        chosen_tokens = self._build_tokenized_answer(prompt, prompt + chosen)
        rejected_tokens = self._build_tokenized_answer(prompt, prompt + rejected)

        # Truncate prompt if necessary
        prompt_len = min(len(chosen_tokens["prompt_input_ids"]), len(rejected_tokens["prompt_input_ids"]))
        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len]

        # Truncate responses if necessary
        longer_response = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
        max_response_len = self.max_length - self.max_prompt_length
        for tokens in [chosen_tokens, rejected_tokens]:
            if len(tokens["prompt_input_ids"]) + longer_response > self.max_length:
                tokens["input_ids"] = tokens["input_ids"][:max_response_len]

        # Combine and create labels for chosen and rejected
        for type_key, p_toks, r_toks in [("chosen", prompt_tokens, chosen_tokens), ("rejected", prompt_tokens, rejected_tokens)]:
            input_ids = p_toks["prompt_input_ids"] + r_toks["input_ids"]
            labels = ([-100] * len(p_toks["prompt_input_ids"])) + r_toks["input_ids"]
            
            batch_element[f"{type_key}_input_ids"] = input_ids
            batch_element[f"{type_key}_labels"] = labels
            batch_element[f"{type_key}_attention_mask"] = [1] * len(input_ids)
            
        return batch_element

    def _collate_batch(self, tokenized_batch: list[dict]) -> dict:
        """Pads a batch of tokenized examples."""
        padded_batch = {}
        for key in tokenized_batch[0].keys():
            if "prompt" in key: continue # Skip prompts for now, handled in main sequences
                
            padding_value = 0
            if key.endswith("input_ids"): padding_value = self.tokenizer.pad_token_id
            elif key.endswith("labels"): padding_value = self.label_pad_token_id
            
            to_pad = [torch.LongTensor(ex[key]) for ex in tokenized_batch]
            padded_batch[key] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            
        return padded_batch
    
@dataclass
class OfflineDPODataCollator(BaseDPOCollator):
    """
    Collator for standard offline DPO. It reads (prompt, chosen, rejected)
    from a static dataset and uses the base class to process them.
    """
    def __call__(self, features: list[dict]) -> dict:
        triplets = []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            triplets.append((prompt, chosen, rejected))
            
        return self.process_and_pad(triplets)
    
@dataclass
class DITTODataCollator(BaseDPOCollator):
    """
    Implements the DITTO online data sampling strategy by inheriting tokenization
    and padding mechanics from the BaseDPOCollator.
    """
    train_dataset: Dataset
    batch_size: int = 8
    
    # DITTO-specific parameters
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 3
    bootstrap_count: int = 10
    
    cache: dict[int, dict[str, list[str]]] = field(default_factory=dict, init=False, repr=False)
    last_sampled_step: int = field(default=0, init=False)

    def _process_and_cache_generations(self, generated_ids, prompts, prompt_len, step):
        """Decodes and caches generated text."""
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

    def resample(self, step: int):
        """
        Generates new responses using direct PyTorch model.generate calls
        and caches them for DITTO's dynamic batching.
        """
        print(f"--- DITTO: Resampling data at step {step} ---")
        self.last_sampled_step = step
        if step not in self.cache: self.cache[step] = {}

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
            print("Warning: DITTO sampling returned no triplets for this batch. Returning empty dict.")
            return {}

        return self.process_and_pad(final_triplets)
        