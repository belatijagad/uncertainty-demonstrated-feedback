# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn
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

import logging

import torch
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
from typing import Any
from dataclasses import dataclass

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
    logger: logging.Logger
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
