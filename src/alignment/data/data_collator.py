# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import Optional, Union, Any
from dataclasses import dataclass

@dataclass
class SFTDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    
    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer(
            [f["text"] for f in features],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        
        assistant_token_id = self.tokenizer.additional_special_tokens_ids[1]
        
        for i in range(len(features)):
            labels = batch["labels"][i]
            assistant_indices = (labels == assistant_token_id).nonzero(as_tuple=True)[0]
            if len(assistant_indices) > 0:
                last_assistant_idx = assistant_indices[-1]
                labels[:last_assistant_idx + 1] = -100

        batch["labels"][batch["attention_mask"] == 0] = -100
        
        return batch

@dataclass
class DPODataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    max_target_length: Optional[int] = None
    train_dataset = None
                
    def build_tokenized_answer(self, prompt, answer):
        # Tokenize the full sequence once
        full_tokenized = self.tokenizer(answer, add_special_tokens=False)
        full_input_ids = full_tokenized["input_ids"]

        # Tokenize the prompt separately to find the boundary
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        
        response_token_ids_start_idx = len(prompt_input_ids)
        
        # Handle tokenizer merge issue
        if prompt_input_ids != full_input_ids[:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1
            
        # Recalculate prompt and answer splits with the correct index
        prompt_input_ids = full_input_ids[:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        answer_input_ids = full_input_ids[response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
            
    def tokenize_row(self, prompt, chosen, rejected):
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str): raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str): raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str): raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # bos and eos are already added.

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch
        
    def collate(self, batch):
        padded_batch = {}
        
        for key in batch[0].keys():
            if key.endswith(("_input_ids", "_attention_mask", "_labels")):
                
                if key.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif key.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    padding_value = 0
                
                sequences = [torch.LongTensor(sample[key]) for sample in batch]
                padded_batch[key] = pad_sequence(
                    sequences, 
                    batch_first=True, 
                    padding_value=padding_value, 
                    padding_side='left'
                )
                
            else:
                padded_batch[key] = [sample[key] for sample in batch]
                
        return padded_batch

    def _format_chat_template(self, turns: list[dict]):
        text = ""
        for turn in turns:
            if turn['role'] == 'user':
                text += f"<|prompter|>{turn['content']}<|endoftext|>"
            elif turn['role'] == 'assistant':
                text += f"<|assistant|>{turn['content']}<|endoftext|>"
        return text
            
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        tokenized_batch = []
        
        for feature in features:
            prompt_turns = feature["chosen"][:-1]
            
            prompt_str = self._format_chat_template(prompt_turns) + "<|assistant|>"

            chosen_response = feature["chosen"][-1]["content"] + "<|endoftext|>"
            rejected_response = feature["rejected"][-1]["content"] + "<|endoftext|>"
            
            batch_element = self.tokenize_row(prompt_str, prompt_str + chosen_response, prompt_str + rejected_response)
            tokenized_batch.append(batch_element)
            
        collated_batch = self.collate(tokenized_batch)
        
        return collated_batch
