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
from typing import Any, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, pipeline
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
    
    vllm_model: Optional[LLM] = field(default=None)
    lora_adapter_path: Optional[str] = field(default=None)
    
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

        prompts = sorted(set(self.train_dataset["prompt"]))

        self.model.eval()

        if self.vllm_model is not None:
            if not self.lora_adapter_path:
                raise ValueError("lora_adapter_path must be set when using vllm_model.")

            logger.info(f"Resampling with vLLM using adapter: {self.lora_adapter_path}")

            self.vllm_model.wake_up()

            sampling_params = SamplingParams(
                n=self.bootstrap_count,
                temperature=1.0,
                max_tokens=self.max_length - self.max_prompt_length,
            )
            
            lora_request = LoRARequest(
                lora_name="ditto",
                lora_int_id=1,
                lora_local_path=self.lora_adapter_path
            )

            outputs = self.vllm_model.generate(
                prompts,
                sampling_params,
                lora_request=lora_request
            )

            for output in tqdm(outputs, desc="Processing vLLM Samples", leave=False):
                prompt = output.prompt
                if prompt not in self.cache[step]:
                    self.cache[step][prompt] = []
                
                generated_texts = [o.text + self.tokenizer.eos_token for o in output.outputs]
                self.cache[step][prompt].extend(generated_texts)
            
            self.vllm_model.sleep(level=2)
        else:
            self.model.eval()
            generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
            )
            
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
            eval_triplets = [
                (f["prompt"], f["chosen"], f["rejected"]) for f in features
            ]
            return self.process_and_pad(eval_triplets)

        final_triplets = self._get_ditto_sampled_triplets(features)
        
        if not final_triplets:
            logger.warning("DITTO sampling returned no triplets for this batch. Returning empty dict.")
            
            return {}

        return self.process_and_pad(final_triplets)
