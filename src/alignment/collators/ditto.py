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
from typing import Any, Optional, Union
from dataclasses import dataclass, field

from torch.utils.data import Dataset
from transformers import PreTrainedModel

try:
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from alignment.collators import BaseDPOCollator
from alignment.utils import batched_generate

logger = logging.getLogger(__name__)

@dataclass
class DITTODataCollator(BaseDPOCollator):
    """
    Implements the DITTO online data sampling strategy by inheriting tokenization
    and padding mechanics from the BaseDPOCollator.
    """
    train_dataset: Dataset = field(default=None)
    batch_size: int = 24
    model: Union[PreTrainedModel | LLM] = None
    
    # DITTO-specific parameters
    mode: str = "train"
    frac_expert: float = 0.7
    frac_replay: float = 0.2
    frac_noisy: float = 0.1
    rescale_batch: int = 1
    bootstrap_count: int = 10
    
    lora_config: Optional[dict] = None
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

        prompts = list(dict.fromkeys(self.train_dataset["prompt"]))

        lora_request = (
            None if self.lora_adapter_path is None or not VLLM_AVAILABLE
            else LoRARequest("ditto", self.lora_config.get("r"), str(self.lora_adapter_path))
            )

        responses = batched_generate(
            prompts,
            max_new_tokens=self.max_length - self.max_prompt_length,
            model=self.model,
            tokenizer=self.tokenizer,
            device=None if isinstance(self.model, LLM) else self.model.device,
            lora_request=lora_request,
            num_return_sequences=self.bootstrap_count,
            do_sample=True,
            disable_peft_adapter=False,
            adapter_name="ditto",
        )

        for prompt, generations in zip(prompts, responses, strict=True):
            cache_slot = self.cache[step].setdefault(prompt, [])
            for generated_text in generations:
                if not generated_text.endswith(self.tokenizer.eos_token):
                    generated_text = generated_text + self.tokenizer.eos_token
                cache_slot.append(generated_text)

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
