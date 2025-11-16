import torch
from typing import Optional
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

from vllm import LLM
from vllm.lora.request import LoRARequest

from alignment.collators import DITTODataCollator
from alignment.estimators import BaseEstimator
from alignment.utils import batched_generate
# TODO: UDITTO

@dataclass
class UDITTODataCollator(DITTODataCollator):
    """
    Implements Uncertainty-DITTO by overriding the parent DITTODataCollator.

    It changes the preference pairing logic from being based on recency
    to being based on an uncertainty score (average log probability).
    """
    higher_score_is_better: bool = False
    estimator: Optional[BaseEstimator] = None
    threshold: Optional[float] = None

    def resample(self, step: int) -> None:
        """
        Generates new responses using Hugging Face's text generation pipeline
        and caches them for DITTO's dynamic batching.
        """

        if self.model is None:
            self.logger.error("DITTOCollator's model is None.")
            raise ValueError("Model is not defined.")
        if isinstance(self.model, LLM):
            self.logger.error("UDITTO requires a Hugging Face model for uncertainty scoring.")
            raise ValueError("UDITTODataCollator only supports Hugging Face models for resampling.")

        self.logger.info(f"Resampling data at step {step}")
        self.last_sampled_step = step
        if step not in self.cache:
            self.cache[step] = {}

        if len(self.train_dataset) == 0 or len(self.train_dataset["prompt"]) == 0:
            self.logger.warning("Empty dataset provided for DITTO resampling")
            return

        prompts = list(dict.fromkeys(self.train_dataset["prompt"]))

        lora_request = (
            None if self.lora_adapter_path is None
            else LoRARequest("ditto", 1, str(self.lora_adapter_path))
            )
        gen_kwargs = {
            "max_new_tokens": self.max_length - self.max_prompt_length,
            "num_return_sequences": self.bootstrap_count,
            "do_sample": True,
            "return_dict_in_generate": True,
            "output_logits": True,
        }

        responses = batched_generate(
            prompts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=None if isinstance(self.model, LLM) else self.model.device,
            lora_request=lora_request,
            disable_peft_adapter=False,
            adapter_name="ditto",
            gen_kwargs=gen_kwargs,
        )

        model_device = next(self.model.parameters()).device
        flattened_sequences: list[torch.Tensor] = []

        for _prompt, generations in zip(prompts, responses, strict=True):
            generation_list = generations if isinstance(generations, list) else [generations]
            for gen in generation_list:
                token_ids = gen.get("generated_token_ids")
                if token_ids is None:
                    continue
                if not isinstance(token_ids, torch.Tensor):
                    token_ids = torch.tensor(token_ids, dtype=torch.long)
                flattened_sequences.append(token_ids.to(model_device))

        if not flattened_sequences:
            self.logger.warning("UDITTO resample produced no generations to cache.")
            return

        generated_ids = pad_sequence(
            flattened_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        self._process_and_cache_generations(
            generated_ids=generated_ids,
            prompts=prompts,
            prompt_len=self.max_prompt_length,
            step=step,
        )

    def _process_and_cache_generations(self, generated_ids: torch.Tensor, prompts: list[str], prompt_len: int, step: int):
        """
        Overrides the base DITTO method.

        Instead of caching only text, this version calculates an uncertainty score
        (average log probability) for each generation and caches (text, score) tuples.

        Cache Structure:
            The `self.cache` attribute is a nested dictionary with the following
            structure: `cache[step][prompt]`.
            - `step` (int): The current iteration or generation step.
            - `prompt` (str): The original input prompt text.
            - The value is a list of `(response, score)` tuples, storing each
            generated candidate and its associated uncertainty score.
        """
        labels = generated_ids.clone()
        labels[:, :prompt_len] = self.label_pad_token_id

        outputs = self.model(input_ids=generated_ids, attention_mask=(generated_ids != self.tokenizer.pad_token_id))
        
        # Calculate scores using the log-probability of the generated tokens
        scores = self.estimator(outputs, generated_ids, self.nli_model)
        
        response_ids = generated_ids[:, prompt_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        response_idx = 0
        for prompt in prompts:
            if prompt not in self.cache[step]:
                self.cache[step][prompt] = []
            for _ in range(self.bootstrap_count):
                score = scores[response_idx].item()

                # Threshold based rejection sampling
                if self.threshold:
                    should_reject = (
                        (self.higher_score_is_better and score < self.threshold)
                        or (not self.higher_score_is_better and score > self.threshold)
                    )

                    if not should_reject:
                        self.cache[step][prompt].append(
                            (
                                responses[response_idx] + self.tokenizer.eos_token, 
                                score
                            )
                        )
                else:
                    self.cache[step][prompt].append(
                        (
                            responses[response_idx] + self.tokenizer.eos_token, 
                            score
                        )
                    )
                response_idx += 1
    
    def _get_noisy_pairs(self, prompt: str, step_a: int) -> list[tuple]:
        """
        Overrides the base DITTO method.

        Creates 'noisy' preference pairs by comparing the scores of generations
        from different historical steps, rather than just their recency.
        """
        noisy_pairs = []
        
        # Compare against all previous steps
        for step_b in range(step_a):
            if prompt not in self.cache.get(step_a, {}) or prompt not in self.cache.get(step_b, {}):
                continue
            
            # The cache now contains (text, score) tuples
            for text_a, score_a in self.cache[step_a][prompt]:
                for text_b, score_b in self.cache[step_b][prompt]:
                    
                    # Determine chosen/rejected by comparing their scores
                    if (self.higher_score_is_better and score_a > score_b) or \
                       (not self.higher_score_is_better and score_a < score_b):
                        chosen, rejected = text_a, text_b
                    else:
                        chosen, rejected = text_b, text_a

                    if chosen != rejected:
                        noisy_pairs.append((prompt, chosen, rejected))
                        
        return noisy_pairs
        