import torch
from typing import Optional
from dataclasses import dataclass
from alignment.collators import DITTODataCollator
from alignment.estimators import BaseEstimator

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
            if prompt not in self.cache[step]: self.cache[step][prompt] = []
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
        