import torch
from abc import ABC, abstractmethod
from transformers.modeling_outputs import CausalLMOutputWithPast

class BaseEstimator(ABC):
    @abstractmethod
    def __call__(self, outputs: CausalLMOutputWithPast, input_ids: torch.Tensor) -> torch.Tensor:
        pass
