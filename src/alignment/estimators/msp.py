import torch
import torch.nn.functional as F
from alignment.estimators import BaseEstimator
from transformers.modeling_outputs import CausalLMOutputWithPast

class MSP(BaseEstimator):
    """
    Calculates the uncertainty score using Maximum Softmax Probability (MSP).

    This method computes the total log-probability of a given sequence (`input_ids`)
    based on the model's `logits`. It's a common baseline for uncertainty
    estimation, where a higher score indicates higher model confidence.

    The score is calculated by summing the log-softmax probabilities of the
    actual tokens in the input sequence.
    """
    def __call__(self, outputs: CausalLMOutputWithPast, input_ids: torch.Tensor) -> torch.Tensor:
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        actual_log_probs = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        sequence_log_prob = torch.sum(actual_log_probs, dim=1)
        return sequence_log_prob
