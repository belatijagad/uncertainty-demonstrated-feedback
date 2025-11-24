import torch


class BaseEstimator:
    def __call__(
        self,
        input_text: str,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0.0)


class RandomEstimator(BaseEstimator):
    def __call__(
        self,
        input_text: str,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.rand(1)


class MSP(BaseEstimator):
    """
    Calculates the uncertainty score using Maximum Softmax Probability (MSP).

    This method computes the total log-probability of a given sequence (`input_ids`)
    based on the model's `logits`. It's a common baseline for uncertainty
    estimation, where a higher score indicates higher model confidence.

    The score is calculated by summing the log-softmax probabilities of the
    actual tokens in the input sequence.
    """

    def __call__(
        self,
        input_text: str,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        return logprobs.sum()
