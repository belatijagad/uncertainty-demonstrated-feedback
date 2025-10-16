import logging
from typing import Optional
import wandb
from wandb.sdk.wandb_run import Run
from .base import TrainerCallback

try:
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    RequestOutput = None
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class WandbCallback(TrainerCallback):
    """Callback for logging training metrics to Weights & Biases."""
    
    def __init__(self, wandb_run: Optional[Run] = None):
        self.wandb_run = wandb_run
        self._policy_table = None
        self._policy_table_columns = ["step", "prompt", "sample"]
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics to wandb after each training step."""
        if self.wandb_run and hasattr(state, 'last_metrics'):
            self.wandb_run.log(state.last_metrics, step=state.global_step)
    
    def on_eval_end(self, args, state, control, **kwargs):
        """Log evaluation metrics and samples to wandb, handling both vLLM and str outputs."""
        if not self.wandb_run:
            return

        eval_metrics = kwargs.get("eval_metrics", {})
        policy_samples = kwargs.get("policy_samples")

        wandb_log_data = {f"eval/{k}": v for k, v in eval_metrics.items()}

        if not policy_samples:
            if wandb_log_data:
                self.wandb_run.log(wandb_log_data, step=state.global_step)
            return

        sample_prompts = kwargs.get("sample_prompts", [])

        if self._policy_table is None:
            self._policy_table = wandb.Table(columns=self._policy_table_columns)
        policy_table = self._policy_table

        num_samples_to_log = min(len(sample_prompts), len(policy_samples))
        if num_samples_to_log == 0:
            if wandb_log_data:
                self.wandb_run.log(wandb_log_data, step=state.global_step)
            return

        is_vllm_output = VLLM_AVAILABLE and isinstance(policy_samples[0], RequestOutput)
        is_string_output = isinstance(policy_samples[0], str)

        for idx in range(num_samples_to_log):
            prompt = sample_prompts[idx]
            sample = policy_samples[idx]

            if is_vllm_output:
                generated_text = sample.outputs[0].text
            elif is_string_output:
                generated_text = sample

            policy_table.add_data(state.global_step, prompt, generated_text)

        wandb_log_data["eval/policy_samples"] = policy_table

        if wandb_log_data:
            self.wandb_run.log(wandb_log_data, step=state.global_step)
