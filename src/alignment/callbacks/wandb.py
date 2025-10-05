import logging
from typing import Optional
import wandb
from wandb.sdk.wandb_run import Run
from .base import TrainerCallback

logger = logging.getLogger(__name__)


class WandbCallback(TrainerCallback):
    """Callback for logging training metrics to Weights & Biases."""
    
    def __init__(self, wandb_run: Optional[Run] = None):
        self.wandb_run = wandb_run
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics to wandb after each training step."""
        if self.wandb_run and hasattr(state, 'last_metrics'):
            self.wandb_run.log(state.last_metrics, step=state.global_step)
    
    def on_eval_end(self, args, state, control, **kwargs):
        """Log evaluation metrics and samples to wandb."""
        if not self.wandb_run:
            return
            
        eval_metrics = kwargs.get('eval_metrics', {})
        policy_samples = kwargs.get('policy_samples') or []

        wandb_log_data = {**eval_metrics}

        if policy_samples:
            sample_prompts = kwargs.get('sample_prompts') or []
            policy_table = wandb.Table(columns=["step", "prompt", "sample"])
            min_len = min(len(sample_prompts), len(policy_samples))

            if min_len < len(policy_samples):
                logger.warning(
                    "Number of policy samples (%d) exceeds available prompts (%d). "
                    "Only logging the first %d pairs.",
                    len(policy_samples),
                    len(sample_prompts),
                    min_len,
                )

            for idx in range(min_len):
                policy_table.add_data(state.global_step, sample_prompts[idx], policy_samples[idx])
            
            wandb_log_data["policy_samples"] = policy_table
        
        self.wandb_run.log(wandb_log_data, step=state.global_step)