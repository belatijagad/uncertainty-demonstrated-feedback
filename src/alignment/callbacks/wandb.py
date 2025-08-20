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
        policy_samples = kwargs.get('policy_samples')
        ref_samples = kwargs.get('ref_samples')
        
        wandb_log_data = {**eval_metrics}
        
        if policy_samples is not None and ref_samples is not None:
            sample_prompts = kwargs.get('sample_prompts', [])
            
            policy_table = wandb.Table(columns=["step", "prompt", "sample"])
            ref_table = wandb.Table(columns=["step", "prompt", "sample"])
            
            for prompt, policy_sample, ref_sample in zip(sample_prompts, policy_samples, ref_samples, strict=True):
                policy_table.add_data(state.global_step, prompt, policy_sample)
                ref_table.add_data(state.global_step, prompt, ref_sample)
                
            wandb_log_data["policy_samples"] = policy_table
            wandb_log_data["reference_samples"] = ref_table
        
        self.wandb_run.log(wandb_log_data, step=state.global_step)