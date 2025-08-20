import logging
from .base import TrainerCallback

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Callback for logging training metrics to the console."""
    
    def __init__(self, logging_steps: int = 500):
        self.logging_steps = logging_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log metrics to console at specified intervals."""
        if (state.global_step) % self.logging_steps == 0:
            metrics = getattr(state, "last_metrics", {})
            if metrics:
                metrics_str = " | ".join([
                    f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                    for k, v in metrics.items()
                ])
                report = f"Step {state.global_step:>6} | {metrics_str}"
                logger.info(report)
    
    def on_eval_end(self, args, state, control, **kwargs):
        """Log evaluation results to console."""
        eval_metrics = kwargs.get("eval_metrics", {})
        if eval_metrics:
            eval_metrics_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                for k, v in eval_metrics.items()
            ])
            logger.info(f"Step {state.global_step} Evaluation | {eval_metrics_str}")
