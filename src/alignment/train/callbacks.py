import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict

class TrainerCallback:
    """Base class for all callbacks."""
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_step_end(self, step: int, metrics: dict, trainer): pass
    def on_eval_end(self, metrics: dict, trainer): pass
            
class JsonLoggingCallback(TrainerCallback):
    """A callback that saves training and evaluation metrics to a JSON file."""
    def __init__(self, log_dir: str):
        self.log_path = os.path.join(log_dir, "metrics.json")
        self.metrics = defaultdict(list)
        os.makedirs(log_dir, exist_ok=True)

    def on_step_end(self, step: int, metrics: dict, trainer):
        if step % trainer.config.trainer.logging_steps == 0:
            log_str = f"Step {step}: {metrics}"
            tqdm.write(log_str)
            
            serializable_metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in metrics.items()
            }
            self.metrics["train"].append({"step": step, **serializable_metrics})
            self._save_metrics()

    def on_eval_end(self, metrics: dict, trainer):
        log_str = f"Evaluation results @ Step {trainer.step_counter}: {metrics}"
        tqdm.write(log_str)
        
        serializable_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v 
            for k, v in metrics.items()
        }
        self.metrics["eval"].append({"step": trainer.step_counter, **serializable_metrics})
        self._save_metrics()

    def _save_metrics(self):
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
            
class ResampleCallback(TrainerCallback):
    def __init__(self, collator, model, mode, resample_rate, reset_rate):
        
        self.collator = collator
        self.model = model
        self.mode = mode
        self.resample_rate = resample_rate
        self.reset_rate = reset_rate

        self.last_step_num = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)

    def reset_and_resample(self, args, state, control, **kwargs):
        step_num = int(state.global_step)

        if self.last_step_num == step_num:
            return
        
        print("STARTING EPOCH: " + str(step_num))

        if self.resample_rate != None and step_num % self.resample_rate == 0:
            self.collator.resample(step=step_num)

        self.last_step_num = step_num
