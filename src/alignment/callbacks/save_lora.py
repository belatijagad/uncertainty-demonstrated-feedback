from pathlib import Path
from alignment.callbacks import TrainerCallback
from peft import PeftModel

class LoraCallback(TrainerCallback):
    def __init__(self, model: PeftModel, save_path: Path, adapter_name: str):
        self.model = model
        self.save_path = save_path
        self.adapter_name = adapter_name

    def on_eval_start(self, args, state, **kwargs):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.save_path, adapter_name=self.adapter_name)
        