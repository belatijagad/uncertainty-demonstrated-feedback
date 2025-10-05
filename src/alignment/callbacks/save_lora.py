from pathlib import Path
from alignment.callbacks import TrainerCallback
from peft import PeftModel

class LoraCallback(TrainerCallback):
    def __init__(self, model: PeftModel):
        self.model = model

    def on_eval_start(self, args, state, control, **kwargs):
        save_path = Path(__file__).parent / "temp"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        