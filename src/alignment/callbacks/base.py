class TrainerCallback:
    """Base class for all callbacks."""
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_step_end(self, step: int, metrics: dict, trainer): pass
    def on_eval_end(self, metrics: dict, trainer): pass
