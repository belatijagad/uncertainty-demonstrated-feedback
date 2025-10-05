class TrainerCallback:
    """Base class for all callbacks."""
    def on_train_begin(self, args, state, control, **kwargs): pass
    def on_train_end(self, args, state, control, **kwargs): pass
    def on_step_begin(self, args, state, control, **kwargs): pass
    def on_step_end(self, args, state, control, **kwargs): pass
    def on_eval_start(self, args, state, control, **kwargs): pass
    def on_eval_end(self, args, state, control, **kwargs): pass
