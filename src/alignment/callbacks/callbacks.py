from alignment.callbacks.base import TrainerCallback

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

        if self.resample_rate and step_num % self.resample_rate == 0:
            self.collator.resample(step=step_num)

        self.last_step_num = step_num
