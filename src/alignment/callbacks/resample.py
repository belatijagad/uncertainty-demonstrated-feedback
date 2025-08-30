# DITTO Authors: Omar Shaikh, Michelle S. Lam, Joey Hejna, Yijia Shao, Hyundong Cho, Michael S. Bernstein, Diyi Yang
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PreTrainedModel

from alignment.callbacks import TrainerCallback
from alignment.collators import DITTODataCollator

class ResampleCallback(TrainerCallback):
    def __init__(self, collator: DITTODataCollator, model: PreTrainedModel, resample_rate: int):
        self.collator = collator
        self.model = model
        self.resample_rate = resample_rate

        self.last_step_num = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)

    def reset_and_resample(self, args, state, control, **kwargs):
        step_num = int(state.global_step)

        if self.last_step_num == step_num:
            return
        
        if self.resample_rate and step_num % self.resample_rate == 0:
            self.collator.resample(step=step_num)

        self.last_step_num = step_num
