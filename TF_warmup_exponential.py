# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf
import numpy as np

class TF_warmup_exponential(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: float, min_lr: float, num_warmup: int, a: float):
        super(TF_warmup_exponential, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_warmup = num_warmup
        self.a = a

    def __call__(self, step: int) -> tf.Tensor :
        return tf.cond(
            step <= self.num_warmup,
            lambda: self.max_lr / self.num_warmup * step,
            lambda: self.max_lr * np.exp(-self.a * (step - self.num_warmup)) + self.min_lr
        )

    def get_config(self) -> dict:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "num_warmup": self.num_warmup,
            "a": self.a
        }
