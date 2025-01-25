# Copyright 2024 The CAM Lab at ETH Zurich.
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
""""Trainer Library"""

from GenCFD.train.trainers import DenoisingTrainer
from GenCFD.train.training_loop import run as run_training
from GenCFD.train.training_loop import cb as callbacks
from GenCFD.train import train_states

__all__ = ['DenoisingTrainer', 'run_training', 'callbacks', 'train_states']