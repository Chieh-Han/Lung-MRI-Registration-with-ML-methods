# Copyright 2022 Arnd Koeppe and the CIDS team
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
#
"""Collection of training functions (CIDS with Tensorflow). Part of the CIDS toolbox."""
from kerastuner import HyperParameters


def basic_training_function(hp: HyperParameters):
    learning_rate = hp.Choice(
        "learning_rate", [1e-2, 3e-2, 3e-3, 1e-3, 3e-4, 1e-4], default=3e-3
    )
    batch_size = hp.Fixed("batch_size", 32)
    schedule = {
        "count": [1, 301],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    return schedule


def interdependent_training_function(hp: HyperParameters):
    learning_rate_exp = hp.Float("learning_rate_exponent", -5.0, 0.0, default=-2.0)
    learning_rate = 10.0**learning_rate_exp
    batch_size = hp.Int("batch_size", 32, 128, step=32, default=32)
    schedule = {
        "count": [1, 301],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    return schedule
