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
"""Tensorflow callbacks for CIDS. Part of the CIDS toolbox.

    Classes:
        CIDSCheckpoint: Saves CIDSModel at regular intervals during training
        FreezeControl: Freezes online normalization parameters after defined epoch
        StepProgressCallback:   Print training progress for each batch
        EpochProgressCallback:  Print training progress for each epoch
"""
import os
import sys
from collections import OrderedDict

import numpy as np
from tensorflow.keras.callbacks import Callback
from tqdm.auto import trange


class CIDSCheckpoint(Callback):

    DEBUG = False

    def __init__(
        self,
        cids_model,
        monitor="val_loss",
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq=100,
    ):
        self.cids_model = cids_model
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0

        if mode not in ["auto", "min", "max"]:
            self.cids_model.warn(
                f"Model checkpoint mode {mode:s} is unknown. Falling back to auto mode."
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf
        self.set_model(getattr(self.cids_model, "core_model"))

    @property
    def best(self):
        return self.cids_model._best_monitor

    @best.setter
    def best(self, value):
        self.cids_model._best_monitor = value

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.save_freq:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    self.cids_model.warn(
                        f"Can save best model only with {self.monitor:s} available. "
                        + "Skipping.",
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.cids_model.log(
                                f"Epoch {epoch + 1:05d}: {self.monitor:s} improved "
                                + f"from {self.best:0.5f} to {current:0.5f}. "
                                + f" Saving model to {self.cids_model.count:d}."
                            )
                        self.best = current
                        self.cids_model.save(self.cids_model.count)
                    else:
                        if self.verbose > 0:
                            self.cids_model.log(
                                f"Epoch {epoch + 1:05d}: {self.monitor:s} did not "
                                + f"improve from {self.best:0.5f}. "
                            )
            else:
                if self.verbose > 0:
                    self.cids_model.log(
                        f"Epoch {epoch + 1:05d}: saving model "
                        + f"to {self.cids_model.count:s}."
                    )
                self.cids_model.save(self.cids_model.count)


class FreezeControl(Callback):

    DEBUG = False

    def __init__(self, cids_model):
        self.cids_model = cids_model
        self._old_freeze = False
        super().__init__()

    def on_test_begin(self, logs=None):
        try:
            self._old_freeze = self.cids_model.freeze.numpy()
        except AttributeError:
            self._old_freeze = self.cids_model.freeze
        self.cids_model.freeze = True

    def on_test_end(self, logs=None):
        self.cids_model.freeze = self._old_freeze

    def on_predict_begin(self, logs=None):
        try:
            self._old_freeze = self.cids_model.freeze.numpy()
        except AttributeError:
            self._old_freeze = self.cids_model.freeze
        self.cids_model.freeze = True

    def on_predict_end(self, logs=None):
        self.cids_model.freeze = self._old_freeze


class StepProgressCallback(Callback):

    DEBUG = False

    def __init__(self, cids_model, num_steps, phase):
        self.cids_model = cids_model
        self.num_steps = num_steps - self.cids_model.count
        self.phase = phase
        self.progbar = None
        self.postfix = OrderedDict()
        self._step_format_str = "{:07d}"
        self._loss_format_str = "{:.3e}"
        super().__init__()

    def on_train_begin(self, logs=None):
        self.postfix = OrderedDict()
        self.postfix["step"] = self._step_format_str.format(0)
        if self.cids_model.VERBOSITY:
            self.tqdm_target = self.cids_model.stream_to_logger()
        else:
            self.tqdm_target = open(  # pylint: disable=consider-using-with
                os.devnull, "w", encoding="utf8"
            )
        self.progbar = trange(
            self.num_steps,
            miniters=self.cids_model.report_freq,
            file=self.tqdm_target,
            dynamic_ncols=True,
            desc=f"Training phase {self.phase:d}",
        )

    def on_train_batch_end(self, batch, logs=None):
        # Increase step
        self.cids_model.count += 1
        # Get data
        step = self.cids_model.count
        # Set postfix
        if step % self.cids_model.report_freq == 0:
            update_dict = OrderedDict(
                [
                    (k, logs[k])
                    for k in sorted(logs.keys())
                    if k not in ["size", "val_batch", "batch", "val_batch"]
                ]
            )
            self.postfix.update(update_dict)

    def on_test_batch_end(self, batch, logs=None):
        # Get data
        step = self.cids_model.count
        initial_step = self.cids_model.initial_count
        if step % self.cids_model.report_freq == 0:
            # Set postfix
            self.postfix["step"] = self._step_format_str.format(step)
            sorted_logs_keys = sorted(logs.keys())
            update_dict = OrderedDict(
                [
                    ("val_" + k, logs[k])
                    for k in sorted_logs_keys
                    if k not in ["size", "val_batch", "batch", "val_batch"]
                ]
            )
            self.postfix.update(update_dict)
            # Update progress bar
            self.progbar.set_postfix(self.postfix, refresh=False)
            self.progbar.update(step - initial_step - self.progbar.n)

    def on_train_end(self, logs=None):
        if self.progbar is not None:
            self.progbar.close()
            self.progbar = None
        if self.tqdm_target is not sys.stdout:
            self.tqdm_target.close()


class EpochProgressCallback(Callback):

    DEBUG = False

    def __init__(self, cids_model, num_epochs, phase):
        self.cids_model = cids_model
        self.num_epochs = num_epochs - self.cids_model.count
        self.phase = phase
        self.progbar = None
        self.postfix = OrderedDict()
        self._epoch_format_str = "{:04d}"
        self._loss_format_str = "{:.3e}"
        super().__init__()

    def on_train_begin(self, logs=None):
        self.postfix = OrderedDict()
        self.postfix["epoch"] = self._epoch_format_str.format(0)
        if self.cids_model.VERBOSITY:
            self.tqdm_target = self.cids_model.stream_to_logger()
        else:
            self.tqdm_target = open(  # pylint: disable=consider-using-with
                os.devnull, "w", encoding="utf8"
            )
        self.progbar = trange(
            self.num_epochs,
            miniters=self.cids_model.report_freq,
            file=self.tqdm_target,
            dynamic_ncols=True,
            desc=f"Training phase {self.phase:d}",
        )

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Increase step
        self.cids_model.count += 1
        # Get data
        epoch = self.cids_model.count
        initial_epoch = self.cids_model.initial_count
        if epoch % self.cids_model.report_freq == 0:
            # Set postfix
            self.postfix["epoch"] = self._epoch_format_str.format(epoch)
            update_dict = OrderedDict(
                [
                    (k, logs[k])
                    for k in sorted(logs.keys())
                    if k not in ["size", "val_batch", "batch", "val_batch"]
                ]
            )
            self.postfix.update(update_dict)
            # Update progress bar
            self.progbar.set_postfix(self.postfix, refresh=False)
            self.progbar.update(epoch - initial_epoch - self.progbar.n)

    def on_train_end(self, logs=None):
        if self.progbar is not None:
            self.progbar.close()
            self.progbar = None
            if self.tqdm_target is not sys.stdout:
                self.tqdm_target.close()
