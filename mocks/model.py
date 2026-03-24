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
"""Tensorflow/Keras model for CIDS. Part of the CIDS toolbox.

    Classes:
        CIDSModelTF: A Tensorflow/Keras model for CIDS
        CIDSModel: Alias for CIDSModelTF

"""
import gc
import glob
import json
import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from kerastuner import HyperParameters
from kerastuner.engine.oracle import Objective
from kerastuner.oracles import BayesianOptimizationOracle
from kerastuner.oracles import HyperbandOracle
from kerastuner.oracles import RandomSearchOracle
from matplotlib import gridspec
from matplotlib import pyplot as plt
from tensorflow.python.keras.engine import base_layer_utils

from ..base.model import BaseModel
from ..data.definition import DataDefinition
from ..data.definition import Feature
from ..external import adahessian
from ..external import innvestigate
from ..external.levenberg_marquardt import levenberg_marquardt
from .callbacks import CIDSCheckpoint
from .callbacks import EpochProgressCallback
from .callbacks import FreezeControl
from .callbacks import StepProgressCallback
from .custom_models import GANModel
from .custom_models import get_gan_mode
from .custom_steps import make_hessian_steps
from .custom_steps import make_parallel_steps
from .online_processing import ExpandMergeFeatures
from .online_processing import OnlineNormalize
from .tuner import CIDSTuner
from .tuner import SearchResults
from .utility import disable_tensorflow_memory_greed
from .utility import get_available_cpus
from .utility import get_available_gpus
from kadi_ai import projects
from kadi_ai.dash.constants import WebAppIdentifiers as ID
from kadi_ai.dash.hyperparameter_definitions import HyperparameterDefinition
from kadi_ai.dash.hyperparameter_definitions import ModelDefinition

import cids.tensorflow.losses as cus_losses

def create_legacy_data_definition(
    data_shape, data_format, input_indices, output_indices
):
    """Create a data definition to ensure compatibility for legacy code.

    Args:
        data_shape (list): A data shape
        data_format (str): A data format
        input_indices (list): indices of the input features
        output_indices (list): indices of the output feature

    Returns:
        DataDefinition: legacy data definition
    """
    data_shape = list(data_shape)
    data_shape[data_format.index("N")] = None
    input_features = [f"data{repr(list(input_indices)):s}"]
    output_features = [f"data{repr(list(output_indices)):s}"]
    return DataDefinition(
        Feature(
            "data",
            data_shape,
            data_format,
            dtype=tf.string,
            decode_str_to=tf.float64,
        ),
        input_features=input_features,
        output_features=output_features,
    )


class CIDSModelTF(BaseModel):  # pylint: disable=abstract-method
    def __init__(self, data_definition, model, **kwargs):
        """A wrapper for neural network models using CIDS and Keras.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           a (function returning a) tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        # BaseModel init
        super().__init__(data_definition, model, **kwargs)
        # Hyperparameters
        self._hp = None
        # Model
        if isinstance(model, (tf.keras.models.Model, tf.keras.models.Sequential)):
            # Fixed model, hyper parameter search not possible
            self._core_model_function = None
            self._core_model = model
        elif isinstance(model, dict):
            self._core_model_function = None
            for k, v in model.items():
                assert isinstance(
                    v, (tf.keras.models.Model, tf.keras.models.Sequential)
                ), f"Model {k:s} not a keras model."
            self._core_model = model
        elif callable(model):
            # Model given as a function, hyper parameter search possible
            self._core_model_function = model
            self._core_model = None
        else:
            raise ValueError(
                "Model must be keras model, dictionary of models, or model "
                + "building function."
            )
        # Building properties to be set later
        self.input_online_normalizer = None
        self.output_online_normalizer = None
        self.input_preprocess_model = None
        self.output_preprocess_model = None
        self.core_model = None
        self.postprocess_model = None
        self.state_extractor_model = None
        self.dtype = kwargs.get("dtype", tf.float32)
        self.built_input_shape = None
        self.built_output_shape = None
        self.states = []
        self.built = False
        self.built_submodels = None
        # Training
        optimizer_class = kwargs.get("optimizer", tf.optimizers.Adam)
        if isinstance(optimizer_class, tf.keras.optimizers.Optimizer):
            self.optimizer_class = optimizer_class.__class__
            self.optimizer_config = optimizer_class.get_config()
        else:
            self.optimizer_class = optimizer_class
            self.optimizer_config = {}
        self.optimizer = None  # Set later by self.optimizer_class
        if isinstance(optimizer_class, str):
            assert optimizer_class in ["lm", "levenberg_marquardt"]
            self.loss = levenberg_marquardt.MeanSquaredError()
        else:
            self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = []
        self.monitor = "val_loss"
        self._callbacks = [FreezeControl(self)]
        self._best_monitor = None
        self.save_best_only = True
        # Reporting
        if not self.DEBUG:
            # Disable tensorflow warnings and infos
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            tf.get_logger().setLevel("ERROR")
            tf.autograph.set_verbosity(1)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.count_mode = "epochs"
        self.initial_count = 0
        self.count = 0
        self.phase = 0
        self.report_freq = kwargs.get("report_freq", 1)
        self.report_samples_above_threshold = False
        self.save_freq = kwargs.get("save_freq", 1)
        self.parallel_inp = None  # set by class method parallel multi model
        self.loss_weights = []
        self.validation_loss_tracker = tf.keras.metrics.Mean()
        self.train_loss_tracker = tf.keras.metrics.Mean()
        # Check GPUs and select distribution strategy
        self.num_gpus = kwargs.get("num_gpus", len(get_available_gpus()))
        self.strategy = self._select_strategy(self.num_gpus)

    def _select_strategy(self, num_gpus):
        if num_gpus > 1:
            if self.VERBOSITY > 1:
                self.log("Multiple GPUs found. Using mirror distribution.")
            strategy = tf.distribute.MirroredStrategy()
            disable_tensorflow_memory_greed()
        elif num_gpus:
            if self.VERBOSITY > 1:
                self.log("Single GPU found. Using one device distribution.")
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            disable_tensorflow_memory_greed()
        else:
            if self.VERBOSITY > 1:
                self.log("Single CPU found. Using one device distribution.")
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        return strategy

    @classmethod
    def regression(cls, *args, **kwargs):
        """Regression variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        inst = cls(*args, **kwargs)
        inst.meta_architecture = "serial"
        inst.num_classes = None
        inst.online_normalize = True
        inst.encode_categorical = False
        inst.loss = tf.keras.losses.MeanSquaredError()
        return inst

    @classmethod
    def binary_classification(cls, *args, **kwargs):
        """Binary classification variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        inst = cls(*args, **kwargs)
        inst.meta_architecture = "serial"
        inst.num_classes = 1
        inst.online_normalize = "input"
        inst.encode_categorical = "output"
        inst.loss = tf.keras.losses.BinaryCrossentropy()
        return inst

    @classmethod
    def categorical_classification(cls, *args, **kwargs):
        """Categorical classification variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            num_classes:     number of classes for one_hot encoding of targets
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        if isinstance(args[0], DataDefinition):
            num_classes = kwargs.get("num_classes")
        else:
            num_classes = args[0]
        inst = cls(*args[1:], **kwargs)
        inst.meta_architecture = "serial"
        inst.num_classes = num_classes
        inst.online_normalize = "input"
        inst.encode_categorical = "output"
        inst.loss = tf.keras.losses.CategoricalCrossentropy()
        return inst

    @classmethod
    def parallel_multi_model(cls, *args, **kwargs):
        """Parallel multi model variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            parallel_inp:    model input selection from  parallel submodels available
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        parallel_inp: str = kwargs.get("parallel_inp")
        inst = cls(*args, **kwargs)
        inst.parallel_inp = parallel_inp
        inst.meta_architecture = "parallel"
        inst.online_normalize = "input"
        inst.loss = tf.keras.losses.MeanSquaredError()
        return inst

    @classmethod
    def spatial_classification(cls, *args, **kwargs):
        """Spatial classification variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        inst = cls(*args, **kwargs)
        inst.num_classes = 1
        inst.meta_architecture = "serial"
        inst.online_normalize = "input"
        inst.encode_categorical = "output"
        inst.loss = tf.keras.losses.BinaryCrossentropy()
        return inst

    @classmethod
    def generative_adversarial(cls, *args, **kwargs):
        """Generative adversarial variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        inst = cls(*args, **kwargs)
        inst.meta_architecture = "adversarial"
        inst.num_classes = 1
        inst.online_normalize = "input"
        inst.encode_categorical = "output"
        inst.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        inst.save_best_only = False  # Important since metrics may stop to improve
        return inst

    @classmethod
    def vxm_network(cls, *args, **kwargs):
        """VoxelMorph variant of CIDSModel.

        This wrapper manages training, evaluation and inference of neural
        network models. It manages session creation, initialization and
        clean up.

        Args:
            data_definition: a DataDefinition object
            model:           tensorflow keras model
        Keyword Args:
            optimizer:       tensorflow optimizer class (defaults to Adam)
            num_gpus:        number of GPUs to use (None for all available)
            dtype:           data type for computations
            report_freq:     number of steps inbetween reports
            save_freq:       number of steps inbetween model saving
            name:            name of the model
            result_dir:      a directory to store the results in
        """
        inst = cls(*args, **kwargs)
        inst.online_normalize = False
        inst.loss = ['mse', cus_losses.Grad("l2").loss]
        inst.loss_weights = kwargs["loss_weights"]
        # print(f"test{inst.loss_weights}")
        return inst

    @property
    def name(self):
        if self.suppress_architecture_string or self._core_model is None:
            return super().name
        if hasattr(self._core_model, "layers"):
            return self.assemble_model_name(
                self.base_name, self.identifier, self._core_model.layers
            )
        if isinstance(self._core_model, dict):
            ordered_keys = self._core_model.keys()
            forward_keys = [k for k in ordered_keys if "adversarial" not in k]
            if forward_keys:
                assert hasattr(
                    self._core_model[forward_keys[0]], "layers"
                ), "Incompatible model defined as first forward core model."
                return self.assemble_model_name(
                    self.base_name,
                    self.identifier,
                    self._core_model[forward_keys[0]].layers,
                )
            raise ValueError("No forward model defined in core model dictionary.")
        raise ValueError(
            "Core model must be keras model or dictionary of keras models."
        )

    def _check_execute(self, flag, mode, data_format=None):
        if flag:
            if flag in [True, "both", "all"]:
                return True
            if mode in flag:
                if data_format is None or data_format in flag:
                    return True
        return False

    def _compute_preprocess_shape(self, shape, mode="input", data_format=None):
        shape = deepcopy(shape)
        if isinstance(shape, dict):
            return {
                k: self._compute_preprocess_shape(v, mode=mode, data_format=k)
                for k, v in shape.items()
            }
        if self._check_execute(self.encode_categorical, mode, data_format=data_format):
            feature_axis = self.feature_axis
            assert (
                shape[feature_axis] == 1
            ), f"Categorical encoding requires single feature {mode}!"
            num_classes = self.num_classes
            if isinstance(num_classes, dict):
                if mode == "input":
                    num_classes = num_classes["X"]
                else:
                    num_classes = num_classes["Y"]
            shape[feature_axis] = num_classes
        if self._check_execute(
            self.collapse_repeated_features, mode, data_format=data_format
        ):
            feature_axis = self.feature_axis
            batch_axis = self.batch_axis
            shape = [shape[batch_axis], shape[feature_axis]]
        return shape

    @property
    def input_preprocess_shape(self):
        shape = deepcopy(self.input_shape)
        return self._compute_preprocess_shape(shape, mode="input")

    @property
    def output_preprocess_shape(self):
        shape = deepcopy(self.output_shape)
        return self._compute_preprocess_shape(shape, mode="output")

    @property
    def forward_model(self):
        # Get forward model
        if isinstance(self.core_model, dict):
            assert isinstance(self.core_model["forward"], tf.keras.Model)
            return self.core_model["forward"]
        if isinstance(self.core_model, tf.keras.Model):
            return self.core_model
        if self.core_model is None:
            # Not instantiated yet
            return None
        raise ValueError("Invalid forward model.")

    @property
    def freeze(self):
        """Freeze state of the preprocessor."""
        if isinstance(self.input_online_normalizer, dict):
            values_input = [
                input_normalizer.freeze
                for input_normalizer in self.input_online_normalizer.values()
                if input_normalizer is not None
            ]
            values_output = [
                output_normalizer.freeze
                for output_normalizer in self.output_online_normalizer.values()
                if output_normalizer is not None
            ]
        else:
            values_input = []
            values_output = []
            if self.input_online_normalizer is not None:
                values_input = [self.input_online_normalizer.freeze]
            if self.output_online_normalizer is not None:
                values_output = [self.output_online_normalizer.freeze]
        values = values_input + values_output
        assert len({v.numpy() for v in values}) <= 1, "Inconsistent freezing!"
        return all(values)

    @freeze.setter
    def freeze(self, freeze):
        """Freeze or unfreeze the preprocessor variables."""
        if isinstance(self.input_online_normalizer, dict):
            for input_normalizer in self.input_online_normalizer.values():
                if input_normalizer is not None:
                    input_normalizer.freeze = freeze
            for output_normalizer in self.output_online_normalizer.values():
                if output_normalizer is not None:
                    output_normalizer.freeze = freeze
        else:
            if self.input_online_normalizer is not None:
                self.input_online_normalizer.freeze = freeze
            if self.output_online_normalizer is not None:
                self.output_online_normalizer.freeze = freeze

    def _maybe_wrap_built_core_models(self):
        if self.optimizer_class in ["lm", "levenberg_marquardt"]:
            # Wrap model for LM optimizer
            if isinstance(self.core_model, tf.keras.Model):
                self.core_model = levenberg_marquardt.ModelWrapper(self.core_model)
            elif isinstance(self.core_model, dict):
                for k, v in self.core_model.items():
                    self.core_model[k] = levenberg_marquardt.ModelWrapper(v)

    def _get_core_model_name(self, key):
        return "".join(s.capitalize() for s in key.split("_"))

    def _build_core_model(self, input_shape, output_shape, submodels=None):
        """Define the keras model for the core neural network."""
        # Inputs and outputs
        if isinstance(input_shape, dict) and isinstance(output_shape, dict):
            x0 = [
                tf.keras.layers.Input(shape=inp_shape[1:], name=name)
                for name, inp_shape in input_shape.items()
            ]
            y0 = [
                tf.keras.layers.Input(out_shape[1:], name=name)
                for name, out_shape in output_shape.items()
            ]
        else:
            x0 = [tf.keras.layers.Input(input_shape[1:])]
            y0 = [tf.keras.layers.Input(output_shape[1:])]
        # Single core model
        if isinstance(self._core_model, tf.keras.Model):
            model = self._core_model
            self._check_cudnn_compatibility(model)
            y_ = self._assemble_model_sequential_graph(model, x0)
            self.core_model = tf.keras.Model(inputs=(x0,), outputs=y_, name="Core")
            self.core_model.build(input_shape)
        elif isinstance(self._core_model, dict):
            # Multiple core models
            ordered_keys = list(self._core_model.keys())
            if not submodels:
                submodels = ordered_keys
            forward_keys = [
                k for k in ordered_keys if "adversarial" not in k and k in submodels
            ]
            adversarial_keys = [
                k for k in ordered_keys if "adversarial" in k and k in submodels
            ]
            self.core_model = {}
            # Forward core models
            x1 = x0
            x1p = x0
            y_ = x1
            yp_ = x1
            for k in forward_keys:
                # Create model from input to output
                model = self._core_model[k]
                self._check_cudnn_compatibility(model)
                cm_key = "subcore_" + k
                # Check for new parallel branch
                if "parallel" in k:
                    # Paralle forward passself._core_model
                    y1p_ = self._assemble_model_sequential_graph(model, x1p)
                    self.core_model[cm_key] = tf.keras.Model(
                        inputs=(x1p,),  # TODO: will fail, if no parallel input defined
                        outputs=y1p_,
                        name=self._get_core_model_name(cm_key),
                    )
                    self.core_model[cm_key].build([xp.shape[1:] for xp in x1p])
                    # Apply model to graph
                    yp_ = self.core_model[cm_key](yp_)
                    # Update inputs
                    x1p = [tf.keras.Input(yp_.shape[1:])]
                else:
                    # Normal forward pass
                    y1_ = self._assemble_model_sequential_graph(model, x1)
                    self.core_model[cm_key] = tf.keras.Model(
                        inputs=(x1,),
                        outputs=y1_,
                        name=self._get_core_model_name(cm_key),
                    )
                    self.core_model[cm_key].build([x.shape[1:] for x in x1])
                    # Apply model to graph
                    y_ = self.core_model[cm_key](y_)
                    # Update input
                    x1 = [tf.keras.Input(y_.shape[1:])]
                    # Update parallel branch if it exists
                    if self.parallel_inp:
                        input_model_name = "subcore_" + self.parallel_inp
                        if input_model_name == cm_key:
                            yp_ = y_
                            x1p = [tf.keras.Input(yp_.shape[1:])]
            # Adversarial core models
            x2 = x0
            y2 = y0
            e_ = y2
            # assert len(adversarial_keys) < 2, "Only 1 adversarial layers allowed."
            for k in adversarial_keys:
                model = self._core_model[k]
                cm_key = "subcore_" + k
                if "conditional" in k:
                    xy2 = ExpandMergeFeatures()([x2, y2])
                    # Create model from input to output
                    self._check_cudnn_compatibility(model)
                    e2_ = self._assemble_model_sequential_graph(model, xy2)
                    self.core_model[cm_key] = tf.keras.Model(
                        inputs=(x2, y2),
                        outputs=e2_,
                        name=self._get_core_model_name(cm_key),
                    )
                    self.core_model[cm_key].build((x2.shape[1:], y2.shape[1:]))
                    # Apply model to graph
                    e_ = self.core_model[cm_key]((x2, e_))
                else:
                    # Create model from input to output
                    self._check_cudnn_compatibility(model)
                    e2_ = self._assemble_model_sequential_graph(model, y2)
                    self.core_model[cm_key] = tf.keras.Model(
                        inputs=(y2,),
                        outputs=e2_,
                        name=self._get_core_model_name(cm_key),
                    )
                    self.core_model[cm_key].build([yi.shape[1:] for yi in y2])
                    # Apply model to graph
                    e_ = self.core_model[cm_key](e_)
                # Update input
                y2 = [tf.keras.Input(ei.shape[1:]) for ei in e_]
            # Training wrappers
            if self.meta_architecture == "serial":
                self.core_model["forward"] = tf.keras.Model(
                    inputs=(x0,), outputs=y_, name="CoreForward"
                )
            elif self.meta_architecture == "adversarial":
                mode = get_gan_mode(ordered_keys)
                generator = tf.keras.Model(inputs=(x0,), outputs=y_, name="CoreForward")
                if mode == "conditional":
                    adversary = tf.keras.Model(
                        inputs=(x0, y0), outputs=e_, name="CoreAdversarial"
                    )
                else:
                    adversary = tf.keras.Model(
                        inputs=(y0,), outputs=e_, name="CoreAdversarial"
                    )
                self.core_model["forward"] = GANModel(generator, adversary)
                self.core_model["forward"].build(
                    ([xi.shape[1:] for xi in x0], [yi.shape[1:] for yi in y0])
                )
            elif self.meta_architecture == "parallel":
                self.core_model["forward_parallel"] = tf.keras.Model(
                    inputs=(x0,), outputs=yp_, name="CoreForwardParallel"
                )
            else:
                raise ValueError("Invalid core model.")
        # State extractor model
        states = self.states
        self.state_extractor_model = tf.keras.Model(
            inputs=(x0,), outputs=states, name="StateExtractor"
        )
        self.state_extractor_model.build(input_shape)
        # Maybe wrap all core models for Levenberg Marquardt
        self._maybe_wrap_built_core_models()
        return self.core_model

    def _build_input_preprocess_model(self, input_shape, hp=None):
        """Define the keras model for preprocessing of the inputs."""
        if isinstance(input_shape, dict):
            # Input
            x0 = {
                k: tf.keras.layers.Input(shape=v[1:], name=k)
                for k, v in input_shape.items()
            }
            # Preprocess model
            x = {k: self.preprocess_inputs(v, data_format=k) for k, v in x0.items()}
        elif isinstance(input_shape, (list, tuple)):
            # Input
            x0 = tf.keras.layers.Input(input_shape[1:])
            # Preprocess model
            x = self.preprocess_inputs(x0)
        else:
            raise ValueError("Input shape must be either list, tuple or dictionary")
        self.input_preprocess_model = tf.keras.Model(
            inputs=(x0,), outputs=(x,), name="InputPreprocess"
        )
        self.input_preprocess_model.build(input_shape)
        return self.input_preprocess_model

    def _build_output_preprocess_model(self, output_shape, hp=None):
        """Define the keras model for preprocessing of the outputs."""
        if isinstance(output_shape, dict):
            # Input
            y0 = {
                k: tf.keras.layers.Input(shape=v[1:], name=k)
                for k, v in output_shape.items()
            }
            # Preprocess model
            y = {k: self.preprocess_outputs(v, data_format=k) for k, v in y0.items()}
        elif isinstance(output_shape, (list, tuple)):
            # Input
            y0 = tf.keras.layers.Input(output_shape[1:])
            # Preprocess model
            y = self.preprocess_outputs(y0)
        else:
            raise ValueError("Output shape must be either list, tuple or dictionary")
        self.output_preprocess_model = tf.keras.Model(
            inputs=(y0,), outputs=(y,), name="OutputPreprocess"
        )
        self.output_preprocess_model.build(output_shape)
        return self.output_preprocess_model

    def _build_postprocess_model(self, output_shape, hp=None):
        """Define the keras model for postprocessing of the outputs."""
        if isinstance(output_shape, dict):
            # Input
            y0_ = {
                k: tf.keras.layers.Input(shape=v[1:], name=k)
                for k, v in output_shape.items()
            }
            # Postprocess model
            y_ = {k: self.postprocess_outputs(v, data_format=k) for k, v in y0_.items()}
        elif isinstance(output_shape, list):
            # Input
            y0_ = tf.keras.layers.Input(output_shape[1:])
            # Postprocess model
            y_ = self.postprocess_outputs(y0_)
        else:
            raise ValueError("Output shape must be either list, tuple or dictionary")
        self.postprocess_model = tf.keras.Model(
            inputs=(y0_,), outputs=(y_,), name="Postprocess"
        )
        # does this work with the state extractor?
        self.postprocess_model.build(output_shape)
        return self.postprocess_model

    def _train_callbacks(self, count, phase, report=True):
        """Create callbacks for training."""
        train_callbacks = []
        # Base callbacks
        train_callbacks += self._callbacks
        # Progress bar
        #   Do not remove or change this functionality, since it updates step!
        if self.count_mode == "epochs":
            self.progress_callback = EpochProgressCallback(self, count, phase)
        else:
            self.progress_callback = StepProgressCallback(self, count, phase)
        train_callbacks += [self.progress_callback]
        # Tensorboard summary callbacks
        if self.report_freq and report:
            if self.DEBUG:
                # Enable graph writing
                # This causes annoying CUPTI errors that do not affect training
                # write_graph = True
                # profile_batch = 2
                write_graph = False
                profile_batch = 0
            else:
                # Disable graph writing and profiling to accelerate training
                write_graph = False
                profile_batch = 0
            if isinstance(self.core_model, tf.keras.Model):
                core_model_writer = tf.keras.callbacks.TensorBoard(
                    self.summary_dir,
                    histogram_freq=self.report_freq,
                    update_freq="epoch",
                    write_graph=write_graph,
                    write_grads=False,
                    profile_batch=profile_batch,
                )
                core_model_writer.set_model(self.core_model)
                train_callbacks += [core_model_writer]
            elif isinstance(self.core_model, dict):
                for k, cm in self.core_model.items():
                    if "subcore_" not in k:
                        cm_writer = tf.keras.callbacks.TensorBoard(
                            self.summary_dir,
                            histogram_freq=self.report_freq,
                            update_freq="epoch",
                            write_graph=write_graph,
                            write_grads=False,
                            profile_batch=profile_batch,
                        )
                        cm_writer.set_model(cm)
                        train_callbacks += [cm_writer]
            else:
                raise ValueError("Invalid core model.")
            # causes CUPTI errors
            # if self.DEBUG:
            #     input_preprocess_model_writer = tf.keras.callbacks.TensorBoard(
            #         self.summary_dir, histogram_freq=self.report_freq,
            #         update_freq="epoch",
            #         write_graph=True, write_grads=False, profile_batch=2)
            #     input_preprocess_model_writer.set_model(
            #         self.input_preprocess_model)
            #     output_preprocess_model_writer = tf.keras.callbacks.TensorBoard(
            #         self.summary_dir, histogram_freq=self.report_freq,
            #         update_freq="epoch",
            #         write_graph=True, write_grads=False, profile_batch=2)
            #     output_preprocess_model_writer.set_model(
            #         self.output_preprocess_model)
            #     train_callbacks += [input_preprocess_model_writer,
            #                         output_preprocess_model_writer]

        # Model saving callbacks
        if self.save_freq:
            verbose = int(self.VERBOSITY > 2)
            monitor = self.monitor
            direction = "auto"
            if isinstance(monitor, Objective):
                direction = monitor.direction
                monitor = monitor.name
            model_saver = CIDSCheckpoint(
                self,
                save_best_only=self.save_best_only,
                monitor=monitor,
                mode=direction,
                save_weights_only=True,
                save_freq=self.save_freq,
                verbose=verbose,
            )
            train_callbacks += [model_saver]
        return train_callbacks

    # pylint: disable=method-hidden
    def build(
        self,
        hp=None,
        input_shape=None,
        output_shape=None,
        input_preprocess_shape=None,
        output_preprocess_shape=None,
        use_gpu=True,
        checkpoint=None,
        batch_size=None,
        submodels=None,
    ):
        """Build the model tensorflow graph."""
        # Hyperparameters

        if hp is None:
            if self._hp is None:
                hp = HyperParameters()
            else:
                hp = self._hp
        self._hp = hp
        # Dynamic shapes
        chunk_size = self.data_reader.chunk_size
        input_shape = self.get_input_shape(chunk_size=chunk_size)
        output_shape = self.get_output_shape(chunk_size=chunk_size)
        if self.meta_architecture == "parallel":
            input_preprocess_shape = self._compute_preprocess_shape(
                input_shape, "input", self.input_format
            )
            output_preprocess_shape = self._compute_preprocess_shape(
                output_shape, "output", self.output_format
            )
        else:
            input_preprocess_shape = self._compute_preprocess_shape(
                input_shape, "input", None
            )
            output_preprocess_shape = self._compute_preprocess_shape(
                output_shape, "output", None
            )
        # Extract Training Hyperparameters
        if self.core_model is None:
            training_hps = {"space": {}, "values": {}}
            if not hasattr(self, "hp_dict"):
                self.hp_dict = {"Model": {}, "Training": {}}
            if "trial" in self.meta_folder:
                model_keys = list(self.hp_dict["Model"]["values"].keys())
                training_hps["values"] = {
                    h: hp.values[h]
                    for h in hp.values
                    if h not in model_keys and "tuner" not in h
                }
                training_hps["space"] = self._extract_hp_dict_from_hp_object(
                    hp, model_keys
                )
            else:
                training_hps["values"] = {
                    h: hp.values[h] for h in hp.values if "tuner" not in h
                }
                training_hps["space"] = self._extract_hp_dict_from_hp_object(
                    hp, ["tuner"]
                )
            create_hp_dict = True
        else:
            create_hp_dict = False
        # Shapes
        input_shape = input_shape or self.input_shape
        output_shape = output_shape or self.output_shape
        input_preprocess_shape = input_preprocess_shape or self.input_preprocess_shape
        output_preprocess_shape = (
            output_preprocess_shape or self.output_preprocess_shape
        )
        # Check if model changed
        if self.built:
            # Shapes canged?
            model_changed = np.any(
                np.asarray(input_shape) != np.asarray(self.built_input_shape)
            ) or np.any(np.asarray(output_shape) != np.asarray(self.built_output_shape))
            # Submodels changed?
            model_changed = self.built_submodels != submodels
            # Clear graph
            if model_changed:
                if self.VERBOSITY > 1:
                    self.log(
                        "Input or output shape changed, "
                        + "likely due to chunking. "
                        + "Recreating model and reloading."
                    )
                self.clear(reset_counts=False, reset_cached_datasets=True)
                if self.phase > 0:
                    checkpoint = f"phase{self.phase - 1:02d}"
        if not self.built:
            if isinstance(input_shape, dict):
                normalize_mode = "stddev"
                if (
                    isinstance(self.online_normalize, str)
                    and "minmax" in self.online_normalize
                ):
                    normalize_mode = "minmax"
                self.input_online_normalizer = {
                    k: OnlineNormalize(
                        v,
                        k,
                        dtype=tf.float64,
                        name="InputOnlineNormalize" + k,
                        normalize_mode=normalize_mode,
                    )
                    for k, v in input_shape.items()
                    if self._check_execute(
                        self.online_normalize, "input", data_format=k
                    )
                }
            elif self._check_execute(
                self.online_normalize,
                "input",
            ):
                normalize_mode = "stddev"
                if (
                    isinstance(self.online_normalize, str)
                    and "minmax" in self.online_normalize
                ):
                    normalize_mode = "minmax"
                self.input_online_normalizer = OnlineNormalize(
                    input_shape,
                    self.input_format,
                    dtype=tf.float64,
                    name="InputOnlineNormalize",
                    normalize_mode=normalize_mode,
                )
            if isinstance(output_shape, dict):
                normalize_mode = "stddev"
                if (
                    isinstance(self.online_normalize, str)
                    and "minmax" in self.online_normalize
                ):
                    normalize_mode = "minmax"
                self.output_online_normalizer = {
                    k: OnlineNormalize(
                        v,
                        k,
                        dtype=tf.float64,
                        name="OutputOnlineNormalize" + k,
                        normalize_mode=normalize_mode,
                    )
                    for k, v in output_shape.items()
                    if self._check_execute(
                        self.online_normalize, "output", data_format=k
                    )
                }
            elif self._check_execute(self.online_normalize, "output"):
                normalize_mode = "stddev"
                if (
                    isinstance(self.online_normalize, str)
                    and "minmax" in self.online_normalize
                ):
                    normalize_mode = "minmax"
                self.output_online_normalizer = OnlineNormalize(
                    output_shape,
                    self.output_format,
                    dtype=tf.float64,
                    name="OutputOnlineNormalize",
                    normalize_mode=normalize_mode,
                )
            # Scopes
            preprocessing_postprocessing_scope, core_scope = self._get_execution_scopes(
                use_gpu=use_gpu
            )
            # Preprocess
            with preprocessing_postprocessing_scope:  # Not necessary, better to be safe
                self._build_input_preprocess_model(input_shape, hp=hp)
                self._build_output_preprocess_model(output_shape, hp=hp)
            # Core model
            with core_scope:
                # Set core model from function (Must be done under strategy scope)
                if self._core_model is None:
                    self._core_model = self._call_model_function(
                        self._core_model_function, hp
                    )
                # Build
                self._build_core_model(
                    input_preprocess_shape,
                    output_preprocess_shape,
                    submodels=submodels,
                )
            if create_hp_dict:
                # Extract Model Hyperparameters
                model_hps = {"space": {}, "values": {}}
                training_keys = list(training_hps["values"].keys())
                model_hps["values"] = {
                    h: hp.values[h]
                    for h in hp.values
                    if h not in training_keys and "tuner" not in h
                }
                model_hps["space"] = self._extract_hp_dict_from_hp_object(
                    hp, training_keys
                )
                # Create Hyperparameter Dictionary
                self.hp_dict = {"Model": model_hps, "Training": training_hps}
                # Save cidsmodel json
                cidsmodel_json_path = os.path.join(
                    self.base_model_dir, "cidsmodel.json"
                )
                with open(cidsmodel_json_path, "w+", encoding="utf8") as json_file:
                    cidsmodel_dict = {
                        "key": "CIDS:Model",
                        "type": "dict",
                        "value": [
                            {"key": "Name", "type": "str", "value": self.base_name},
                            {
                                "key": "Hyperparameters",
                                "type": "dict",
                                "value": [
                                    {
                                        "key": "Model",
                                        "type": "dict",
                                        "value": model_hps["space"],
                                    },
                                    {
                                        "key": "Training",
                                        "type": "dict",
                                        "value": training_hps["space"],
                                    },
                                ],
                            },
                        ],
                    }
                    json_file.write(
                        json.dumps(
                            cidsmodel_dict,
                            sort_keys=True,
                            indent=4,
                            separators=(",", ": "),
                        )
                    )
            # Postprocess
            with preprocessing_postprocessing_scope:  # Not necessary, better to be safe
                self._build_postprocess_model(output_preprocess_shape)
            # Built successfully
            self.built = True
            self.built_submodels = submodels
            self.built_input_shape = input_shape
            self.built_output_shape = output_shape
            # Load
            self.load(checkpoint)
            # Print
            if self.VERBOSITY > 1:
                self.summary(batch_size=batch_size)
                if checkpoint is None:
                    self.plot_models()  # plot only on first model creation
            # Disable logging
            if not self.DEBUG:
                # Disable tensorflow warnings and infos
                # this does not work
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                tf.get_logger().setLevel("ERROR")
                tf.autograph.set_verbosity(1)
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def summary(self, batch_size=None):
        """Print a model summary."""
        self.log("Model Summary: " + self.name)
        if self.input_preprocess_model is not None:
            self.input_preprocess_model.summary(print_fn=self.log)
        if self.output_preprocess_model is not None:
            self.output_preprocess_model.summary(print_fn=self.log)
        if isinstance(self.core_model, tf.keras.Model):
            self.core_model.summary(print_fn=self.log)
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                if "subcore_" in k:
                    self.log(k.capitalize() + " Model:")
                    cm.summary(print_fn=self.log)
        else:
            raise ValueError("Invalid core model.")
        if self.postprocess_model is not None:
            self.postprocess_model.summary(print_fn=self.log)
        if batch_size is not None:
            if isinstance(self.core_model, tf.keras.Model):
                cm_memory = self._calc_model_memory_usage(batch_size, self.core_model)
                self.log(f"Estimated memory (Core model): {cm_memory:f} GBytes")
            elif isinstance(self.core_model, dict):
                for k, cm in self.core_model.items():
                    if "subcore_" in k:
                        cm_memory = self._calc_model_memory_usage(batch_size, cm)
                        self.log(
                            f"Estimated memory ({k.capitalize():s} model):"
                            + f" {cm_memory:f} GBytes"
                        )
            else:
                raise ValueError("Invalid core model.")
            ip_memory = self._calc_model_memory_usage(
                batch_size, self.input_preprocess_model
            )
            op_memory = self._calc_model_memory_usage(
                batch_size, self.output_preprocess_model
            )
            pp_memory = self._calc_model_memory_usage(
                batch_size, self.postprocess_model
            )
            self.log(f"Estimated memory (Input preprocess model): {ip_memory:f} GBytes")
            self.log(
                f"Estimated memory (Output preprocess model): {op_memory:f} GBytes"
            )
            self.log(f"Estimated memory (Postprocess model): {pp_memory:f} GBytes")

    def plot_models(self, file_format=None):
        """Plot all models."""
        self.log("Plotting models.")
        try:
            file_format = file_format or plt.rcParams["savefig.format"]
        except KeyError:
            file_format = "png"
        if self.input_preprocess_model is not None:
            plot_file = os.path.join(
                self.plot_dir, f"input_preprocess_model.{file_format}"
            )
            tf.keras.utils.plot_model(
                self.input_preprocess_model, to_file=plot_file, show_shapes=True
            )
        if self.output_preprocess_model is not None:
            plot_file = os.path.join(
                self.plot_dir, f"output_preprocess_model.{file_format}"
            )
            tf.keras.utils.plot_model(
                self.output_preprocess_model, to_file=plot_file, show_shapes=True
            )
        if isinstance(self.core_model, tf.keras.Model):
            plot_file = os.path.join(self.plot_dir, f"model.{file_format}")
            tf.keras.utils.plot_model(
                self.core_model, to_file=plot_file, show_shapes=True
            )
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                if "subcore_" in k:
                    plot_file = os.path.join(self.plot_dir, f"{k}_model.{file_format}")
                    tf.keras.utils.plot_model(cm, to_file=plot_file, show_shapes=True)
        else:
            raise ValueError("Invalid core model.")
        if self.postprocess_model is not None:
            plot_file = os.path.join(self.plot_dir, f"postprocess_model.{file_format}")
            tf.keras.utils.plot_model(
                self.postprocess_model, to_file=plot_file, show_shapes=True
            )

    def clear(self, reset_counts=True, reset_cached_datasets=True, reset_hps=True):
        self.data_reader.clear(reset_cached_datasets=reset_cached_datasets)
        self.built_input_shape = None
        self.built_output_shape = None
        self.input_online_normalizer = None
        self.output_online_normalizer = None
        self.input_preprocess_model = None
        self.output_preprocess_model = None
        self.core_model = None
        self._callbacks = []
        if self._core_model_function is not None:
            self._core_model = None
        if reset_hps:
            self._hp = None
        self.postprocess_model = None
        self.state_extractor_model = None
        self.states = []
        tf.keras.backend.clear_session()
        gc.collect()
        self.optimizer = None
        self._callbacks = [FreezeControl(self)]
        self._best_monitor = None
        self.built = False
        self.built_submodels = None
        if reset_counts:
            self.initial_count = 0
            self.count = 0
            self.phase = 0

    @staticmethod
    def _save_weights(model, checkpoint_path):
        model.save_weights(checkpoint_path)

    @staticmethod
    def _load_weights(model, checkpoint_path):
        model.load_weights(checkpoint_path, by_name=True)

    @staticmethod
    def _export_model(model, path):
        model.save(path)

    @staticmethod
    def _import_model(path):
        return tf.keras.models.load_model(path)

    def save(self, checkpoint=None):
        """Manually save model weights and variables to checkpoint."""
        # Get base directory
        if checkpoint is not None:
            if isinstance(checkpoint, int):
                base_dir = os.path.join(
                    self.checkpoint_dir,
                    self._checkpoint_format_str.format(epoch=checkpoint),
                )
            else:
                base_dir = os.path.join(self.checkpoint_dir, str(checkpoint))
        else:
            base_dir = os.path.join(
                self.checkpoint_dir,
                "manual-" + self._checkpoint_format_str.format(epoch=self.count),
            )
        # Create base directory
        self.create_dir(base_dir)
        # Write checkpoint files
        if isinstance(self.core_model, tf.keras.Model):
            checkpoint_path = os.path.join(base_dir, "model_weights.h5")
            self._save_weights(self.core_model, checkpoint_path)
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                if "subcore_" in k:
                    checkpoint_path = os.path.join(base_dir, k + "_model_weights.h5")
                    self._save_weights(cm, checkpoint_path)
        else:
            raise ValueError("Invalid core model.")
        checkpoint_path = os.path.join(base_dir, "input_preprocess_weights.h5")
        self._save_weights(self.input_preprocess_model, checkpoint_path)
        checkpoint_path = os.path.join(base_dir, "output_preprocess_weights.h5")
        self._save_weights(self.output_preprocess_model, checkpoint_path)
        # Remove directory if empty (no files were saved)
        if not os.listdir(base_dir):
            os.rmdir(base_dir)
        if self.VERBOSITY > 2:
            self.log(f"Saved checkpoint to {base_dir:s}.")

    def load(self, checkpoint="last"):
        """Load model weights and variables from checkpoint subdirectory."""
        if checkpoint == "last" and self.save_best_only:
            self.warn(
                "Attempting to load checkpoint='last' but model.save_best_only = True. "
                + "This will not load the final model state at the end of training "
                + "but the last checkpoint that improved the model.monitor = "
                + f"'{self.monitor}'. Use checkpoint='last_phase' to get final state."
            )
        if checkpoint not in [None, "None"] and os.path.exists(self._checkpoint_dir):
            # Get base directory
            if isinstance(checkpoint, int):
                self.count = checkpoint
                self.initial_count = checkpoint
                checkpoint_name = self._checkpoint_format_str.format(epoch=checkpoint)
                base_dir = os.path.join(self._checkpoint_dir, checkpoint_name)
                if not os.path.exists(base_dir):
                    raise ValueError(f"Checkpoint {checkpoint_name:s}: Not found.")
                if self.VERBOSITY > 1:
                    self.log(
                        f"Continuing from {self.count_mode[:-1]:s} {self.count:d} "
                        + f"(checkpoint {checkpoint_name:s})."
                    )
            elif isinstance(checkpoint, str):
                candidates = sorted(
                    name
                    for name in os.listdir(self._checkpoint_dir)
                    if (
                        os.path.isdir(os.path.join(self._checkpoint_dir, name))
                        and os.listdir(os.path.join(self._checkpoint_dir, name))
                    )
                )
                if checkpoint == "last":
                    # Find all checkpoint directories that are numbers and not empty
                    candidates = [c for c in candidates if c.isdigit()]
                elif checkpoint == "last_phase":
                    # Find all checkpoint directories that are phases
                    candidates = [c for c in candidates if "phase" in c]
                else:
                    candidates = [c for c in candidates if c == checkpoint]
                try:
                    checkpoint = candidates[-1]
                except IndexError as e:
                    msg = f"Checkpoint {checkpoint:s}: No suitable checkpoint found."
                    if checkpoint not in ["last", "last_phase"]:
                        raise ValueError(msg) from e
                    if self.VERBOSITY:
                        self.warn(msg)
                    return
                try:
                    self.count = int(checkpoint)
                    self.initial_count = int(checkpoint)
                except ValueError:
                    pass
                base_dir = os.path.join(self._checkpoint_dir, checkpoint)
                if self.VERBOSITY > 1:
                    self.log(
                        f"Continuing from {self.count_mode[:-1]:s} {self.count:d} "
                        + f"(checkpoint {checkpoint:s})."
                    )
            else:
                raise ValueError(f"Invalid checkpoint type: {str(type(checkpoint))}")
        else:
            if self.VERBOSITY > 1:
                self.log("No checkpoint loaded.")
            if os.path.exists(self._model_dir):
                shutil.rmtree(self._model_dir)
                if self.VERBOSITY > 1:
                    self.warn("Clearing non-empty model directory.")
            return
        # Read checkpoint files
        if isinstance(self.core_model, tf.keras.Model):
            checkpoint_path = os.path.join(base_dir, "model_weights.h5")
            self._load_weights(self.core_model, checkpoint_path)
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                if "subcore_" in k:
                    try:
                        checkpoint_path = os.path.join(
                            base_dir, k + "_model_weights.h5"
                        )
                        self._load_weights(cm, checkpoint_path)
                    except OSError:
                        self.log("Failed loading weights: " + k)
        else:
            raise ValueError("Invalid core model.")
        checkpoint_path = os.path.join(base_dir, "input_preprocess_weights.h5")
        self._load_weights(self.input_preprocess_model, checkpoint_path)
        checkpoint_path = os.path.join(base_dir, "output_preprocess_weights.h5")
        self._load_weights(self.output_preprocess_model, checkpoint_path)

    def export(self, path, key=None):
        """Export model to path."""
        base_path, extension = os.path.splitext(path)
        if isinstance(self.core_model, tf.keras.Model):
            self._export_model(self.core_model, base_path + extension)
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                self._export_model(cm, base_path + "_" + k + extension)
        else:
            raise ValueError("Invalid core model.")
        self._export_model(
            self.input_preprocess_model, base_path + "_input_preprocess" + extension
        )
        self._export_model(
            self.output_preprocess_model, base_path + "_output_preprocess" + extension
        )
        self._export_model(
            self.postprocess_model, base_path + "_postprocess" + extension
        )

    def train_data_pipeline(
        self, train_data, valid_data=None, batch_size=32, chunk_size=None
    ):
        """Create the data pipeline for training and validation."""
        # if isinstance(train_data[0], str) and os.path.exists(train_data[0]):
        #     self.data_reader.src_type = "file"
        # else:
        #     self.data_reader.src_type = "placeholder"
        # Generate data sets
        if self.count_mode == "epochs":
            repeats = 0
        else:
            repeats = None
        assert len(train_data) > 1, "Need at least 2 samples to train!"
        # Left-over (remainder) batches cause changes in batch size, which
        #   breaks distribution strategies for multiple GPUs
        drop_remainder = self.num_gpus > 1 and self.strategy is not None
        train_dataset = self.data_reader.generate_batch_dataset(
            train_data,
            batch_size=batch_size,
            chunk_size=chunk_size,
            mode="train",
            shuffle=True,
            repeats=repeats,
            drop_remainder=drop_remainder,
        )
        if valid_data is not None:
            if isinstance(batch_size, dict):
                valid_batch_size = batch_size["valid"]
            # valid batch size of 1 would make more sense, but breaks CUDNN
            # elif self.count_mode == "epochs":
            #     valid_batch_size = 1
            elif isinstance(batch_size, int):
                valid_batch_size = batch_size
            else:
                if isinstance(valid_data[0], str):
                    valid_batch_size = len(valid_data)
                elif (
                    len(valid_data) == 2
                    and hasattr(valid_data[0], "shape")
                    and (len(valid_data[0].shape) == len(self.input_format))
                ):
                    # Assume if tuple with two elements and full shape
                    #    is given that it is (inputs, targets)
                    valid_batch_size = len(valid_data[0])
                else:
                    valid_batch_size = len(valid_data)
            if self.num_gpus > 1:
                assert (
                    valid_batch_size == batch_size
                ), "Batch size must be constant when distributing to multiple GPUs!"
            valid_dataset = self.data_reader.generate_batch_dataset(
                valid_data,
                batch_size=valid_batch_size,
                mode="valid",
                shuffle=True,
                repeats=0,
                chunk_size=chunk_size,
                drop_remainder=drop_remainder,
            )
            return train_dataset, valid_dataset
        return train_dataset

    def test_data_pipeline(self, test_data, batch_size=1):
        """Create the data pipeline for testing."""
        return self.data_reader.generate_batch_dataset(
            test_data,
            batch_size=batch_size,
            mode="test",
            shuffle=False,
            repeats=0,
        )

    def _get_execution_scopes(self, use_gpu=False):
        cpus = get_available_cpus()
        preprocessing_postprocessing_scope = tf.device(cpus[0])
        # TODO: it would be cool to select multiple preprocessing devices.
        if use_gpu:
            if self.strategy is None:
                gpus = get_available_gpus()
                try:
                    core_scope = tf.device(gpus[0])
                except IndexError:
                    self.warn(
                        "No GPU found despite requested use of GPU. Computing on CPU."
                    )
                    core_scope = tf.device(cpus[0])
            else:
                core_scope = self.strategy.scope()
        else:
            core_scope = tf.device(cpus[0])
        return preprocessing_postprocessing_scope, core_scope

    def _is_training(self, training=None):
        call_context = base_layer_utils.call_context()
        if training is None:
            # Priority 2: `training` was passed to a parent layer.
            if call_context.training is not None:
                return call_context.training
            # Priority 3a: `learning_phase()` has been set.
            return tf.keras.backend.learning_phase()
        return False

    def preprocess_inputs(self, x, data_format=None):
        """Preprocess the inputs. Redefine for special behavior."""
        # Augment during training
        x = tf.cond(
            self._is_training(),
            lambda: self.augment_inputs(x),
            lambda: tf.identity(x),
        )
        # Normalize
        if self._check_execute(self.online_normalize, "input", data_format=data_format):
            if data_format is None:
                x = self.input_online_normalizer(x)
            else:
                x = self.input_online_normalizer[data_format](x)
        # Collapse categorical labels
        if self._check_execute(
            self.collapse_repeated_features, "input", data_format=data_format
        ):
            # Remove unnecessary repetitions of labels along spatiotemporal axes
            input_preprocess_shape = self.input_preprocess_shape
            if data_format is not None:
                input_preprocess_shape = input_preprocess_shape[data_format]
            # Chunk size does not matter
            if len(x.shape) > len(input_preprocess_shape):
                for _ in range(len(x.shape) - len(input_preprocess_shape)):
                    x = x[..., 0, :]
        # Encode categorical inputs
        if self._check_execute(
            self.encode_categorical, "input", data_format=data_format
        ):
            num_classes = self.num_classes
            if isinstance(num_classes, dict):
                num_classes = num_classes["X"]
            if num_classes is not None and num_classes > 1:
                feature_axis = self.feature_axis
                if x.shape[feature_axis] == 1:
                    x = x[..., 0]
                indices = tf.cast(x, tf.int32)
                x = tf.one_hot(indices, num_classes)
        # Cast to compute dtype
        x = tf.cast(x, self.dtype)
        return x

    def preprocess_outputs(self, y, data_format=None):
        """Preprocess the outputs. Redefine for custom behavior."""
        # Augment during training
        y = tf.cond(
            self._is_training(),
            lambda: self.augment_outputs(y),
            lambda: tf.identity(y),
        )
        # Normalize
        if self._check_execute(
            self.online_normalize, "output", data_format=data_format
        ):
            if data_format is None:
                y = self.output_online_normalizer(y)
            else:
                y = self.output_online_normalizer[data_format](y)
        # Collapse categorical labels
        if self._check_execute(
            self.collapse_repeated_features, "output", data_format=data_format
        ):
            # Remove unnecessary repetitions of labels along spatiotemporal axes
            output_preprocess_shape = self.output_preprocess_shape
            if data_format is not None:
                output_preprocess_shape = output_preprocess_shape[data_format]
            # Chunk size does not matter
            if len(y.shape) > len(output_preprocess_shape):
                for _ in range(len(y.shape) - len(output_preprocess_shape)):
                    y = y[..., 0, :]
        # Encode categorical outputs
        if self._check_execute(
            self.encode_categorical, "output", data_format=data_format
        ):
            # Encode categorical
            num_classes = self.num_classes
            if isinstance(num_classes, dict):
                num_classes = num_classes["Y"]
            if num_classes is not None and num_classes > 1:
                feature_axis = self.feature_axis
                if y.shape[feature_axis] == 1:
                    y = y[..., 0]
                indices = tf.cast(y, tf.int32)
                y = tf.one_hot(indices, num_classes)
        # Cast to compute dtype
        y = tf.cast(y, self.dtype)
        return y

    def augment_inputs(self, x):
        """Augment preprocessed inputs. Redefine for special behavior."""
        return tf.identity(x)

    def augment_outputs(self, y):
        """Augment preprocessed outputs. Redefine for special behavior."""
        return tf.identity(y)

    def postprocess_outputs(self, y, data_format=None):
        """Postprocess the outputs. Redefine for custom behavior."""
        # Rescale
        if self.output_online_normalizer is not None:
            if self._check_execute(
                self.online_normalize, "output", data_format=data_format
            ):
                if data_format is None:
                    y = self.output_online_normalizer(y, invert=True)
                else:
                    y = self.output_online_normalizer[data_format](y, invert=True)
        # Decode categorical outputs
        if self._check_execute(
            self.encode_categorical, "output", data_format=data_format
        ):
            feature_axis = self.feature_axis
            # Decode categorical
            num_classes = self.num_classes
            if isinstance(num_classes, dict):
                num_classes = num_classes["Y"]
            if num_classes is not None and num_classes > 1:
                y = tf.argmax(y, axis=feature_axis)
                y = tf.expand_dims(y, axis=-1)
        return y

    def _check_cudnn_compatibility(self, model):
        if not model.built:
            if isinstance(model, tf.keras.Sequential):
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.RNN):
                        if (
                            hasattr(layer, "could_use_cudnn")
                            and not layer.could_use_cudnn
                        ):
                            self.warn(
                                "Recurrent layer with invalid configuration"
                                + " for CUDNN (causes memory leaks!): "
                                + layer.name
                            )

    def _assemble_model_sequential_graph(self, model, x):
        if isinstance(model, tf.keras.Sequential):
            y = x
            for y_ in y:
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.RNN):
                        if layer.return_state:
                            # Extract states and pass only activation to next layer
                            out = layer(y_)
                            y_ = out[0]
                            self.states.append(out[1:])
                        else:
                            y_ = layer(y_)
                    else:
                        y_ = layer(y_)
        else:
            y_ = model(x)
        return y_

    def _maybe_customize_model(self):
        if isinstance(self.core_model, tf.keras.Model):
            return self.core_model
        # Update training function
        self.core_model["forward"].cids_model = self
        if self.meta_architecture == "parallel":
            train_step, test_step = make_parallel_steps(self)
        elif self.optimizer_class == adahessian.AdaHessian:
            train_step = make_hessian_steps(self)
            # TODO: implement hessian test step
        else:
            return self.core_model["forward"]
        self.core_model["forward"].train_step = train_step
        self.core_model["forward"].test_step = test_step
        return self.core_model

    def _compile_model(
        self, model, loss, optimizer, learning_rate=0.0, clipnorm=0.0, metrics=None, 
    ):
        # Fallback for GANs
        if isinstance(self.forward_model, GANModel):
            if not isinstance(learning_rate, dict):
                learning_rate = {"forward": learning_rate, "adversarial": learning_rate}
            if not isinstance(loss, dict):
                loss = {"forward": loss, "adversarial": loss}
            if not isinstance(metrics, dict):
                metrics = {"forward": metrics, "adversarial": metrics}
            optimizer_gen = self.optimizer_class(**self.optimizer_config)
            optimizer_gen.learning_rate = learning_rate["forward"]
            optimizer_adv = self.optimizer_class(**self.optimizer_config)
            optimizer_adv.learning_rate = learning_rate["adversarial"]
            self.forward_model.compile(
                optimizer_gen,
                optimizer_adv,
                loss_gen=loss["forward"],
                loss_adv=loss["adversarial"],
                metrics_gen=metrics["forward"],
                metrics_adv=metrics["adversarial"],
            )
            return self.forward_model
        # Recursion for multi models
        if isinstance(model, dict):
            # Expand values
            if not isinstance(loss, dict):
                loss = {k: deepcopy(loss) for k in model.keys()}
            if not isinstance(optimizer, dict):
                optimizer = {k: deepcopy(optimizer) for k in model.keys()}
            if not isinstance(learning_rate, dict):
                learning_rate = {k: deepcopy(learning_rate) for k in model.keys()}
            if not isinstance(clipnorm, dict):
                clipnorm = {k: deepcopy(clipnorm) for k in model.keys()}
            if not isinstance(metrics, dict):
                metrics = {k: deepcopy(metrics) for k in model.keys()}
            # Compile models
            return {
                k: self._compile_model(
                    v,
                    loss[k],
                    optimizer[k],
                    learning_rate[k],
                    clipnorm.get(k),
                    metrics.get(k),
                )
                if "subcore_" not in k
                else v
                for k, v in model.items()
            }
        if not isinstance(model, tf.keras.models.Model):
            raise ValueError("Invalid core model.")
        # Set optimizer
        if optimizer is None:
            if self.optimizer_class in ["lm", "levenberg_marquardt"]:
                optimizer = tf.keras.optimizers.SGD()
            else:
                optimizer = self.optimizer_class(**self.optimizer_config)
        # Set learning rate and clipnorm
        assert isinstance(learning_rate, float)
        optimizer.learning_rate = learning_rate
        if clipnorm:
            optimizer.clipnorm = clipnorm
        # Metrics
        if isinstance(metrics, (str, tf.keras.metrics.Metric)):
            metrics = [metrics]
        if not isinstance(metrics, (list, tuple)):
            raise ValueError(
                "Attribute model.metrics must be a list/tuple of metrics for a "
                + "single model or a dictionary of lists/tuple for multiple models."
            )
        # Compile
        if model.optimizer is None:
            model.compile(optimizer, loss, metrics=metrics, loss_weights=self.loss_weights )
        return model

    def train_phase(
        self,
        train_data,
        valid_data,
        count,
        batch_size=32,
        learning_rate=0.01,
        chunk_size=None,
        freeze=None,
        clipnorm=None,
        checkpoint="last",
        callbacks=None,
        use_gpu=True,
        hp=None,
        submodels=None,
        report=True,
    ):
        """Train the model for a single training phase.

        Args:
            train_data:         list of samples/sample function (training)
            valid_data:         list of samples/sample function (validation)
            count:              maximum number of epochs to train
            batch_size:         number of samples to process each step
            learning_rate:      the learning rate
            chunk_size:         optional chunking of sequential data
            freeze:             freeze online normalization statistics?
            clipnorm:           value for gradient clipping norm (None to disable)
            callbacks:          additional keras callbacks
            checkpoint:         which checkpoint to load ("last", a name, or None)
            hp:                 a keras tuner HyperParameter set

        Returns:
            e:      final loss
            ev:     final validation loss
        """
        # Callbacks
        callbacks = callbacks or []
        # Scopes
        preprocessing_postprocessing_scope, core_scope = self._get_execution_scopes(
            use_gpu=use_gpu
        )
        # Build
        #   If model uses single GPU, build core on GPU
        #   If model uses multiple GPUs, build fully on CPU and distribute later
        #   If model uses CPU, build entirely on CPU
        with preprocessing_postprocessing_scope:
            # Dynamic shapes
            chunk_size = chunk_size or self.data_reader.chunk_size
            input_shape = self.get_input_shape(chunk_size=chunk_size)
            output_shape = self.get_output_shape(chunk_size=chunk_size)
            if self.meta_architecture == "parallel":
                input_preprocess_shape = self._compute_preprocess_shape(
                    input_shape, "input", self.input_format
                )
                output_preprocess_shape = self._compute_preprocess_shape(
                    output_shape, "output", self.output_format
                )
            else:
                input_preprocess_shape = self._compute_preprocess_shape(
                    input_shape, "input", None
                )
                output_preprocess_shape = self._compute_preprocess_shape(
                    output_shape, "output", None
                )
            self.build(
                hp=hp,
                input_shape=input_shape,
                output_shape=output_shape,
                input_preprocess_shape=input_preprocess_shape,
                output_preprocess_shape=output_preprocess_shape,
                use_gpu=use_gpu,
                checkpoint=checkpoint,
                batch_size=batch_size,
                submodels=submodels,
            )
        # Data pipeline
        with preprocessing_postprocessing_scope:
            if valid_data is not None:
                # Read dataset
                train_src, valid_src = self.train_data_pipeline(
                    train_data, valid_data, batch_size=batch_size, chunk_size=chunk_size
                )
                # Preprocess dataset
                #   The stars are necessary, for unpacking
                #   (TF datasets calling keras models don"t do that)

                # Enable tensorflow Eager execution
                train_src = train_src.map(
                    lambda x, y: (
                        *self.input_preprocess_model(x, training=True),
                        *self.output_preprocess_model(y, training=True),
                    )
                )
                valid_src = valid_src.map(
                    lambda x, y: (
                        *self.input_preprocess_model(x, training=False),
                        *self.output_preprocess_model(y, training=False),
                    )
                )
            else:
                # Read dataset
                train_src = self.train_data_pipeline(
                    train_data, valid_data, batch_size=batch_size, chunk_size=chunk_size
                )
                valid_src = None
                # Preprocess dataset
                #   The stars are necessary, for unpacking
                #   (TF datasets calling keras models don"t do that)
                train_src = train_src.map(
                    lambda x, y: (
                        *self.input_preprocess_model(x, training=True),
                        *self.output_preprocess_model(y, training=True),
                    )
                )
        # Prefetch
        train_src = train_src.prefetch(tf.data.experimental.AUTOTUNE)
        if valid_src is not None:
            valid_src = valid_src.prefetch(tf.data.experimental.AUTOTUNE)
        # Setup callbacks
        train_callbacks = list(callbacks)
        train_callbacks += self._train_callbacks(count, self.phase, report=report)
        # Set freeze
        if freeze is not None:
            self.freeze = freeze
        # Train
        if self.count_mode == "epochs":
            steps_per_epoch = None
        else:
            steps_per_epoch = 1
        with core_scope:
            self._maybe_customize_model()
            self._compile_model(
                self.core_model,
                self.loss,
                self.optimizer,
                learning_rate,
                clipnorm,
                self.metrics,
            )
            history = self.forward_model.fit(
                train_src,
                epochs=count,
                initial_epoch=self.initial_count,
                steps_per_epoch=steps_per_epoch,
                validation_data=valid_src,
                verbose=0,
                validation_steps=None,
                validation_freq=self.report_freq,
                callbacks=train_callbacks,
            )
        if len(history.epoch):
            self.initial_count = history.epoch[-1] + 1
        readable_history = history.history
        readable_history[self.count_mode] = history.epoch
        return readable_history

    def _extract_num_phases(self, schedule, initial_count=None, limit_count=None):
        # Get number of phases
        num_phases = len(schedule["count"])
        # Ensure lengths are equal
        for k1, v1 in schedule.items():
            if isinstance(v1, (tuple, list)):
                assert num_phases == len(
                    v1
                ), f"Schedule count different length than {k1:s}"
        # Get initial and last phase
        if initial_count is not None:
            initial_phase = np.argmax(np.asarray(schedule["count"]) > initial_count)
        else:
            initial_phase = 0
        if limit_count is not None:
            if schedule["count"][-1] > limit_count:
                num_phases = np.argmin(np.asarray(schedule["count"]) < limit_count) + 1
            else:
                raise ValueError(
                    "Cannot limit count (epoch) for a schedule that never "
                    + "exceeds that count. In a model.search, reduce "
                    + 'max_epochs or increase schedule["count"].'
                )
        # Reduce count of last phase
        if limit_count is not None:
            schedule["count"][num_phases - 1] = limit_count
        return initial_phase, num_phases, schedule

    def _extract_current_hyper_parameters(self, schedule, phase):
        hyper_parameters = {
            k: v[phase] if isinstance(v, (list, tuple)) else v
            for k, v in schedule.items()
        }
        return hyper_parameters

    def train(
        self,
        train_data,
        valid_data,
        schedule,
        eval_fun=None,
        callbacks=None,
        save_after_phases=True,
        limit_epochs=None,
        initial_epoch=None,
        write=True,
        checkpoint=None,
        hp=None,
        submodels=None,
        report=True,
    ):
        """Train the model for multiple training phases.

        A schedule controls training hyperparameters during different phases of
        training. Usually, the first phase is short (one epoch) and used to
        adjust online normalization statistics. These statistics are usually
        frozen after the first phase to ensure stable and reproducible mapping.

        Args:
            train_data:         list of samples/sample function (training)
            valid_data:         list of samples/sample function (validation)
            schedule:           dictionary or function providing a dictionary
                                of keys and training hyperparameters
                                    count           (mandatory)
                                    batch_size      (optional)
                                    learning_rate   (optional)
                                    chunk_size      (optional)
                                    freeze          (optional, default:
                                                     freeze after first phase)
                                to specific values each phase:
                                    int/float (fixed value for all phases)
                                    list/tuple (different value each phase)
                                    None (use default)
            callbacks:          additional keras callbacks
            save_after_phases:  save checkpoint after each phase?
            eval_fun:           opt.: evaluation function at each save
            limit_epochs:       which epoch to pause training at (for search)
            initial_epoch:      from which epoch to continue (for search)
            write:              opt.: write results to json
            checkpoint:         which checkpoint to load ("last", a name, or None)
            hp:                 a keras tuner HyperParameter set

        Returns:
            history:  training history of all losses and metrics

        """
        # Save training_function
        schedule_function = HyperparameterDefinition(
            schedule, "training_function", ID.editor_training_function
        )
        schedule_function.to_py(self.base_model_dir + "/training_function.py")
        # Save model_function
        model_function = ModelDefinition(
            self._core_model_function,
            "model_function",
            ID.editor_model_function,
            data_definition=self.data_definition,
        )
        model_function.to_py(self.base_model_dir + "/model_function.py")
        # # Create Hyperparameter Dictionary
        # Hyperparameters
        if hp is None:
            if self._hp is None:
                hp = HyperParameters()
            else:
                hp = self._hp
        self._hp = hp
        # Instantiate schedule
        if callable(schedule):
            schedule = schedule(hp)
        # Check even length of all schedules
        initial_phase, num_phases, schedule = self._extract_num_phases(
            schedule, initial_epoch, limit_epochs
        )
        # Freeze data_processor variables after first phase if not specified
        if "freeze" not in schedule.keys() or schedule["freeze"] is None:
            schedule["freeze"] = [bool(i) for i in range(num_phases)]
        if self.VERBOSITY > 1:
            self.log("Starting training schedule.")
        try:
            # Training schedule
            for i in range(initial_phase, num_phases):
                if self.VERBOSITY > 1:
                    self.log(f"Starting training phase {i:d}.")
                self.phase = i
                # Get current parameter set
                hyper_parameters = self._extract_current_hyper_parameters(
                    schedule, self.phase
                )
                # Training phase
                try:
                    forward_model = self.forward_model
                    if forward_model is None or not (
                        hasattr(forward_model, "stop_training")
                        and forward_model.stop_training
                    ):
                        self.train_phase(
                            train_data,
                            valid_data,
                            **hyper_parameters,
                            callbacks=callbacks,
                            checkpoint=checkpoint,
                            hp=hp,
                            submodels=submodels,
                            report=report,
                        )
                        if (
                            hasattr(forward_model, "stop_training")
                            and forward_model.stop_training
                        ):
                            if self.optimizer_class in ["lm", "levenberg_marquardt"]:
                                self.warn("Levenberg Marquardt algorithm diverged.")
                            if self.VERBOSITY > 1:
                                self.log("Early Stopping detected.")
                    else:
                        if self.VERBOSITY > 1:
                            self.log("Stopped at previous phase.")
                    if self.VERBOSITY > 1:
                        self.log("Training phase finished successfully.")
                except tf.errors.InvalidArgumentError as e:
                    if "nan in summary" in e.message.lower():
                        if self.VERBOSITY > 0:
                            self.log("Training diverged.")
                    else:
                        raise e
                except KeyboardInterrupt:
                    if self.VERBOSITY > 1:
                        self.log("Training phase interrupted by user.")
                        self.log("... waiting for training schedule interrupt.")
                    time.sleep(3.0)
                # Extract some metrics
                try:
                    forward_model = self.forward_model
                    history = forward_model.history.history
                    history[self.count_mode] = forward_model.history.epoch
                    try:
                        loss = history["loss"][-1]
                        val_loss = history["val_loss"][-1]
                        min_val_loss = min(history["val_loss"])
                        count = history[self.count_mode][-1] + 1
                    except KeyError:
                        loss = -1
                        val_loss = -1
                        min_val_loss = -1
                    # Save
                    if save_after_phases:
                        self.save(f"phase{self.phase:02d}")
                    # Apply eval function to validation set
                    if eval_fun:
                        # Run eval function
                        eval_results = self.infer_data(
                            valid_data,
                            batch_size=hyper_parameters["batch_size"],
                            hp=hp,
                            submodels=submodels,
                        )
                        try:
                            eval_fun(*eval_results, count=count)
                        except TypeError:
                            eval_fun(*eval_results)
                    # Write results to human readable format
                    if write:
                        # Model and test results
                        out_dict = {
                            "schedule": schedule,
                            "train_loss": np.float64(loss),
                            "valid_loss": np.float64(val_loss),
                            "valid_loss_best": np.float64(min_val_loss),
                            "hyper_parameters": hyper_parameters,
                            "history": history,
                        }
                        # Write
                        self._threaded_to_json(
                            out_dict, f"train_results_phase{self.phase:02d}.json"
                        )
                except (AttributeError, KeyError) as e:
                    raise e
            if self.VERBOSITY > 1:
                self.log("Training schedule finished.")
        except KeyboardInterrupt:
            if self.VERBOSITY:
                self.log("Training schedule interrupted by user.")
                self.log("... waiting for further interrupts.")
            time.sleep(3.0)
        except tf.errors.OutOfRangeError:
            if self.VERBOSITY:
                self.log("Training finished: Maximum number of epochs.")
        return history

    def training_schedule(self, *args, **kwargs):
        """Alias for CIDSModel.train(...) for legacy compatibility."""
        return self.train(*args, **kwargs)

    def predict(
        self,
        inputs,
        checkpoint="last",
        use_gpu=False,
        return_states=False,
        hp=None,
        submodels=None,
        preprocess=True,
        postprocess=True,
    ):
        """Compute the output prediction given some inputs.

        Args:
            inputs:         the input tensors
            checkpoint:     which checkpoint to load
            use_gpu:        use the gpu?
            return_states:  also return the states of recurrent layers?
            hp:             a keras tuner HyperParameter set

        Returns:
            outputs:        the prediction of the CIDSModel
            states:         optional: the state of recurrent layers
        """
        # Hyperparameters
        if hp is None:
            if self._hp is None:
                hp = HyperParameters()
            else:
                hp = self._hp
        self._hp = hp
        # Scopes
        preprocessing_postprocessing_scope, core_scope = self._get_execution_scopes(
            use_gpu=use_gpu
        )
        # Build
        with preprocessing_postprocessing_scope:
            self.build(
                hp=hp,
                use_gpu=use_gpu,
                checkpoint=checkpoint,
                submodels=submodels,
            )
        # Change settings
        old_freeze = self.freeze
        self.freeze = True
        # Set correct keras execution scope
        # pylint: disable=not-context-manager
        with tf.keras.backend.learning_phase_scope(0):
            # Preprocessing
            if preprocess:
                if self.VERBOSITY:
                    self.log("Preprocessing...")
                with preprocessing_postprocessing_scope:
                    x = self.input_preprocess_model(inputs, training=False)
            else:
                x = inputs
            # Forward pass
            # pylint: disable=not-callable
            if self.VERBOSITY:
                self.log("Predicting...")
            with core_scope:
                y_ = self.forward_model(x, training=False)
            # Postprocess
            if postprocess:
                if self.VERBOSITY:
                    self.log("Postprocessing...")
                with preprocessing_postprocessing_scope:
                    prediction = self.postprocess_model(y_, training=False)
            else:
                prediction = y_
            # To numpy
            prediction = [y_.numpy() for y_ in prediction]
        # Reset settings
        self.freeze = old_freeze
        #
        if self.VERBOSITY:
            self.log("Prediction completed.")
        # Return
        if not return_states:
            if len(prediction) == 1:
                return prediction[0]
            return prediction
        # Loop over sequence to extract final states
        input_shape = self.built_input_shape
        sequence_length = input_shape[self.sequence_axis["X"]]
        X = [
            tf.pad(
                x[0][:, : (i + 1), ...],
                [[0, 0]]
                + [[0, sequence_length - i - 1]]
                + [[0, 0]] * (len(input_shape) - 2),
            )
            for i in range(sequence_length - 2)
        ]  # this can be unreliable
        X = tf.concat(X, 0)
        states_sequence = self.state_extractor_model(X, training=False)
        states_sequence = [
            np.reshape(
                sv.numpy(),
                [prediction[0].shape[0], -1, sv.numpy().shape[-1]],
                order="F",
            )
            for state in states_sequence
            for sv in state
        ]
        # Return
        return (*prediction, *states_sequence)

    def _unify_results(self, XY, Y_, input_shape, output_shape):
        # Concat batches together: inputs and targets
        XY = list(zip(*XY))
        if isinstance(input_shape, dict) and isinstance(output_shape, dict):
            XY = [
                {k: np.concatenate([batch[k] for batch in xy], axis=0) for k in xy[0]}
                for xy in XY
            ]
        else:
            XY = [np.concatenate(xy, axis=0) for xy in XY]
        # Concat batches together: predictions
        if not isinstance(Y_, (list, tuple)):
            Y_ = (Y_,)
        return *XY, *Y_

    def _forward_pass(self, fn_name, test_src, submodels=None, **kwargs):
        if isinstance(self.output_format, list):
            results = {}
            if submodels is None or not all("parallel" in sub for sub in submodels):
                fun = getattr(self.forward_model, fn_name)
                results[self.output_format[0]] = fun(test_src, **kwargs)
                # TODO: improve assignment
            if submodels is None or any("parallel" in sub for sub in submodels):
                fun = getattr(self.core_model["forward_parallel"], fn_name)
                results[self.output_format[1]] = fun(test_src, **kwargs)
                # TODO: improve assignment
        elif submodels is not None:
            results = {}
            for s in submodels:
                model = self.core_model.get(s, self.core_model.get("subcore_" + s))
                fun = getattr(model, fn_name)
                if "adversarial" in s:
                    # Flip inputs / outputs when prediction adversarial layers
                    test_src_adv = test_src.map(lambda x, y: (y, x))
                    results[s] = fun(test_src_adv, **kwargs)
                else:
                    results[s] = fun(test_src, **kwargs)
        else:
            fun = getattr(self.forward_model, fn_name)
            results = fun(test_src, **kwargs)
        return results

    def infer_data(
        self,
        test_data,
        batch_size=32,
        checkpoint="last",
        postprocess=True,
        use_gpu=False,
        hp=None,
        submodels=None,
    ):
        """Computes and returns inputs, targets and predictions given test data.

        Processes test data, returning inputs, targets and prediction by the
        CIDSModel. Depending on the batch size, not all test data will be
        processed, if batch size is smaller than the number of samples in the
        dataset.

        Args:
            test_data:      the test data, containing inputs and targets
            batch_size:     the number of test samples to process
            checkpoint:     which checkpoint to load
            postprocess:    whether to apply postprocessing
            use_gpu:        use the gpu?
            hp:             a keras tuner HyperParameter sets
            submodels:      list of submodel names to infer over

        Returns:
            X, Y, Y_:       input, target and predictions tensors of batch size
        """
        verbose = int(self.VERBOSITY > 0)
        # Hyperparameters
        if hp is None:
            if self._hp is None:
                hp = HyperParameters()
            else:
                hp = self._hp
        self._hp = hp
        # Scopes
        preprocessing_postprocessing_scope, core_scope = self._get_execution_scopes(
            use_gpu=use_gpu
        )
        # Build
        with preprocessing_postprocessing_scope:
            self.build(
                hp=hp,
                use_gpu=use_gpu,
                checkpoint=checkpoint,
                batch_size=batch_size,
                submodels=submodels,
            )
        # Change settings
        old_freeze = self.freeze
        self.freeze = True
        # Data pipeline
        with preprocessing_postprocessing_scope:
            test_src0 = self.test_data_pipeline(test_data, batch_size=batch_size)
            # Preprocess dataset
            if self.VERBOSITY:
                self.log("Preprocessing...")
            test_src1 = test_src0.map(
                lambda x, y: (
                    *self.input_preprocess_model(x),
                    *self.output_preprocess_model(y),
                )
            )
            # Read raw data or preprocessed data
            input_shape = self.built_input_shape
            output_shape = self.built_output_shape
            if postprocess:
                test_iter0 = test_src0.as_numpy_iterator()
                XY = list(test_iter0)
            else:
                if self.meta_architecture == "parallel":
                    input_shape = self._compute_preprocess_shape(
                        input_shape, "input", self.input_format
                    )
                    output_shape = self._compute_preprocess_shape(
                        output_shape, "output", self.output_format
                    )
                else:
                    input_shape = self._compute_preprocess_shape(
                        input_shape, "input", None
                    )
                    output_shape = self._compute_preprocess_shape(
                        output_shape, "output", None
                    )
                test_iter1 = test_src1.as_numpy_iterator()
                XY = list(test_iter1)
        # Infer
        if self.VERBOSITY:
            self.log("Inferring...")
        with core_scope:
            Y_ = self._forward_pass(
                "predict", test_src1, submodels=submodels, verbose=verbose
            )
        # Postprocessing
        if postprocess:
            if self.VERBOSITY:
                self.log("Postprocessing...")
            with preprocessing_postprocessing_scope:
                Y_ = self.postprocess_model.predict(
                    Y_, batch_size=batch_size, verbose=verbose
                )
        if self.VERBOSITY:
            self.log("Inference completed.")
        # Reset settings
        self.freeze = old_freeze
        # Pack inputs, outputs, predictions
        return self._unify_results(XY, Y_, input_shape, output_shape)

    def _not_empty(self, metrics):
        if isinstance(metrics, dict):
            return any(self._not_empty(v) for v in metrics.values())
        return metrics

    def eval_data(
        self,
        test_data,
        batch_size=32,
        checkpoint="last",
        use_gpu=False,
        metrics=None,
        out_dict=None,
        hp=None,
        write=True,
        submodels=None,
    ):
        """Evaluates losses and metrics of the entire test data.

        Processes test data, returning losses and metrics by the
        CIDSModel. This method will process all test data in batches of batch
        size, until the entire dataset has been consumed. The batch size does
        influence the averaging, use batch size of 1 for canon mean metrics,
        but that is slower.

        Args:
            test_data:      the test data, containing inputs and targets
            batch_size:     the number of test samples to process
            checkpoint:     which checkpoint to load
            use_gpu:        use the gpu?
            metrics:        additional keras metrics callbacks
            out_dict:       dictionary to add to eval results
            hp:             a keras tuner HyperParameter set

        Returns:
            results:        losses and evaluated metrics on test batch
        """
        # Hyperparameters
        if hp is None:
            if self._hp is None:
                hp = HyperParameters()
            else:
                hp = self._hp
        self._hp = hp
        # Scopes
        preprocessing_postprocessing_scope, core_scope = self._get_execution_scopes(
            use_gpu=use_gpu
        )
        # Build
        with preprocessing_postprocessing_scope:
            self.build(
                hp=hp,
                use_gpu=use_gpu,
                checkpoint=checkpoint,
                batch_size=batch_size,
                submodels=submodels,
            )
        # Change settings
        old_freeze = self.freeze
        self.freeze = True
        # Data pipeline
        with preprocessing_postprocessing_scope:
            test_src = self.test_data_pipeline(test_data, batch_size=batch_size)
            # Preprocess dataset
            test_src = test_src.map(
                lambda x, y: (
                    *self.input_preprocess_model(x, training=False),
                    *self.output_preprocess_model(y, training=False),
                )
            )
        # Evaluate
        if self.VERBOSITY:
            self.log("Evaluating...")
        # Distinguish between keras, sklearn and own metrics
        metrics = metrics or self.metrics or []
        keras_metrics, functional_metrics = self._filter_metrics(
            metrics, prioritize="keras"
        )
        results = out_dict or {}
        # Evaluate loss and keras metrics
        with core_scope:
            self._maybe_customize_model()
            self._compile_model(
                self.core_model,
                self.loss,
                self.optimizer,
                metrics=keras_metrics,
            )
            eval_results = self.forward_model.evaluate(
                test_src, verbose=self.VERBOSITY, return_dict=True
            )
            eval_results = self.forward_model.evaluate(
                test_src, verbose=self.VERBOSITY, return_dict=True
            )
            results.update(eval_results)
        # Evaluate Scikit-learn and functional metrics
        if self._not_empty(functional_metrics):
            if self.VERBOSITY:
                self.log(
                    "Computing functional metrics on preprocessed outputs "
                    + "(normalized, standardized, scaled, encoded, ...)."
                )
            _, Y, Y_ = self.infer_data(
                test_data,
                batch_size=batch_size,
                checkpoint=checkpoint,
                use_gpu=use_gpu,
                postprocess=False,
                hp=hp,
                submodels=submodels,
            )
            if isinstance(self.output_format, list):
                functional_results = {
                    m.__name__ + f: m(Y, Y_)
                    for f, mm in zip(self.output_format, functional_metrics.values())
                    for m in mm
                }
            elif isinstance(functional_metrics, dict):
                functional_results = {
                    m.__name__ + k: m(Y, Y_)
                    for k, v in functional_metrics.items()
                    for m in v
                }
            else:
                functional_results = {m.__name__: m(Y, Y_) for m in functional_metrics}
            results.update(functional_results)
        if self.VERBOSITY:
            self.log("Evaluating completed.")
        # Reset settings
        self.freeze = old_freeze
        # Write results to human readable format
        if write:
            # Model and test results
            out_dict = {"eval_results": results, "hyper_parameters": hp}
            # Write
            self._threaded_to_json(out_dict, "eval_results.json")
        return results

    def _ensure_cv_dataset(self, data):
        if not isinstance(data[0], (list, tuple, np.ndarray)):
            raise ValueError(
                "Given data for crossvalidation must be nested (e.g. list of lists)."
            )

    def cross_validate_train(self, train_data, valid_data, schedule, **train_kwargs):
        """Cross-validate and train the model for multiple training phases.

        A schedule controls training hyperparameters during different phases of
        training. Usually, the first phase is short (one epoch) and used to
        adjust online normalization statistics. These statistics are usually
        frozen after the first phase to ensure stable and reproducible mapping.

        Args:
            train_data:         list of lists of samples/sample function (training)
            valid_data:         list of lists of samples/sample function (validation)
            schedule:           dictionary or function providing a dictionary
                                of keys and training hyperparameters
                                    count           (mandatory)
                                    batch_size      (optional)
                                    learning_rate   (optional)
                                    chunk_size      (optional)
                                    freeze          (optional, default:
                                                    freeze after first phase)
                                to specific values each phase:
                                    int/float (fixed value for all phases)
                                    list/tuple (different value each phase)
                                    None (use default)
            train_kwargs:       training keyword arguments

        Returns:
            train_errors:   final losses for each fold
            valid_errors:   final validation losses for each fold

        """
        self.log("Starting training cross validation.")
        histories = []
        self._ensure_cv_dataset(train_data)
        num_folds = len(train_data)
        for fold, (train_fold, valid_fold) in enumerate(zip(train_data, valid_data)):
            # Train each fold
            self.log(f"Training fold {fold + 1} of {num_folds}")
            self.identifier += f"-cv-fold{fold + 1:02d}"
            history = self.train(train_fold, valid_fold, schedule, **train_kwargs)
            self.clear(reset_cached_datasets=True, reset_hps=False)
            self.identifier = self.identifier[: self.identifier.rindex("-cv-fold")]
            histories.append(history)
        return histories

    def cross_validate_evaluate(self, test_data, **eval_kwargs):
        """Cross-validate and evaluates losses and metrics of the entire test data.

        Processes test data, returning losses and metrics by the
        CIDSModel. This method will process all test data in batches of batch
        size, until the entire dataset has been consumed. The batch size does
        influence the averaging, use batch size of 1 for canon mean metrics,
        but that is slower.

        Args:
            test_data:      the test data, containing inputs and targets
            eval_kwargs:    evaluation keyword arguments

        Returns:
            results_list:      losses and metrics evaluated on test data
        """
        self.log("Starting evaluate cross validation.")
        results_list = []
        num_folds = len(test_data)
        for fold, test_fold in enumerate(test_data):
            # Train each fold
            self.log(f"Evaluating fold {fold + 1} of {num_folds}")
            self.identifier += f"-cv-fold{fold + 1:02d}"
            results = self.eval_data(test_fold, **eval_kwargs)
            self.clear(reset_cached_datasets=True, reset_hps=False)
            self.identifier = self.identifier[: self.identifier.rindex("-cv-fold")]
            results_list.append(results)
        return results_list

    def cross_validate_infer(self, test_data, **infer_kwargs):
        """Cross-validate and infer outputs on the entire test data.

        Processes test data, returning losses and metrics by the
        CIDSModel. This method will process all test data in batches of batch
        size, until the entire dataset has been consumed. The batch size does
        influence the averaging, use batch size of 1 for canon mean metrics,
        but that is slower.

        Args:
            test_data:      the test data, containing inputs and targets
            eval_kwargs:    evaluation keyword arguments

        Returns:
            Xs, Ys, Ys_:       input, target and predictions tensors of batch size
        """
        self.log("Starting inference cross validation.")
        Xs = []
        Ys = []
        Ys_ = []
        num_folds = len(test_data)
        for fold, test_fold in enumerate(test_data):
            # Train each fold
            self.log(f"Inferring fold {fold + 1} of {num_folds}")
            self.identifier += f"-cv-fold{fold + 1:02d}"
            X, Y, Y_ = self.infer_data(test_fold, **infer_kwargs)
            self.clear(reset_cached_datasets=True, reset_hps=False)
            self.identifier = self.identifier[: self.identifier.rindex("-cv-fold")]
            Xs.append(X)
            Ys.append(Y)
            Ys_.append(Y_)
        return Xs, Ys, Ys_

    def cross_validate_predict(self, input_array, num_folds, **infer_kwargs):
        """Cross-validate and infer outputs on the entire test data.

        Processes test data, returning losses and metrics by the
        CIDSModel. This method will process all test data in batches of batch
        size, until the entire dataset has been consumed. The batch size does
        influence the averaging, use batch size of 1 for canon mean metrics,
        but that is slower.

        Args:
            test_data:      the test data, containing inputs and targets
            eval_kwargs:    evaluation keyword arguments

        Returns:
            Xs, Ys, Ys_:       input, target and predictions tensors of batch size
        """
        self.log("Starting inference cross validation.")
        Ys_ = []
        for fold in range(num_folds):
            # Train each fold
            self.log(f"Inferring fold {fold + 1} of {num_folds}")
            self.identifier += f"-cv-fold{fold + 1:02d}"
            Y_ = self.predict(input_array, **infer_kwargs)
            self.clear(reset_cached_datasets=True)
            self.identifier = self.identifier[: self.identifier.rindex("-cv-fold")]
            Ys_.append(Y_)
        return Ys_

    def search(
        self,
        *train_args,
        method="hyperband",
        num_trials=10,
        num_brackets=3,
        max_epochs=100,
        executions_per_trial=1,
        objective=None,
        direction=None,
        overwrite=False,
        reload=False,
        **train_kwargs,
    ):
        """Searches for the best possible hyper parameters using keras tuner.

        Keras tuner provides several methods for hyperparameter search. To
        add a hyperparameter to the search space, the schedule in *train_args
        or the keras model must be defined as a function that processes a hp
        argument. This hp argument allows the definition of a range or choice of
        values for each hyper parameters. Outside of the search function, the
        default value is used.

        Examples:

            def architecture_fun(hp):
                keep_prob = hp.Choice("keep_prob", values=[0.4, 0.5, 0.6, 0.7],
                                      default=0.7)
                size_layer_1 = hp.Int("size_layer_1", 100, 400, step=100,
                                      default=400)
                architecture = [
                    tf.keras.layers.LSTM(size_layer_1, return_sequences=True,
                                         dropout=1.0 - keep_prob),
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Dense(len(output_idx), activation=None))]
                return tf.keras.models.Sequential(architecture)

            def schedule_fun(hp):
                schedule = {}
                schedule["count"] = [1, 11, 21, 31, 41]
                schedule["batch_size"] = 64
                schedule["learning_rate"] = hp.Choice(
                    "learning_rate", values=[3e-4, 1e-4, 3e-5, 1e-5],
                    default=1e-4)
                return schedule

            model = CIDSModel.regression(
                data_shape, data_format, input_idx, output_idx,
                architecture_fun, save_freq=save_freq, report_freq=report_freq,
                name=name, identifier="test_cids2", result_dir=result_dir)

            best_hp = model.search(train_samples, valid_samples, schedule_fun,
                               num_trials=10, method="hyperband",
                               overwrite=False, max_epochs=20,
                               executions_per_trial=3,
                               checkpoint=None)

        Args:
            *train_args:            arguments for model.train(...)
            method:                 "random", "hyperband" or "bayesian"
            num_trials:             number of trials (Random/Bayesian search)
            num_brackets:           number of brackets (Hyperband)
            max_epochs:             limits the number of epochs during searches
                                        (Hyperband)
            executions_per_trial:   number of repeated executions per trial
            objective:              objective metric for the search
            direction:              direction for objective function optimization
                                    ("min", "max", or None for automatic, default: None)
            overwrite:              overwrite a previous search
            **train_kwargs:         keyword arguments for model.train(...)

        Returns:
            A dictionary with the best hyper parameters.
        """
        if self.VERBOSITY:
            self.log(f"Starting {method:s} search...")

        # Add schedule and model hyperparameters
        hp = self._hp or HyperParameters()
        self._hp = hp
        schedule = train_kwargs["schedule"]
        if callable(schedule):
            schedule = schedule(self._hp)
        if callable(self._core_model_function):
            self._call_model_function(self._core_model_function, hp)
        # Infer objective and explicitly write direction if given
        objective = objective or self.monitor or "val_loss"
        if direction is not None:
            if isinstance(objective, str):
                objective = Objective(objective, direction)
            else:
                raise ValueError(
                    "Invalid search objective type with direction specified in search: "
                    + f"{type(objective)}"
                )
        if "random" in method:
            oracle = RandomSearchOracle(objective=objective, max_trials=num_trials)
        elif "hyper" in method:
            if self.VERBOSITY:
                num_epochs_expected = (
                    (num_brackets**2) * max_epochs * executions_per_trial
                )
                self.log(f"   Expected total number of epochs: {num_epochs_expected:d}")
            factor = max(2, int(max_epochs ** (1.0 / num_brackets)))
            oracle = HyperbandOracle(
                objective=objective,
                max_epochs=max_epochs,
                hyperband_iterations=1,
                factor=factor,
            )
            if self.save_best_only:
                self.warn(
                    "Hyperband relies on resuming from previous checkpoints."
                    + " Setting self.save_best_only=False to ensure proper checkpoints."
                )
                self.save_best_only = False
        elif "bayes" in method:
            num_initial_points = None  # Defaults to 3 x num_hyperparameters
            beta = 2.6  # (Default) exploitation vs exploration
            oracle = BayesianOptimizationOracle(
                objective=objective,
                max_trials=num_trials,
                num_initial_points=num_initial_points,
                beta=beta,
            )
        else:
            raise ValueError("Unknown hyperparameter optimization method: " + method)
        try:
            # Find previous searches
            previous_searches = glob.glob(os.path.join(self.base_model_dir, "*search*"))
            previous_searches = [
                s
                for s in previous_searches
                if os.path.isdir(s) and os.path.exists(os.path.join(s, "oracle.json"))
            ]
            num_previous_searches = len(previous_searches)
            # Update seeds, if previous search to ensure different
            #  different hyperparameters are sampled
            for _ in previous_searches:
                seed = np.random.get_state()[1][0] + np.prod(
                    np.arange(1, len(previous_searches) + 1)
                )
                projects.set_project_seeds(seed)
            # Set search directories
            if overwrite or reload:
                # Delete and overwrite the old search directory
                search_identifier = f"{min(0, num_previous_searches - 1):02d}"
            else:
                # Create a new search directory
                search_identifier = f"{num_previous_searches:02d}"
            self.meta_folder = "_".join([method, "search", search_identifier])
            tuner = CIDSTuner(
                oracle,
                self,
                executions_per_trial=executions_per_trial,
                overwrite=overwrite,
                directory=self.base_model_dir,
                project_name=self.meta_folder,
            )
            # Reload
            if reload:
                tuner.reload()
            # Ensure oracle uses potentially updated hyperparameter space
            tuner.oracle.hyperparameters = hp.copy()
            # Search
            tuner.search(*train_args, **train_kwargs)
            # Print to file and stdout
            search_summary_file = os.path.join(
                self.base_model_dir, self.meta_folder, "search_summary.txt"
            )
            with open(search_summary_file, "w+", encoding="utf8") as outfile:
                old_stdout = sys.stdout
                sys.stdout = outfile
                tuner.search_space_summary()
                tuner.results_summary()
                sys.stdout = old_stdout
            tuner.search_space_summary()
            tuner.results_summary()
        finally:
            self.meta_folder = ""
        best_hyperparameters = tuner.get_best_hyperparameters()[0]
        return best_hyperparameters

    def get_best_hyperparameters(
        self,
        objective=None,
        search_name="*search*",
        identifier="best",
        default_identifier="default",
    ):
        """Identify best hyperparameters from previous hyperparameter search.

        Args:
            objective (str, option): objective to define best configuration. Defaults to
                self.monitor.
            search_name (str, option): search pattern to identify search dirctories.
                Defaults to "*search*".
            identifier (str, optional): model identifier to set after finding best
                configuration. Defaults to "best".
            default_identifier (str, optional): model identifier to set after finding no
                configuration. Defaults to "default".

        Returns:
            hp: HyperParameters()
        """
        try:
            search_results = SearchResults(
                self, search_name=search_name, objective=objective
            )
            hps = search_results.get_best_hyperparameters(print="best")
            hp = hps[0]
            self.identifier = identifier
            if self.VERBOSITY:
                self.log("Successfully loaded hyperparameters from previous search.")
        except (FileNotFoundError, PermissionError) as e:
            if self.VERBOSITY:
                self.log(">> Hyperparameters: " + str(e))
                self.warn(
                    "Failed to load hyperparameters from previous search. "
                    + "Defaults loaded."
                )
            hp = HyperParameters()
            self.identifier = default_identifier
        return hp

    def analyze(
        self,
        X,
        method="lrp.epsilon",
        fit_data=None,
        checkpoint="last",
        use_gpu=False,
        hp=None,
        submodels=None,
        neuron_selection=None,
        plot=True,
        **kwargs,
    ):
        """Analyze the relevance of each element in the input tensors.

        The output of the model is flattened before computing relevance.

        Args:
            X:                  input tensors to analyze relevance on
            method:             specific innvestigate method or "all"
            fit_data:           optional data to fit the anlyzer on
            checkpoint:         which checkpoint to use
            use_gpu:            use gpu for computations?
            hp:                 keras tuner hyperparameters
            neuron_selection:   output neuron to analyze
            plot:               plot the results?
            **kwargs:           keyword arguments passed on to create_analyzer

        Returns:
            Analysis tensors filled with the relevance for each element of the
            input tensors.

        """
        input_range = [np.min(X), np.max(X)]
        noise_scale = np.std(X) * 0.1
        ri = np.mean(X, axis=0, keepdims=True)
        if method == "all":
            methods = [
                "lrp.epsilon",
                "lrp.z",
                "gradient",
                "smoothgrad",
                "deconvnet",
                "guided_backprop",
                "pattern.net",
                "pattern.attribution",
                "deep_taylor.bounded",
                "input_t_gradient",
                "integrated_gradients",
                "deep_lift.wrapper",
            ]
            for m in methods:
                # TODO: merge multiple analyses
                analysis = self.analyze(
                    X,
                    method=m,
                    fit_data=fit_data,
                    checkpoint=checkpoint,
                    use_gpu=use_gpu,
                    hp=hp,
                    neuron_selection=neuron_selection,
                    plot=plot,
                    **kwargs,
                )
        else:
            analyzer_kwargs = deepcopy(kwargs)
            if method == "lrp.epsilon":
                analyzer_kwargs["epsilon"] = kwargs.get("epsilon", 1)
            elif method == "gradient":
                analyzer_kwargs["postprocess"] = kwargs.get("postprocess", "abs")
            elif method == "smoothgrad":
                analyzer_kwargs["postprocess"] = kwargs.get("postprocess", "abs")
                analyzer_kwargs["noise_scale"] = kwargs.get("noise_scale", noise_scale)
            elif method == "pattern.net":
                analyzer_kwargs["pattern_type"] = kwargs.get("pattern_type", "relu")
            elif method == "pattern.attribution":
                analyzer_kwargs["pattern_type"] = kwargs.get("pattern_type", "relu")
            elif method == "deep_taylor.bounded":
                analyzer_kwargs["low"] = kwargs.get("low", input_range[0])
                analyzer_kwargs["high"] = kwargs.get("high", input_range[1])
            elif method == "integrated_gradients":
                analyzer_kwargs["reference_inputs"] = kwargs.get("reference_inputs", ri)
            elif method == "deep_lift.wrapper":
                analyzer_kwargs["reference_inputs"] = kwargs.get("reference_inputs", ri)

            # Hyperparameters
            if hp is None:
                if self._hp is None:
                    hp = HyperParameters()
                else:
                    hp = self._hp
            self._hp = hp

            # Scopes
            preprocessing_postprocessing_scope, _ = self._get_execution_scopes(
                use_gpu=use_gpu
            )
            # Build
            with preprocessing_postprocessing_scope:
                self.build(
                    hp=hp,
                    use_gpu=use_gpu,
                    checkpoint=checkpoint,
                    submodels=submodels,
                )
            # Change settings
            old_freeze = self.freeze
            self.freeze = True
            # Set correct keras execution scope
            forward_model = self.forward_model
            # pylint: disable=not-context-manager
            with tf.keras.backend.learning_phase_scope(0):
                # Preprocessing
                x = self.input_preprocess_model(X, training=False)[0]
                analyzer = innvestigate.create_analyzer(
                    method, forward_model, **analyzer_kwargs
                )
                analyzer.fit(fit_data, verbose=True)
                analysis = analyzer.analyze(x, neuron_selection=neuron_selection)
            # Reset settings
            self.freeze = old_freeze
            # Plot
            if plot:
                if self.input_format == "NSF":
                    self._plot_analysis_nsf(analysis, X, method)
                else:
                    raise NotImplementedError(
                        "No analysis plotting implemented"
                        + f" for data_format: {self.input_format:s}"
                    )
        return analysis

    def _plot_analysis_nsf(self, analysis, X, method):
        """Plot sequentual data analysis."""
        for n, x in enumerate(X):
            # Plot sample
            num_features = np.shape(X)[-1]
            num_columns = min(3, X.shape[-1])
            num_rows = int(np.ceil(num_features / num_columns))
            sequence_length = np.min(np.argwhere(np.sum(np.abs(x), axis=1) == 0.0))
            if sequence_length == 0:
                sequence_length = X.shape[1]
            figsize = (7.0, num_rows * 7.0 / 2.0 / num_columns)
            fig = plt.figure(figsize=figsize)
            width_ratios = [1.0] * num_columns + [0.05]
            gs = gridspec.GridSpec(num_rows, num_columns + 1, width_ratios=width_ratios)
            cbar_ax = fig.add_subplot(gs[:, -1])
            axes = np.asarray(
                [
                    [fig.add_subplot(gs[r, c]) for c in range(num_columns)]
                    for r in range(num_rows)
                ]
            )
            for f in range(num_features):
                indices = np.unravel_index(f, (num_rows, num_columns), order="C")
                r, c = indices[0], indices[1]
                ax = axes[r, c]
                ax.plot(x[:sequence_length, f], "C0")
                handle = ax.scatter(
                    range(sequence_length),
                    x[:sequence_length, f],
                    c=np.abs(analysis[n, :sequence_length, f]),
                    cmap="Oranges",
                    vmin=0.0,
                    vmax=np.max(np.abs(analysis[n])),
                    alpha=0.5,
                )
                # handle = ax.scatter(
                #     range(sequence_length), Xa[n, :sequence_length, f],
                #     c=analysis[n, :sequence_length, f], cmap="RdBu",
                #     vmin=np.min(analysis), vmax=np.max(analysis), alpha=0.5)
            cbar = plt.colorbar(mappable=handle, cax=cbar_ax, alpha=1.0)
            cbar.solids.set_rasterized(True)
            plt.tight_layout()
            plt.subplots_adjust(left=0.075, right=0.95, top=0.98, bottom=0.035)
            file = f"sample{n:04d}_{method:s}.png"
            path = os.path.join(self.analyze_dir, file)
            plt.savefig(path)
            os.chmod(path, 0o666)
            plt.close(fig)

    def to_json(self, file, **kwargs):
        """Serialize model to human-readable json file.

        Args:
            file:       a json file
            **kwargs:   a dictionary of additional data to store in the json
        """
        out_dict = deepcopy(kwargs)
        # Add model information
        model_dict = {
            "settings": {k: v for k, v in self.__dict__.items() if not callable(v)},
            "input_preprocess_model": self.input_preprocess_model.get_config(),
            "output_preprocnameess_model": self.output_preprocess_model.get_config(),
            "postprocess_model": self.postprocess_model.get_config(),
        }
        # Add core model information
        if isinstance(self.core_model, tf.keras.Model):
            model_dict["core_model"] = self.core_model.get_config()
        elif isinstance(self.core_model, dict):
            for k, cm in self.core_model.items():
                if "subcore_" in k:
                    model_dict[k + "_model"] = cm.get_config()
        else:
            raise ValueError("Invalid core model.")
        # Collect, clean, and output
        out_dict["model"] = model_dict
        # Call BaseModel
        super().to_json(file, **out_dict)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @staticmethod
    def _extract_hp_dict_from_hp_object(hp, exclude_regex):
        # TODO: normally, CIDS does not use kadi notation for json
        #       files. However, Kadi notation is convenient for the
        #       type and validation, so leaving it for now.
        if isinstance(exclude_regex, str):
            exclude_regex = [exclude_regex]
        hp_meta = []
        for item in hp.space:
            if item.name in exclude_regex:
                continue
            paramtype = type(item).__name__
            vtype = ""
            vlist = ""
            name = item.name
            vdefault = item.default
            if paramtype == "Choice":
                vtype = type(item.values[0]).__name__
                vlist = item.values
                validation = {"options": vlist, "required": True}
            elif paramtype == "Fixed":
                vtype = type(item.default).__name__
                validation = {}
            else:
                vtype = type(item.min_value).__name__
                vlist = [item.min_value]
                if item.step is None:
                    step = round((item.max_value - item.min_value) / 10, 2)
                else:
                    step = item.step
                for _ in range(int((item.max_value - item.min_value) / step)):
                    if vtype == "float":
                        vlist.append(round(vlist[-1] + step, str(step)[::-1].find(".")))
                    else:
                        vlist.append(vlist[-1] + step)
                if vdefault not in vlist:
                    vlist.append(vdefault)
                    vlist.sort()
                validation = {"options": vlist, "required": True}
            hp_meta.append(
                {
                    "key": name,
                    "type": vtype,
                    "validation": validation,
                    "value": vdefault,
                }
            )
        return hp_meta


CIDSModel = CIDSModelTF
