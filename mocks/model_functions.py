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
"""Collection of model functions (CIDS with Tensorflow). Part of the CIDS toolbox."""
import numpy as np
import tensorflow as tf
from kerastuner import HyperParameters

from ..data import DataDefinition
from .layers import Sampling


def _get_output_shape(data_definition: DataDefinition = None):
    try:
        return data_definition.data_shape["Y"]
    except TypeError as e:
        raise ValueError(
            "DataDefinition requires input_features or output_features to return "
            + "a data_shape"
        ) from e


def _get_input_shape(data_definition: DataDefinition = None):
    try:
        return data_definition.data_shape["X"]
    except TypeError as e:
        raise ValueError(
            "DataDefinition requires input_features or output_features to return "
            + "a data_shape"
        ) from e


def _check_buildable(layers, input_shape):
    if len(layers):
        try:
            tmp_layers = [
                layer.__class__.from_config(layer.get_config()) for layer in layers
            ]
            tmp_model = tf.keras.models.Sequential(tmp_layers)
            tmp_model.build(input_shape=input_shape)
            del tmp_model
            # tf.keras.backend.clear_session()
            # gc.collect()
        except (tf.errors.InvalidArgumentError, ValueError):
            return False
    return True


def simple_dense_model_function(
    hp: HyperParameters, data_definition: DataDefinition, time_distribute=False
):
    """Construct simple dense neural network models.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition
        time_distribute (bool, optional): distribute layers in time? Defaults to False.

    Returns:
        tf.keras.models.Sequential: a keras sequential model
    """
    # Read global data definition
    num_output_units = _get_output_shape(data_definition)[-1]
    # Hyperparameters
    num_layers = hp.Int("num_layers", 1, 5, step=1, default=1)
    activation = hp.Choice(
        "activation", ["relu", "sigmoid", "tanh", "elu", "softplus"], default="relu"
    )
    num_units = hp.Int("units", 4, 1024, step=4, default=32)
    # Assemble
    layers = []
    layers.append(tf.keras.layers.Flatten())
    for _ in range(num_layers):
        layers.append(tf.keras.layers.Dense(num_units, activation=activation))
    layers.append(tf.keras.layers.Dense(num_output_units, activation=None))
    # Time distribute
    if time_distribute:
        layers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
    return tf.keras.models.Sequential(layers)


def dense_model_function(
    hp: HyperParameters, data_definition: DataDefinition, time_distribute=False
):
    """Construct dense neural network models.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition
        time_distribute (bool, optional): distribute layers in time? Defaults to False.

    Returns:
        tf.keras.models.Sequential: a keras sequential model
    """
    # Read global data definition
    num_output_units = _get_output_shape(data_definition)[-1]
    # Hyperparameters
    num_layers = hp.Int("num_layers", 1, 5, step=1, default=1)
    activation = hp.Choice(
        "activation", ["relu", "sigmoid", "tanh", "elu", "softplus"], default="relu"
    )
    num_units = hp.Int("units", 4, 4096, step=4, default=128)
    # Assemble
    layers = []
    layers.append(tf.keras.layers.Flatten())
    for _ in range(num_layers):
        layers.append(tf.keras.layers.Dense(num_units, activation=activation))
    layers.append(tf.keras.layers.Dense(num_output_units, activation=None))
    # Time distribute
    if time_distribute:
        layers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
    return tf.keras.models.Sequential(layers)


def dense_dropout_model_function(
    hp: HyperParameters, data_definition: DataDefinition, time_distribute=False
):
    """Construct dense neural network models with dropout.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition
        time_distribute (bool, optional): distribute layers in time? Defaults to False.

    Returns:
        tf.keras.models.Sequential: a keras sequential model
    """
    # Read global data definition
    num_output_units = _get_output_shape(data_definition)[-1]
    # Hyperparameters
    num_layers = hp.Int("num_layers", 1, 5, step=1, default=1)
    activation = hp.Choice(
        "activation", ["relu", "sigmoid", "tanh", "elu", "softplus"], default="relu"
    )
    num_units = hp.Int("num_units", 4, 4096, step=4, default=128)
    dropout_rate = hp.Float("dropout_rate", 0.0, 1.0, default=0.0)
    # Assemble
    layers = []
    layers.append(tf.keras.layers.Flatten())
    for _ in range(num_layers):
        layers.append(tf.keras.layers.Dense(num_units, activation=activation))
        if dropout_rate > 0.0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(num_output_units, activation=None))
    # Time distribute
    if time_distribute:
        layers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
    return tf.keras.models.Sequential(layers)


def auto_convnet_model_function(hp: HyperParameters, data_definition: DataDefinition):
    """Construct convolutional neural network models.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition

    Returns:
        tf.keras.models.Sequential: a keras sequential model
    """
    # Read global data definition
    input_shape = _get_input_shape(data_definition)
    num_output_units = _get_output_shape(data_definition)[-1]

    # Hyperparameters
    conv_layers = hp.Int("conv_layers", 1, 5, step=1, default=3)
    conv_size = hp.Choice("conv_size", [3, 5, 7], default=3)
    conv_strides = hp.Int("conv_strides", 1, 5, step=1, default=1)
    conv_units = hp.Choice("conv_units", [16, 32, 64, 128, 256], default=32)
    conv_dropout = hp.Float("conv_dropout", 0.0, 0.5, step=0.05, default=0.0)
    pool = hp.Choice("pool", ["None", "max", "avg"], default="avg")
    pool_size = hp.Choice("pool_size", [2, 4], default=2)
    num_units_dense = hp.Choice("num_units_dense", [32, 64, 128, 256, 512], default=64)
    num_layers_dense = hp.Int("num_layers_dense", 1, 3, step=1, default=1)
    dense_dropout = hp.Float("dense_dropout", 0.0, 0.5, step=0.05, default=0.0)
    num_conv_blocks = 999999
    # Assemble
    layers = []
    # Convnet part
    for _ in range(num_conv_blocks):
        block_layers = []
        for __ in range(conv_layers):
            block_layers.append(
                tf.keras.layers.Conv3D(
                    conv_units,
                    conv_size,
                    conv_strides,
                    data_format="channels_last",
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            if not _check_buildable(layers + block_layers, input_shape):
                block_layers = block_layers[:-1]
        if len(block_layers) > 0:
            # layers.append(tf.keras.layers.Activation("relu"))
            block_layers.append(tf.keras.layers.Dropout(conv_dropout))
            if pool == "avg":
                block_layers.append(tf.keras.layers.AvgPool3D(pool_size))
            elif pool == "max":
                block_layers.append(tf.keras.layers.MaxPool3D(pool_size))
            elif pool is None or pool == "None":
                pass
            else:
                raise ValueError("Unknown pooling type: " + pool)
            if not _check_buildable(layers + block_layers, input_shape):
                block_layers = block_layers[:-1]
            layers += block_layers
        else:
            layers += block_layers
            break
    # Dense decoder
    layers.append(tf.keras.layers.Flatten())
    for _ in range(num_layers_dense):
        layers.append(
            tf.keras.layers.Dense(
                num_units_dense, activation="relu", kernel_initializer="he_normal"
            )
        )
        layers.append(tf.keras.layers.Dropout(dense_dropout))
    # Last layer
    layers.append(tf.keras.layers.Dense(num_output_units, activation=None))
    return tf.keras.models.Sequential(layers)


def conv_vae_model_function(
    hp: HyperParameters, data_definition: DataDefinition, batch_normalization=True
):
    """Construct convolutional neural network with submodels.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition
        Batch_Normalization(bool, optional): toggle to switch on
                                        batch normalization layer

    Returns:
        model_dict: model dictionary with Submodel layers
    """
    # Read global data definition

    num_output_units = _get_output_shape(data_definition)[-1]

    # Hyperparameters
    num_layers_encoder = hp.Int("num_layers_encoder", 1, 5, step=1, default=4)
    num_layers_decoder = hp.Int("num_layers_decoder", 1, 5, step=1, default=3)
    conv_units = hp.Choice("conv_units", [16, 32, 64, 128, 256], default=32)
    conv_size = hp.Choice("conv_size", [3, 5, 7], default=3)
    conv_strides = hp.Int("conv_strides", 1, 5, step=1, default=2)
    # Latent_dimension
    latent_dim = 128
    # Assemble

    # Encoder
    # Encoder output units
    encoder_output_units = latent_dim * 2
    layers = []

    for _ in range(num_layers_encoder):
        layers.append(
            tf.keras.layers.Conv3D(
                conv_units,
                conv_size,
                conv_strides,
                padding="same",
                activation="relu",
            )
        )
        if batch_normalization:
            layers.append(tf.keras.layers.BatchNormalization())
    # output_layer_encoder
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(encoder_output_units))

    encoder = tf.keras.Sequential(layers)

    # Sampler
    layers = []
    layers.append(
        Sampling(
            add_sampling_loss=True,
            mode="repar",
            latent_dim=latent_dim * 2,
            use_input_as_seed=False,
        )
    )
    sampler = tf.keras.Sequential(layers)

    # Decoder
    layers = []
    input_dense_unit = 8
    decoder_input_shape = (
        input_dense_unit,
        input_dense_unit,
        input_dense_unit,
        latent_dim,
    )
    input_units = np.prod(decoder_input_shape)
    layers.append(tf.keras.layers.Dense(input_units, activation="relu"))
    layers.append(tf.keras.layers.Reshape(decoder_input_shape))
    for _ in range(num_layers_decoder):
        layers.append(
            tf.keras.layers.Conv3DTranspose(
                conv_units,
                conv_size,
                conv_strides,
                padding="same",
                activation="relu",
            )
        )
        if batch_normalization:
            layers.append(tf.keras.layers.BatchNormalization())

    # output_layer_decoder
    layers.append(
        tf.keras.layers.Conv3DTranspose(
            num_output_units,
            conv_size,
            conv_strides,
            padding="same",
            activation="sigmoid",
        )
    )

    decoder = tf.keras.Sequential(layers)
    # Gather models
    model_dict = {}
    model_dict["encoder"] = encoder
    model_dict["sampler"] = sampler
    model_dict["decoder"] = decoder

    return model_dict


def con_vae_regressor_model_function(
    hp: HyperParameters, data_definition: DataDefinition, batch_normalization=True
):

    """Construct convolutional neural network with submodels.

    Args:
        hp (HyperParameters): a kerastuner hyperparameter object
        data_definition (DataDefinition): a cids data definition
        Batch_Normalization(bool, optional): toggle to switch on
                                        batch normalization layer

    Returns:
        model_dict: model dictionary with Submodel layers
    """

    # Hyperparameters
    num_layers_encoder = hp.Int("num_layers_encoder", 1, 5, step=1, default=4)
    num_layers_decoder = hp.Int("num_layers_decoder", 1, 5, step=1, default=3)
    conv_units = hp.Choice("conv_units", [16, 32, 64, 128, 256], default=32)
    num_layers_regressor = hp.Int("num_layers_regressor", 1, 5, step=1, default=1)
    num_output_units = hp.Int("num_layers_regressor", 1, 5, step=1, default=4)
    conv_size = hp.Choice("conv_size", [3, 5, 7], default=3)
    conv_strides = hp.Int("conv_strides", 1, 5, step=1, default=2)
    # Latent_dimension
    latent_dim = 128
    # Assemble
    # Encoder
    # Encoder output units
    encoder_output_units = latent_dim * 2
    layers = []

    for _ in range(num_layers_encoder):
        layers.append(
            tf.keras.layers.Conv2D(
                conv_units,
                conv_size,
                conv_strides,
                padding="same",
                activation="relu",
            )
        )
        if batch_normalization:
            layers.append(tf.keras.layers.BatchNormalization())
    # output_layer_encoder
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(encoder_output_units))

    encoder = tf.keras.Sequential(layers)

    # Sampler
    layers = []
    layers.append(
        Sampling(
            add_sampling_loss=True,
            mode="repar",
            latent_dim=latent_dim * 2,
            use_input_as_seed=False,
        )
    )
    sampler = tf.keras.Sequential(layers)

    # Decoder
    layers = []
    input_dense_unit = 8
    decoder_input_shape = (
        input_dense_unit,
        input_dense_unit,
        latent_dim,
    )
    input_units = np.prod(decoder_input_shape)
    layers.append(tf.keras.layers.Dense(input_units, activation="relu"))
    layers.append(tf.keras.layers.Reshape(decoder_input_shape))
    for _ in range(num_layers_decoder):
        layers.append(
            tf.keras.layers.Conv2DTranspose(
                conv_units,
                conv_size,
                conv_strides,
                padding="same",
                activation="relu",
            )
        )
        if batch_normalization:
            layers.append(tf.keras.layers.BatchNormalization())

    # output_layer_decoder
    layers.append(
        tf.keras.layers.Conv2DTranspose(
            num_output_units,
            conv_size,
            conv_strides,
            padding="same",
            activation="sigmoid",
        )
    )
    decoder = tf.keras.Sequential(layers)
    layers = []
    for _ in range(num_layers_regressor):
        layers.append(tf.keras.layers.Dense(64))
    layers.append(tf.keras.layers.Dense(num_output_units))
    regression = tf.keras.Sequential(layers)
    # Gather models
    model_dict = {}
    model_dict["encoder"] = encoder
    model_dict["sampler"] = sampler
    model_dict["decoder"] = decoder
    model_dict["parallel_regression"] = regression

    return model_dict
