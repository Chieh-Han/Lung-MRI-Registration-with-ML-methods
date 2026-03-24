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
"""Tensorflow layer implementations for CIDS. Part of the CIDS toolbox.

    Classes:
        StateWrapper:           Wraps recurrent neural network state into output
        UNet:                   U-Net implementation
        TimeUNet:               Time-dependent U-Net implementation
        LayerNormalization:     Layer-normalization layer
        MaskedTimeDistributed:  Time distributed layer that passes the mask on
        Sampling:               Sampling layer for Autoencoders1
        NonlinearRegression:    Nonlinear regression layer with exponential kernel
"""
"""
tensorflow/keras layers for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
import copy
import os
import warnings

import neurite as ne
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as KI
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

from . import utility
# local utility

###############################################################################
# vxm layers
###############################################################################


class SpatialTransformer(Layer):
    """
    N-dimensional (ND) spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(
        self,
        interp_method="linear",
        single_transform=False,
        fill_value=None,
        shift_center=True,
        shape=None,
        **kwargs,
    ):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms. Assumes the input and output spaces are identical.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used. Incompatible with `shift_center=True`.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if "indexing" in kwargs:
            del kwargs["indexing"]
            warnings.warn(
                "The `indexing` argument to SpatialTransformer no longer exists. If you "
                "loaded a model, save it again to be able to load it in the future."
            )

        self.interp_method = interp_method
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "interp_method": self.interp_method,
                "single_transform": self.single_transform,
                "fill_value": self.fill_value,
                "shift_center": self.shift_center,
                "shape": self.shape,
            }
        )
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError(
                "Spatial Transformer must be called on a list of length 2: "
                "first argument is the image, second is the transform."
            )

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]

        # make sure transform has reasonable shape (is_affine_shape throws error if not)
        if not utility.is_affine_shape(input_shape[1][1:]):
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(
                    f"Dense transform shape {dense_shape} does not match "
                    f"image shape {image_shape}."
                )

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1] or [B, N+1, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]), vol)
        else:
            return tf.map_fn(
                self._single_transform, [vol, trf], fn_output_signature=vol.dtype
            )

    def _single_transform(self, inputs):
        return utility.transform(
            inputs[0],
            inputs[1],
            interp_method=self.interp_method,
            fill_value=self.fill_value,
            shift_center=self.shift_center,
            shape=self.shape,
        )


class VecInt(Layer):
    """
    Vector integration layer

    Enables vector integration via several methods (ode or quadrature for
    time-dependent vector fields and scaling-and-squaring for stationary fields)

    If you find this function useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(
        self,
        method="ss",
        int_steps=7,
        out_time_pt=1,
        ode_args=None,
        odeint_fn=None,
        **kwargs,
    ):
        """
        Parameters:
            method: Must be any of the methods in neuron.utils.integrate_vec.
            int_steps: Number of integration steps.
            out_time_pt: Time point at which to output if using odeint integration.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        # TODO: remove this block
        # load models saved with the `indexing` argument
        if "indexing" in kwargs:
            del kwargs["indexing"]
            warnings.warn(
                "The `indexing` argument to VecInt no longer exists. If you loaded a "
                "model, save it again to be able to load it in the future."
            )

        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {"rtol": 1e-6, "atol": 1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "method": self.method,
                "int_steps": self.int_steps,
                "out_time_pt": self.out_time_pt,
                "ode_args": self.ode_args,
                "odeint_fn": self.odeint_fn,
            }
        )
        return config

    def build(self, input_shape):
        # confirm built
        self.built = True

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception(
                "transform ndims %d does not match expected ndims %d"
                % (trf_shape[-1], len(trf_shape) - 2)
            )

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi-gpu models
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], "_keras_shape"):
            loc_shift._keras_shape = inputs[0]._keras_shape

        if len(inputs) > 1:
            assert (
                self.out_time_pt is None
            ), "out_time_pt should be None if providing batch_based out_time_pt"

        # map transform across batch
        out = tf.map_fn(
            self._single_int,
            [loc_shift] + inputs[1:],
            fn_output_signature=loc_shift.dtype,
        )
        if hasattr(inputs[0], "_keras_shape"):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return utility.integrate_vec(
            vel,
            method=self.method,
            nb_steps=self.int_steps,
            ode_args=self.ode_args,
            out_time_pt=out_time_pt,
            odeint_fn=self.odeint_fn,
        )


def default_vxm_unet_features():
    nb_features = [[16, 32, 32, 32], 
                   [32, 32, 32, 32, 32, 16, 16]]  # encoder  # decoder
    return nb_features


def _conv_block(
    x,
    nfeat,
    strides=1,
    name=None,
    do_res=False,
    hyp_tensor=None,
    include_activation=True,
    kernel_initializer="he_normal",
):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), "ndims should be one of 1, 2, or 3. found: %d" % ndims

    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, "HyperConv%dDFromDense" % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, "Conv%dD" % ndims)
        extra_conv_params["kernel_initializer"] = kernel_initializer
        conv_inputs = x

    convolved = Conv(
        nfeat,
        kernel_size=3,
        padding="same",
        strides=strides,
        name=name,
        **extra_conv_params,
    )(conv_inputs)

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print(
            "note: this is a weird thing to do, since its not really residual training anymore"
        )
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(
                nfeat,
                kernel_size=3,
                padding="same",
                name="resfix_" + name,
                **extra_conv_params,
            )(conv_inputs)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    if include_activation:
        name = name + "_activation" if name else None
        convolved = KL.LeakyReLU(0.2, name=name)(convolved)

    return convolved


def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), "ndims should be one of 1, 2, or 3. found: %d" % ndims
    UpSampling = getattr(KL, "UpSampling%dD" % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    name = name + "_concat" if name else None
    return KL.concatenate([upsampled, connection], name=name)


class VXM_Unet(tf.keras.Model):
    """
    A unet architecture that builds off either an input keras model or input shape. Layer features
    can be specified directly as a list of encoder and decoder features or as a single integer along
    with a number of unet levels. The default network features per layer (when no options are
    specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(
        self,
        inshape=None,
        input_model=None,
        nb_features=None,
        nb_levels=None,
        max_pool=2,
        feat_mult=1,
        nb_conv_per_level=1,
        do_res=False,
        nb_upsample_skips=0,
        hyp_input=None,
        hyp_tensor=None,
        final_activation_function=None,
        kernel_initializer="he_normal",
        name="unet",
    ):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            nb_upsample_skips: Number of upsamples to skip in the decoder (to downsize the
                the output resolution). Default is 0.
            hyp_input: Hypernetwork input tensor. Enables HyperConvs if provided. Default is None.
            hyp_tensor: Hypernetwork final tensor. Enables HyperConvs if provided. Default is None.
            final_activation_function: Replace default activation function in final layer of unet.
            kernel_initializer: Initializer for the kernel weights matrix for conv layers. Default
                is 'he_normal'.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError("inshape must be supplied if input_model is None")
            unet_input = KL.Input(shape=inshape, name="%s_input" % name)
            model_inputs = [unet_input]
        else:
            if len(input_model.outputs) == 1:
                unet_input = input_model.outputs[0]
            else:
                unet_input = KL.concatenate(
                    input_model.outputs, name="%s_input_concat" % name
                )
            model_inputs = input_model.inputs
        
        # add hyp_input tensor if provided
        if hyp_input is not None and not any(
            [hyp_input is inp for inp in model_inputs]
        ):
            model_inputs = model_inputs + [hyp_input]

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_vxm_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )
        MaxPooling = getattr(KL, "MaxPooling%dD" % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = "%s_enc_conv_%d_%d" % (name, level, conv)
                last = _conv_block(
                    last,
                    nf,
                    name=layer_name,
                    do_res=do_res,
                    hyp_tensor=hyp_tensor,
                    kernel_initializer=kernel_initializer,
                )
            enc_layers.append(last)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(
                max_pool[level], name="%s_enc_pooling_%d" % (name, level)
            )(last)

        # if final_activation_function is set, we need to build a utility that checks
        # which layer is truly the last, so we know not to apply the activation there
        if final_activation_function is not None and len(final_convs) == 0:
            activate = lambda lvl, c: not (
                lvl == (nb_levels - 2) and c == (nb_conv_per_level - 1)
            )
        else:
            activate = lambda lvl, c: True

        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = "%s_dec_conv_%d_%d" % (name, real_level, conv)
                last = _conv_block(
                    last,
                    nf,
                    name=layer_name,
                    do_res=do_res,
                    hyp_tensor=hyp_tensor,
                    include_activation=activate(level, conv),
                    kernel_initializer=kernel_initializer,
                )

            # upsample
            if level < (nb_levels - 1 - nb_upsample_skips):
                layer_name = "%s_dec_upsample_%d" % (name, real_level)
                last = _upsample_block(
                    last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name
                )

        # now build function to check which of the 'final convs' is really the last
        if final_activation_function is not None:
            activate = lambda n: n != (len(final_convs) - 1)
        else:
            activate = lambda n: True

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = "%s_dec_final_conv_%d" % (name, num)
            last = _conv_block(
                last,
                nf,
                name=layer_name,
                hyp_tensor=hyp_tensor,
                include_activation=activate(num),
                kernel_initializer=kernel_initializer,
            )

        # add the final activation function is set
        if final_activation_function is not None:
            last = KL.Activation(
                final_activation_function, name="%s_final_activation" % name
            )(last)

        super().__init__(inputs=model_inputs, outputs=last, name=name)


###############################################################################
# neurite layers
###############################################################################


class Negate(Layer):
    """
    Keras Layer: negative of the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class SampleNormalLogVar(Layer):
    """
    Keras Layer: Gaussian sample given mean and log_variance

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
        CVPR 2018. https://arxiv.org/abs/1903.03148

    inputs: list of Tensors [mu, log_var]
    outputs: Tensor sample from N(mu, sigma^2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return self._sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _sample(self, args):
        """
        sample from a normal distribution

        args should be [mu, log_var], where log_var is the log of the squared sigma

        This is probably equivalent to
            K.random_normal(shape, args[0], exp(args[1]/2.0))
        """
        mu, log_var = args

        # sample from N(0, 1)
        noise = tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # make it a sample from N(mu, sigma^2)
        z = mu + tf.exp(log_var / 2.0) * noise
        return z


# full wording.

VecIntegration = VecInt


class RescaleTransform(Layer):
    """
    Rescale transform layer

    Rescales a dense or affine transform. If dense, this involves resizing and
    rescaling the vector field.
    """

    def __init__(self, zoom_factor, interp_method="linear", **kwargs):
        """
        Parameters:
            zoom_factor: Scaling factor.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "zoom_factor": self.zoom_factor,
                "interp_method": self.interp_method,
            }
        )
        return config

    def build(self, input_shape):
        # check if transform is affine
        self.is_affine = utility.is_affine_shape(input_shape[1:])
        self.ndims = input_shape[-1] - 1 if self.is_affine else input_shape[-1]

    def compute_output_shape(self, input_shape):
        if self.is_affine:
            return (input_shape[0], self.ndims, self.ndims + 1)
        else:
            shape = [int(d * self.zoom_factor) for d in input_shape[1:-1]]
            return (input_shape[0], *shape, self.ndims)

    def call(self, transform):
        """
        Parameters
            transform: Transform to rescale. Either a dense warp of shape [B, D1, ..., DN, N]
            or an affine matrix of shape [B, N, N+1].
        """
        if self.is_affine:
            return utility.rescale_affine(transform, self.zoom_factor)
        else:
            return utility.rescale_dense_transform(
                transform, self.zoom_factor, interp_method=self.interp_method
            )


class ComposeTransform(Layer):
    """
    Composes a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = ComposeTransform()([A, B, C])
    """

    def __init__(self, interp_method="linear", shift_center=True, shape=None, **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            shift_center: Shift grid to image center when converting matrices to dense transforms.
            shape: ND output shape used for converting matrices to dense transforms. Includes only
                the N spatial dimensions. Only used once, if the rightmost transform is a matrix.
                If None or if the rightmost transform is a warp, the shape of the rightmost warp
                will be used. Incompatible with `shift_center=True`.
        """
        self.interp_method = interp_method
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "interp_method": self.interp_method,
                "shift_center": self.shift_center,
                "shape": self.shape,
            }
        )
        return config

    def build(self, input_shape, **kwargs):

        # sanity check on the inputs
        if not isinstance(input_shape, (list, tuple)):
            raise Exception("ComposeTransform must be called for a list of transforms.")

    def call(self, transforms):
        """
        Parameters:
            transforms: List of affine or dense transforms to compose.
        """
        if len(transforms) == 1:
            return transforms[0]

        compose = lambda trf: utility.compose(
            trf,
            interp_method=self.interp_method,
            shift_center=self.shift_center,
            shape=self.shape,
        )
        return tf.map_fn(compose, transforms, fn_output_signature=transforms[0].dtype)


class AddIdentity(Layer):
    """
    Adds the identity matrix to the input. This is useful when predicting
    affine parameters directly.
    """

    def build(self, input_shape):
        shape = input_shape[1:]

        if len(shape) == 1:
            # let's support 1D flattened affines here, since it's
            # likely the input is coming right from a dense layer
            length = shape[0]
            if length == 6:
                self.ndims = 2
            elif length == 12:
                self.ndims = 3
            else:
                raise ValueError(
                    "Flat affine must be of length 6 (2D) or 12 (3D), got {length}."
                )
            self.nrows = self.ndims
        elif len(shape) == 2:
            # or it could be a 2D matrix
            utility.validate_affine_shape(input_shape)
            self.ndims = shape[1] - 1
            self.nrows = shape[0]
        else:
            raise ValueError(
                "Input to AddIdentity must be a flat 1D array or 2D matrix, "
                f"got shape {input_shape}."
            )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nrows, self.ndims + 1)

    def call(self, transform):
        """
        Parameters
            transform: Affine transform of shape [B, N, N+1] or [B, N+1, N+1] or [B, N*(N+1)].
        """
        transform = tf.reshape(transform, (-1, self.nrows, self.ndims + 1))
        return utility.affine_add_identity(transform)


class InvertAffine(Layer):
    """
    Inverts an affine transform.
    """

    def build(self, input_shape):
        utility.validate_affine_shape(input_shape)
        self.nrows = input_shape[-2]
        self.ndims = input_shape[-1] - 1

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nrows, self.ndims + 1)

    def call(self, matrix):
        """
        Parameters
            matrix: Affine matrix of shape [B, N, N+1] or [B, N+1, N+1] to invert.
        """
        return utility.invert_affine(matrix)


class ParamsToAffineMatrix(Layer):
    """
    Constructs an affine transformation matrix from translation, rotation, scaling and shearing
    parameters in 2D or 3D.

    If you find this layer useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self, ndims=3, deg=True, shift_scale=False, last_row=False, **kwargs):
        """
        Parameters:
            ndims: Dimensionality of transform matrices. Must be 2 or 3.
            deg: Whether the input rotations are specified in degrees.
            shift_scale: Add 1 to any specified scaling parameters. This may be desirable
                when the parameters are estimated by a network.
            last_row: Whether to return a full matrix, including the last row.
        """
        self.ndims = ndims
        self.deg = deg
        self.shift_scale = shift_scale
        self.last_row = last_row
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "ndims": self.ndims,
                "deg": self.deg,
                "shift_scale": self.shift_scale,
                "last_row": self.last_row,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims + int(self.last_row), self.ndims + 1)

    def call(self, params):
        """
        Parameters:
            params: Parameters as a vector which corresponds to translations, rotations, scaling
                    and shear. The size of the last axis must not exceed (N, N+1), for N
                    dimensions. If the size is less than that, the missing parameters will be
                    set to the identity.
        """
        return utility.params_to_affine_matrix(
            par=params,
            deg=self.deg,
            shift_scale=self.shift_scale,
            ndims=self.ndims,
            last_row=self.last_row,
        )


class AffineToDenseShift(Layer):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
        """
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "shape": self.shape,
                "shift_center": self.shift_center,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)

    def build(self, input_shape):
        utility.validate_affine_shape(input_shape)

    def call(self, mat):
        """
        Parameters:
            mat: Affine matrices of shape (B, N, N+1).
        """
        return utility.affine_to_dense_shift(
            mat, self.shape, shift_center=self.shift_center
        )


class DrawAffineParams(Layer):
    """
    Draw translation, rotation, scaling and shearing parameters defining an affine transform in
    N-dimensional space, where N is 2 or 3. Choose parameters wisely: there is no check for
    negative or zero scaling! The batch dimension will be inferred from the input tensor.

    Returns:
        A tuple of tensors with shapes (..., N), (..., M), (..., N), and (..., M) defining
        translation, rotation, scaling, and shear, respectively, where M is 3 in 3D and 1 in 2D.
        With `concat=True`, the layer will concatenate the output along the last dimension.

    See also:
        ParamsToAffineMatrix

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(
        self,
        shift=None,
        rot=None,
        scale=None,
        shear=None,
        normal_shift=False,
        normal_rot=False,
        normal_scale=False,
        normal_shear=False,
        shift_scale=False,
        ndims=3,
        concat=True,
        out_type=tf.float32,
        seeds={},
        **kwargs,
    ):
        """
        Parameters:
            shift: Translation sampling range x around identity. Values will be sampled uniformly
                from [-x, x]. When sampling from a normal distribution, x is the standard
                deviation (SD). The same x will be used for each dimension, unless an iterable of
                length N is passed, specifying a value separately for each axis. None means 0.
            rot: Rotation sampling range (see `shift`). Accepts only one value in 2D.
            scale: Scaling sampling range x. Parameters will be sampled around identity as for
                `shift`, unless `shift_scale` is set. When sampling normally, scaling parameters
                will be truncated beyond two standard deviations.
            shear: Shear sampling range (see `shift`). Accepts only one value in 2D.
            normal_shift: Sample translations normally rather than uniformly.
            normal_rot: See `normal_shift`.
            normal_scale: Draw scaling parameters normally, truncating beyond 2 SDs.
            normal_shear: See `normal_shift`.
            shift_scale: Add 1 to any drawn scaling parameter When sampling uniformly, this will
                result in scaling parameters falling in [1 - x, 1 + x] instead of [-x, x].
            ndims: Number of dimensions. Must be 2 or 3.
            normal: Sample parameters normally instead of uniformly.
            concat: Concatenate the output along the last axis to return a single tensor.
            out_type: Floating-point output data type.
            seeds: Dictionary of integers for reproducible randomization. Keywords must be in
                ('shift', 'rot', 'scale', 'shear').
        """
        self.shift = shift
        self.rot = rot
        self.scale = scale
        self.shear = shear
        self.normal_shift = normal_shift
        self.normal_rot = normal_rot
        self.normal_scale = normal_scale
        self.normal_shear = normal_shear
        self.shift_scale = shift_scale
        self.ndims = ndims
        self.concat = concat
        self.out_type = tf.dtypes.as_dtype(out_type)
        self.seeds = seeds
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "shift": self.shift,
                "rot": self.rot,
                "scale": self.scale,
                "shear": self.shear,
                "normal_shift": self.normal_shift,
                "normal_rot": self.normal_rot,
                "normal_scale": self.normal_scale,
                "normal_shear": self.normal_shear,
                "shift_scale": self.shift_scale,
                "ndims": self.ndims,
                "concat": self.concat,
                "out_type": self.out_type,
                "seeds": self.seeds,
            }
        )
        return config

    def call(self, x):
        """
        Parameters:
            x: Input tensor that we derive the batch dimension from.
        """
        return utility.draw_affine_params(
            shift=self.shift,
            rot=self.rot,
            scale=self.scale,
            shear=self.shear,
            normal_shift=self.normal_shift,
            normal_rot=self.normal_rot,
            normal_scale=self.normal_scale,
            normal_shear=self.normal_shear,
            shift_scale=self.shift_scale,
            ndims=self.ndims,
            batch_shape=tf.shape(x)[:1],
            concat=self.concat,
            dtype=self.out_type,
            seeds=self.seeds,
        )


###############################################################################
# original CIDS layers
###############################################################################


class StateWrapper(tf.keras.layers.Layer):
    def __init__(self, wrapped, *args, **kwargs):
        assert isinstance(wrapped, tf.keras.layers.RNN)
        self.wrapped = wrapped
        self.state_fetch0 = None
        self.state_fetch1 = None
        self.state_placeholder0 = None
        self.state_placeholder1 = None
        super().__init__(*args, **kwargs)

    # TODO activate again
    # @property
    # def units(self):
    #     return self.wrapped.units

    def compute_output_shape(self, input_shape):
        return self.wrapped.compute_output_shape(self, input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        self.state_placeholder0 = tf.compat.v1.placeholder_with_default(
            tf.zeros([1, self.wrapped.units], dtype=self.dtype),
            [None, self.wrapped.units],
            name="state_placeholder0",
        )
        self.state_placeholder1 = tf.compat.v1.placeholder_with_default(
            tf.zeros([1, self.wrapped.units], dtype=self.dtype),
            [None, self.wrapped.units],
            name="state_placeholder1",
        )
        if not self.wrapped.built:
            self.wrapped.build(input_shape)
        shape = self.wrapped.compute_output_shape(input_shape)
        super().build(shape)
        self.built = True

    def call(self, x, *args, **kwargs):
        results = self.wrapped.call(
            x,
            *args,
            initial_state=[
                tf.tile(
                    self.state_placeholder0,
                    [tf.shape(x)[0] // tf.shape(self.state_placeholder0)[0], 1],
                ),
                tf.tile(
                    self.state_placeholder1,
                    [tf.shape(x)[0] // tf.shape(self.state_placeholder1)[0], 1],
                ),
            ],
            **kwargs,
        )
        y_, state0, state1 = results[0], results[1], results[2]
        self.state_fetch0 = state0
        self.state_fetch1 = state1
        return y_


class UNet(tf.keras.layers.Layer):
    def __init__(
        self,
        num_kernels,
        kernel_size=3,
        num_levels=3,
        conv_per_level=3,
        bypass_mode="residual",
        data_format="NXYF",
        keep_prob=1.0,
        normalize=False,
        pool_size=2,
        pool_type="max",
        feature_base=2,
        activation="relu",
        floor_size=None,
        skip_down=False,
        skip_up=False,
        **kwargs,
    ):
        """A U-Net building block."""
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.conv_per_level = conv_per_level
        self.bypass_mode = bypass_mode
        self.data_format = data_format
        self.keep_prob = keep_prob
        self.normalize = normalize
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.skip_down = skip_down
        self.skip_up = skip_up
        self.floor_size = floor_size or int(num_kernels * (feature_base**num_levels))
        self.feature_base = feature_base
        self.num_spatial_dims = (
            data_format.count("X") + data_format.count("Y") + data_format.count("Z")
        )
        self.activation = activation
        # Prepare core
        self.down_levels = []
        self.floor_layers = []
        self.up_levels = []
        self.bypasses = []

        # Down: level 0 to num_levels - 1
        for il in range(self.num_levels):
            level = []

            # Pooling
            if il > 0:
                # Reduce spatial size by 2
                level.append(self.add_pool(self.pool_size, name=f"Udown{il:d}-pool"))
            # Convolutions
            for ic in range(self.conv_per_level):
                level.append(
                    self.add_conv(
                        int(self.num_kernels * (self.feature_base**il)),
                        self.kernel_size,
                        padding="same",
                        activation="linear",
                        use_bias=not self.normalize,
                        name=f"Udown{il:d}-conv{ic:d}",
                    )
                )
                if self.normalize:
                    level.append(
                        self.add_batch_norm(
                            # fused=True,
                            name=f"Udown{il:d}-bn{ic:d}"
                        )
                    )
                # TODO: batch norm before or after relu? Single batch norm per block
                level.append(self.add_activation())

            # Bypass out
            if il < self.num_levels - 1:
                level.append("bypass-out")

            # Append level
            self.down_levels.append(level)

        # Floor: no spatiality
        self.floor_layers.extend(
            self.add_floor(
                self.floor_size,
                activation=self.activation,
                use_bias=not self.normalize,
                name="Ufloor",
            )
        )

        # Up: level num_levels - 1 to 0
        for il in reversed(range(self.num_levels - 1)):

            level = []

            # Unpool
            level.append(
                self.add_upconv(
                    int(self.num_kernels * (self.feature_base**il)),
                    self.pool_size,
                    strides=self.pool_size,
                    name=f"Uup{il:d}-upconv",
                )
            )

            # Bypass in
            level.append("bypass-in")

            # Convolutions
            for ic in range(self.conv_per_level):
                level.append(
                    self.add_conv(
                        int(self.num_kernels * (self.feature_base**il)),
                        self.kernel_size,
                        padding="same",
                        activation="linear",
                        use_bias=not self.normalize,
                        name=f"Uup{il:d}-conv{ic:d}",
                    )
                )
                if self.normalize:
                    level.append(
                        self.add_batch_norm(
                            name=f"Uup{il:d}-bn{ic:d}",
                            # fused=True
                        )
                    )
                # TODO: batch norm before or after relu? Single batch norm per block
                level.append(self.add_activation())

            # Append level
            self.up_levels.append(level)

        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_kernels": self.num_kernels,
                "kernel_size": self.kernel_size,
                "num_levels": self.num_levels,
                "conv_per_level": self.conv_per_level,
                "bypass_mode": self.bypass_mode,
                "data_format": self.data_format,
                "keep_prob": self.keep_prob,
                "normalize": self.normalize,
            }
        )
        return config

    @property
    def units(self):
        return self.num_kernels

    @property
    def layers(self):
        layers = [
            layer
            for level in self.down_levels
            for layer in level
            if isinstance(layer, tf.keras.layers.Layer)
        ]
        layers += [
            layer
            for layer in self.floor_layers
            if isinstance(layer, tf.keras.layers.Layer)
        ]
        layers += [
            layer
            for level in self.up_levels
            for layer in level
            if isinstance(layer, tf.keras.layers.Layer)
        ]
        return layers

    def add_conv(self, *args, **kwargs):
        if self.num_spatial_dims == 2:
            return tf.keras.layers.Conv2D(*args, **kwargs)
        if self.num_spatial_dims == 3:
            return tf.keras.layers.Conv3D(*args, **kwargs)
        raise ValueError(
            f"Invalid number of spatial dimensions: {self.num_spatial_dims:d}"
        )

    def add_pool(self, *args, **kwargs):
        if self.num_spatial_dims == 2:
            if self.pool_type == "max":
                return tf.keras.layers.MaxPooling2D(*args, **kwargs)
            if self.pool_type == "average":
                return tf.keras.layers.AveragePooling2D(*args, **kwargs)
            if self.pool_type == "sum":
                raise NotImplementedError("Arnd: Implement this?")
            if "conv" in self.pool_type:
                kkwargs = copy.deepcopy(kwargs)
                kkwargs["strides"] = args[0]
                kkwargs["padding"] = "valid"
                if "-" in self.pool_type:
                    kkwargs["activation"] = self.pool_type.split("-")[-1]
                else:
                    kkwargs["activation"] = None
                return tf.keras.layers.Conv2D(self.num_kernels, *args, **kkwargs)
            raise ValueError(f"Invalid self.pool_type: {self.pool_type}")
        if self.num_spatial_dims == 3:
            if self.pool_type == "max":
                return tf.keras.layers.MaxPooling3D(*args, **kwargs)
            if self.pool_type == "average":
                return tf.keras.layers.AveragePooling3D(*args, **kwargs)
            if self.pool_type == "sum":
                raise NotImplementedError("Arnd: Implement this?")
            if "conv" in self.pool_type:
                kkwargs = copy.deepcopy(kwargs)
                kkwargs["strides"] = args[0]
                kkwargs["padding"] = "valid"
                if "-" in self.pool_type:
                    kkwargs["activation"] = self.pool_type.split("-")[-1]
                else:
                    kkwargs["activation"] = None
                return tf.keras.layers.Conv3D(self.num_kernels, *args, **kkwargs)
            raise ValueError(f"Invalid self.pool_type: {self.pool_type}")
        raise ValueError(
            f"Invalid number of spatial dimensions: {self.num_spatial_dims:d}"
        )

    def add_upconv(self, *args, **kwargs):
        if self.num_spatial_dims == 2:
            return tf.keras.layers.Conv2DTranspose(*args, **kwargs)
        if self.num_spatial_dims == 3:
            return tf.keras.layers.Conv3DTranspose(*args, **kwargs)
        raise ValueError(
            f"Invalid number of spatial dimensions: {self.num_spatial_dims:d}"
        )

    def add_floor(self, num_units, activation="relu", **kwargs):
        floor = [
            self.add_conv(num_units, 1, padding="same", activation="linear", **kwargs)
        ]
        if self.normalize:
            floor.append(
                self.add_batch_norm(
                    # fused=True,
                    name="Ufloor-bn"
                )
            )  # TODO: batch norm before or after relu? Single batch norm per block
        floor.append(tf.keras.layers.Activation(activation))
        if self.keep_prob < 1.0:
            floor.append(self.add_dropout(rate=1.0 - self.keep_prob))
        return floor

    def add_batch_norm(self, *args, **kwargs):
        if self.normalize == "batch":
            return tf.keras.layers.BatchNormalization(*args, **kwargs)
        if self.normalize == "layer":
            if "fused" in kwargs:
                del kwargs["fused"]
            return LayerNormalization(*args, **kwargs)
        raise ValueError("Invalid normalize value: ", self.normalize)

    def add_dropout(self, *args, **kwargs):
        return tf.keras.layers.Dropout(*args, **kwargs)

    def add_activation(self, *args, **kwargs):
        return tf.keras.layers.Activation(self.activation)

    def _bypass_merge(self, y_, bypass):
        # Pad to spatial size of downward branch
        paddings = []
        for di, dd in enumerate(self.data_format):
            if dd in "XYZ":
                # spatial axes XYZ
                if bypass.shape[di] - y_.shape[di] == 1:
                    paddings += [[0, 1]]
                elif bypass.shape[di] - y_.shape[di] == 0:
                    paddings += [[0, 0]]
                else:
                    raise ValueError("U-net bypass shape not recovered.")
            else:
                # non spatial axes NSF
                paddings += [[0, 0]]
        y_ = tf.pad(y_, paddings, "SYMMETRIC")
        # Merge with bypass
        if self.bypass_mode == "residual":
            y_ = tf.add(y_, bypass)
        elif self.bypass_mode == "concat":
            y_ = tf.concat([y_, bypass], -1)
        return y_

    def _bypass_merge_shape(self, y_shape, bypass_shape):
        if self.bypass_mode == "residual":
            return bypass_shape
        new_shape = list(y_shape)
        new_shape[-1] += bypass_shape[-1]
        return new_shape

    def _merge_skip_down(self, y_, skip):
        y_ = tf.keras.layers.Flatten()(y_)
        skip = tf.keras.layers.Flatten()(skip)
        y_ = tf.concat([y_, skip], -1)
        return y_

    def _merge_skip_down_shape(self, y_shape, skip_shape):
        new_shape = [y_shape[0], np.prod(y_shape[1:])]
        new_shape[-1] += np.prod(skip_shape[1:])
        return new_shape

    def _merge_skip_up(self, y_, skip):
        skip_shape = skip.shape.as_list()
        y_shape = y_.shape.as_list()
        target_shape = list(y_shape[2:])
        target_shape[-1] = int(np.prod(skip_shape[2:])) // int(np.prod(y_shape[2:-1]))
        skip = MaskedTimeDistributed(tf.keras.layers.Reshape(target_shape))(skip)
        y_ = tf.concat([y_, skip], -1)
        return y_

    def _merge_skip_up_shape(self, y_shape, skip_shape):
        new_shape = list(y_shape)
        new_shape[-1] += int(np.prod(skip_shape[2:])) // int(np.prod(y_shape[2:-1]))
        return new_shape

    def call(self, inputs, training=None, mask=None):
        y_ = inputs
        for level in self.down_levels:
            for layer in level:
                if layer == "bypass-out":
                    self.bypasses.append(y_)
                else:
                    kwargs = {}
                    if generic_utils.has_arg(layer.call, "training"):
                        kwargs["training"] = training
                    if generic_utils.has_arg(layer.call, "mask"):
                        kwargs["mask"] = mask
                    y_ = layer.call(y_, **kwargs)
        if self.skip_down:
            y_ = self._merge_skip_down(y_, inputs)
        for layer in self.floor_layers:
            kwargs = {}
            if generic_utils.has_arg(layer.call, "training"):
                kwargs["training"] = training
            if generic_utils.has_arg(layer.call, "mask"):
                kwargs["mask"] = mask
            if isinstance(layer, tf.keras.layers.RNN):
                # TODO: only supports one stateful layer
                results = layer.call(
                    y_,
                    initial_state=[
                        tf.tile(
                            self.state_placeholder0,
                            [
                                tf.shape(y_)[0] // tf.shape(self.state_placeholder0)[0],
                                1,
                            ],
                        ),
                        tf.tile(
                            self.state_placeholder1,
                            [
                                tf.shape(y_)[0] // tf.shape(self.state_placeholder1)[0],
                                1,
                            ],
                        ),
                    ],
                    **kwargs,
                )
                y_, state0, state1 = results[0], results[1], results[2]
                self.state_fetch0 = state0
                self.state_fetch1 = state1
            else:
                y_ = layer.call(y_, **kwargs)
        floor = y_
        for level, bypass in zip(self.up_levels, reversed(self.bypasses)):
            for layer in level:
                if layer == "bypass-in":
                    y_ = self._bypass_merge(y_, bypass)
                else:
                    kwargs = {}
                    if generic_utils.has_arg(layer.call, "training"):
                        kwargs["training"] = training
                    if generic_utils.has_arg(layer.call, "mask"):
                        kwargs["mask"] = mask
                    y_ = layer.call(y_, **kwargs)
        if self.skip_up:
            y_ = self._merge_skip_up(y_, floor)
        return y_

    def compute_output_shape(self, input_shape):
        shape = input_shape
        bypass_shapes = []
        for level in self.down_levels:
            for layer in level:
                if layer == "bypass-out":
                    bypass_shapes.append(shape)
                else:
                    shape = layer.compute_output_shape(shape)
        if self.skip_down:
            shape = self._merge_skip_down_shape(shape, input_shape)
        for layer in self.floor_layers:
            shape = layer.compute_output_shape(shape)
            if isinstance(layer, tf.keras.layers.RNN):
                shape = shape[0]
        floor_shape = shape
        for level, bypass_shape in zip(self.up_levels, reversed(bypass_shapes)):
            for layer in level:
                if layer == "bypass-in":
                    shape = self._bypass_merge_shape(shape, bypass_shape)
                else:
                    shape = layer.compute_output_shape(shape)
        if self.skip_up:
            shape = self._merge_skip_up_shape(shape, floor_shape)
        return shape

    def build(self, input_shape):
        shape = input_shape
        bypass_shapes = []
        for level in self.down_levels:
            for layer in level:
                if layer == "bypass-out":
                    bypass_shapes.append(shape)
                else:
                    if not layer.built:
                        with tf.name_scope(layer.name):
                            layer.build(shape)
                    shape = layer.compute_output_shape(shape)
        if self.skip_down:
            shape = self._merge_skip_down_shape(shape, input_shape)
        for layer in self.floor_layers:
            if isinstance(layer, tf.keras.layers.RNN):
                # TODO: only supports one stateful layer
                self.state_placeholder0 = tf.compat.v1.placeholder_with_default(
                    tf.zeros([1, layer.units], dtype=self.dtype),
                    [None, layer.units],
                    name="state_placeholder0",
                )
                self.state_placeholder1 = tf.compat.v1.placeholder_with_default(
                    tf.zeros([1, layer.units], dtype=self.dtype),
                    [None, layer.units],
                    name="state_placeholder1",
                )
            if not layer.built:
                with tf.name_scope(layer.name):
                    layer.build(shape)
            shape = layer.compute_output_shape(shape)
            if isinstance(layer, tf.keras.layers.RNN):
                shape = shape[0]
        floor_shape = shape
        for level, bypass_shape in zip(self.up_levels, reversed(bypass_shapes)):
            for layer in level:
                if layer == "bypass-in":
                    shape = self._bypass_merge_shape(shape, bypass_shape)
                else:
                    if not layer.built:
                        with tf.name_scope(layer.name):
                            layer.build(shape)
                    shape = layer.compute_output_shape(shape)
        if self.skip_up:
            shape = self._merge_skip_up_shape(shape, floor_shape)
        super().build(shape)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        return mask


class TimeUNet(UNet):
    def __init__(
        self,
        sequence_length,
        num_kernels,
        kernel_size=3,
        num_levels=3,
        conv_per_level=3,
        bypass_mode="residual",
        data_format="NSXYF",
        activation="relu",
        keep_prob=1.0,
        normalize=False,
        time_mode="distribute",
        **kwargs,
    ):
        """A Recurrent U-Net building block."""
        assert time_mode in ["distribute", "recurrent"]
        self.time_mode = time_mode
        self.sequence_length = sequence_length
        self.state_fetch0 = None
        self.state_fetch1 = None
        self.state_placeholder0 = None
        self.state_placeholder1 = None
        super().__init__(
            num_kernels,
            kernel_size=kernel_size,
            num_levels=num_levels,
            conv_per_level=conv_per_level,
            bypass_mode=bypass_mode,
            data_format=data_format,
            keep_prob=keep_prob,
            activation=activation,
            normalize=normalize,
            **kwargs,
        )

    def add_conv(self, *args, **kwargs):
        if self.num_spatial_dims == 2:
            if self.time_mode == "recurrent":
                return tf.keras.layers.ConvLSTM2D(
                    *args, **kwargs, return_sequences=True
                )
            if self.time_mode == "distribute":
                return tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(*args, **kwargs)
                )
            return tf.keras.layers.Conv2D(*args, **kwargs)
        if self.num_spatial_dims == 3:
            return tf.keras.layers.Conv3D(*args, **kwargs)
        raise ValueError(
            f"Invalid number of spatial dimensions: {self.num_spatial_dims:d}"
        )

    def add_pool(self, *args, **kwargs):
        return tf.keras.layers.TimeDistributed(super().add_pool(*args, **kwargs))

    def add_upconv(self, *args, **kwargs):
        return tf.keras.layers.TimeDistributed(super().add_upconv(*args, **kwargs))

    def add_batch_norm(self, *args, **kwargs):
        return tf.keras.layers.TimeDistributed(super().add_batch_norm(*args, **kwargs))

    def _merge_skip_down(self, y_, skip):
        y_ = MaskedTimeDistributed(tf.keras.layers.Flatten())(y_)
        skip = MaskedTimeDistributed(tf.keras.layers.Flatten())(skip)
        y_ = tf.concat([y_, skip], -1)
        return y_

    def _merge_skip_down_shape(self, y_shape, skip_shape):
        new_shape = [y_shape[0], y_shape[1], np.prod(y_shape[2:])]
        new_shape[-1] += np.prod(skip_shape[2:])
        return new_shape

    def add_floor(self, num_units, **kwargs):
        floor = [
            tf.keras.layers.Reshape([self.sequence_length, -1]),
            tf.keras.layers.LSTM(num_units, return_sequences=True, return_state=True),
            tf.keras.layers.Reshape(
                [self.sequence_length] + [1] * self.num_spatial_dims + [num_units]
            ),
        ]
        if self.normalize:
            floor.append(
                self.add_batch_norm(
                    # fused=True,
                    name="Ufloor-bn"
                )
            )  # TODO: batch norm before or after relu? Single batch norm per block
        if self.keep_prob < 1.0:
            floor.append(self.add_dropout(rate=1.0 - self.keep_prob))
        return floor

    def add_dropout(self, *args, **kwargs):
        return tf.keras.layers.TimeDistributed(super().add_dropout(*args, **kwargs))


class LayerNormalization(tf.keras.layers.Layer):
    """Layer normalization layer (Ba et al., 2016).
    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.
    Arguments:
      axis: Integer or List/Tuple. The axis that should be normalized
        (typically the features axis).
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor.
          If False, `beta` is ignored.
      scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.
      trainable: Boolean, if `True` the variables will be marked as trainable.
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as input.
    References:
      - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, trainable=trainable, **kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise ValueError(
                "Expected an int or a list/tuple of ints for the argument 'axis',"
                + f" but received instead: {axis:s}"
            )

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError(f"Input shape {input_shape:s} has undefined rank.")

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError(f"Invalid axis: {x:d}")
        if len(self.axis) != len(set(self.axis)):
            raise ValueError(f"Duplicate axis: {tuple(self.axis)}")

        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )
        else:
            self.beta = None

    def call(self, inputs):
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)

        # Calculate the moments on the last axis (layer activations).
        mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

        # Broadcasting only necessary for norm where the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and self.axis != [ndims - 1]:
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        # Compute layer normalization using the batch_normalization function.
        outputs = nn.batch_normalization(
            inputs,
            mean,
            variance,
            offset=offset,
            scale=scale,
            variance_epsilon=self.epsilon,
        )

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedTimeDistributed(tf.keras.layers.TimeDistributed):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supports_masking = True
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        # output_shape = self.compute_output_shape(inputs.shape)
        # while len(output_shape) < len(mask.shape):
        mask = tf.expand_dims(mask, axis=-1)
        return mask

    def call(self, inputs, mask=None):
        mask = self.compute_mask(inputs, mask=mask)
        outputs = super().call(inputs, mask=mask)
        if mask is None:
            return outputs
        outputs = outputs * tf.cast(mask, outputs.dtype)
        outputs._keras_mask = tf.squeeze(mask, axis=-1)
        return outputs


class Sampling(tf.keras.layers.Layer):
    def __init__(
        self,
        add_sampling_loss=False,
        sampling_model=None,
        mode="repar",
        latent_dim=None,
        beta=1.0,
        analytic_kl=True,
        use_input_as_seed=False,
        **kwargs,
    ):
        """Sample from normal distribution of given mean and logvar tensors."""
        super().__init__(**kwargs)
        self.add_sampling_loss = add_sampling_loss
        assert sampling_model is None or isinstance(sampling_model, tf.keras.Model)
        self.sampling_model = sampling_model
        assert mode in ["repar", "concat", "reparaugment", "drop"]
        self.mode = mode
        if mode == "concat":
            assert (
                latent_dim is not None
            ), "Need to provide latent_dim to concat noise to input tensor."
        self.latent_dim = latent_dim
        self.use_input_as_seed = use_input_as_seed
        self.weight = beta
        self.analytic_kl = analytic_kl

    def get_config(self):
        if self.sampling_model is None:
            sampling_model_config = None
        else:
            sampling_model_config = self.sampling_model.get_config()
        config = {
            "add_sampling_loss": self.add_sampling_loss,
            "sampling_model": sampling_model_config,
            "mode": self.mode,
            "latent_dim": self.latent_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.mode in "repar":
            output_shape[-1] = int(output_shape[-1] // 2)
        elif self.mode == "reparaugment":
            output_shape[-1] = int(output_shape[-1] // 2 + input_shape[-1])
        elif self.mode == "concat":
            output_shape[-1] = output_shape[-1] + self.latent_dim
        return output_shape

    def _log_normal_pdf(self, sample, mean, logvar, axis=-1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=axis,
        )

    def _logpz(self, z):
        return self._log_normal_pdf(z, 0.0, 0.0)

    def _logqz_x(self, z, mean, logvar):
        return self._log_normal_pdf(z, mean, logvar)

    def build(self, input_shape):
        self.beta = tf.Variable(
            self.weight, name="beta", dtype="float32", trainable=False
        )

    def call(self, inputs, eps=None):
        x = inputs
        if self.mode in ["repar", "reparaugment"]:
            if self.sampling_model is not None:
                mean = self.sampling_model(x)
            else:
                mean = x
            # Get mean and variance
            mean, logvar = tf.split(mean, 2, -1)
            # Generate noise
            if eps is None:
                shape = tf.shape(mean)
                if self.use_input_as_seed:
                    seed = tf.stack(
                        [tf.reduce_sum(x) * 10000.0, tf.reduce_max(tf.abs(x)) * 1000.0]
                    )
                    seed = tf.cast(seed, tf.int32)
                    eps = tf.random.stateless_normal(shape=shape, seed=seed)
                else:
                    eps = tf.random.normal(shape=shape)
            # Apply noise
            z = eps * tf.exp(0.5 * logvar) + mean
            if self.add_sampling_loss:
                if self.analytic_kl:
                    kl = tf.reduce_mean(
                        -0.5
                        * tf.reduce_sum(
                            1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1
                        )
                    )
                    kl_loss = tf.math.multiply(self.beta, kl)

                else:
                    kl = -tf.reduce_mean(
                        self._logpz(z) - self._logqz_x(z, mean, logvar)
                    )
                    kl_loss = kl_loss = tf.math.multiply(self.beta, kl)
                self.add_loss(kl_loss)
                # TODO(deepa): is this correct? this seems quite similar to part of the
                # original implementation, but with some missing terms.
                # TODO: check and derive this
            if "augment" in self.mode:
                z = tf.concat([x, z], -1)
        elif self.mode in ["concat", "drop"]:
            # Generate noise
            if eps is None:
                shape = tf.concat([tf.shape(x)[:-1], [self.latent_dim]], -1)
                if self.use_input_as_seed:
                    seed = tf.stack(
                        [tf.reduce_sum(x) * 10000.0, tf.reduce_max(tf.abs(x)) * 1000.0]
                    )
                    seed = tf.cast(seed, tf.int32)
                    eps = tf.random.stateless_normal(shape=shape, seed=seed)
                else:
                    eps = tf.random.normal(shape=shape)
            # Apply noise
            if self.mode == "concat":
                z = tf.concat([x, eps], -1)
            elif self.mode == "drop":
                # Apply noise
                z = eps
            else:
                raise RuntimeError("This should not happen.")
        else:
            raise NotImplementedError
        return z


class NonlinearRegression(tf.keras.layers.Layer):
    def __init__(
        self,
        bias_initializer=None,
        frequency_initializer=None,
        factor_initializer=None,
        **kwargs,
    ):
        """Layer that implements nonlinear regression."""
        super().__init__(**kwargs)
        if bias_initializer is None:
            self.bias_initializer = tf.keras.initializers.Zeros()
        else:
            self.bias_initializer = bias_initializer
        if frequency_initializer is None:
            self.frequency_initializer = tf.keras.initializers.RandomNormal()
        else:
            self.frequency_initializer = frequency_initializer
        if factor_initializer is None:
            self.factor_initializer = tf.keras.initializers.RandomNormal()
        else:
            self.factor_initializer = factor_initializer
        self.factor = None
        self.frequency = None
        self.bias = None

    def get_config(self):
        config = super().get_config()
        config["bias_initializer"] = self.bias_initializer
        config["frequency_initializer"] = self.frequency_initializer
        config["factor_initializer"] = self.factor_initializer
        return config

    def build(self, input_shape):
        self.factor = self.add_weight(
            "factor", (1,), initializer=self.factor_initializer, trainable=True
        )
        self.frequency = self.add_weight(
            "frequency",
            (input_shape[-1], 1),
            initializer=self.frequency_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias", (1,), initializer=self.bias_initializer, trainable=True
        )

    def call(self, inputs):
        return self.factor * tf.exp(inputs @ self.frequency) - self.bias
