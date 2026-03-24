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
"""
Tensorflow/Keras model for CIDS. Part of the CIDS toolbox.
"""
"""
tensorflow/keras networks for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""
# internal python imports
import os
import warnings
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.initializers as KI
import tensorflow.keras.layers as KL
from tensorflow.python.keras.engine import data_adapter

from . import layers
from . import modelio
from . import utility
import neurite as ne

# NOTE: Do not assign core model components to variables inside the compiled
# @tf.function train and test steps! The assignment will be hard coded at compile time
# and the metrics will not update properly.


# make directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel


class VxmDense(modelio.LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @modelio.store_config_args
    def __init__(
        self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        nb_unet_conv_per_level=1,
        int_steps=7,
        svf_resolution=1,
        int_resolution=2,
        int_downsize=None,
        bidir=False,
        use_probs=False,
        src_feats=1,
        trg_feats=1,
        unet_half_res=False,
        input_model=None,
        hyp_model=None,
        fill_value=None,
        reg_field="preintegrated",
        name="vxm_dense",
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            svf_resolution: Resolution (relative voxel size) of the predicted SVF (Stationary Vector Fields).
                Default is 1.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            int_downsize: Deprecated - use int_resolution instead.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Deprecated - use svf_resolution instead.
            input_model: Model to replace default input layer before concatenation. Default is None.
            hyp_model: HyperMorph hypernetwork model. Default is None.
            reg_field: Field to regularize in the loss. Options are 'svf' to return the
                SVF predicted by the Unet, 'preintegrated' to return the SVF that's been
                rescaled for vector-integration (default), 'postintegrated' to return the
                rescaled vector-integrated field, and 'warp' to return the final, full-res warp.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(
                shape=(*inshape, src_feats), name="%s_source_input" % name
            )
            target = tf.keras.Input(
                shape=(*inshape, trg_feats), name="%s_target_input" % name
            )
            input_model = tf.keras.Model(
                inputs=[source, target], outputs=[source, target]
            )
        else:
            source, target = input_model.outputs[:2]

        # configure inputs
        inputs = input_model.inputs
        if hyp_model is not None:
            hyp_input = hyp_model.input
            hyp_tensor = hyp_model.output
            if not any([hyp_input is inp for inp in inputs]):
                inputs = (*inputs, hyp_input)
        else:
            hyp_input = None
            hyp_tensor = None

        if int_downsize is not None:
            warnings.warn(
                "int_downsize is deprecated, use the int_resolution parameter."
            )
            int_resolution = int_downsize

        # compute number of upsampling skips in the decoder (to downsize the predicted field)
        if unet_half_res:
            warnings.warn(
                "unet_half_res is deprecated, use the svf_resolution parameter."
            )
            svf_resolution = 2

        nb_upsample_skips = int(np.floor(np.log(svf_resolution) / np.log(2)))

        # build core unet model and grab inputs
        unet_model = layers.VXM_Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            nb_upsample_skips=nb_upsample_skips,
            hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
            name="%s_unet" % name,
        )

        # transform unet output into a flow field
        Conv = getattr(KL, "Conv%dD" % ndims)
        flow_mean = Conv(
            ndims,
            kernel_size=3,
            padding="same",
            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
            name="%s_flow" % name,
        )(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(
                ndims,
                kernel_size=3,
                padding="same",
                kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                bias_initializer=KI.Constant(value=-10),
                name="%s_log_sigma" % name,
            )(unet_model.output)
            flow_params = KL.concatenate(
                [flow_mean, flow_logsigma], name="%s_prob_concat" % name
            )
            flow_inputs = [flow_mean, flow_logsigma]
            flow = layers.SampleNormalLogVar(name="%s_z_sample" % name)(flow_inputs)
        else:
            flow = flow_mean

        # rescale field to target svf resolution
        pre_svf_size = np.array(flow.shape[1:-1])
        svf_size = np.array([np.round(dim / svf_resolution) for dim in inshape])
        if not np.array_equal(pre_svf_size, svf_size):
            rescale_factor = svf_size[0] / pre_svf_size[0]
            flow = layers.RescaleTransform(rescale_factor, name=f"{name}_svf_resize")(
                flow
            )

        # cache svf
        svf = flow

        # rescale field to target integration resolution
        if int_steps > 0 and int_resolution > 1:
            int_size = np.array([np.round(dim / int_resolution) for dim in inshape])
            if not np.array_equal(svf_size, int_size):
                rescale_factor = int_size[0] / svf_size[0]
                flow = layers.RescaleTransform(
                    rescale_factor, name=f"{name}_flow_resize"
                )(flow)

        # cache pre-integrated flow field
        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = layers.Negate(name="%s_neg_flow" % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(
                method="ss", name="%s_flow_int" % name, int_steps=int_steps
            )(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(
                    method="ss", name="%s_neg_flow_int" % name, int_steps=int_steps
                )(neg_flow)

        # cache the intgrated flow field
        postint_flow = pos_flow

        # resize to final resolution
        if int_steps > 0 and int_resolution > 1:
            rescale_factor = inshape[0] / int_size[0]
            pos_flow = layers.RescaleTransform(
                rescale_factor, name="%s_diffflow" % name
            )(pos_flow)
            if bidir:
                neg_flow = layers.RescaleTransform(
                    rescale_factor, name="%s_neg_diffflow" % name
                )(neg_flow)

        # warp image with flow field
        y_source = layers.SpatialTransformer(
            interp_method="linear", fill_value=fill_value, name="%s_transformer" % name
        )([source, pos_flow])

        if bidir:
            st_inputs = [target, neg_flow]
            y_target = layers.SpatialTransformer(
                interp_method="linear",
                fill_value=fill_value,
                name="%s_neg_transformer" % name,
            )(st_inputs)

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        # determine regularization output
        reg_field = reg_field.lower()
        if use_probs:
            # compute loss on flow probabilities
            outputs.append(flow_params)
        elif reg_field == "svf":
            # regularize the immediate, predicted SVF
            outputs.append(svf)
        elif reg_field == "preintegrated":
            # regularize the rescaled, pre-integrated SVF
            outputs.append(preint_flow)
        elif reg_field == "postintegrated":
            # regularize the rescaled, integrated field
            outputs.append(postint_flow)
        elif reg_field == "warp":
            # regularize the final, full-resolution deformation field
            outputs.append(pos_flow)
        else:
            raise ValueError(f'Unknown option "{reg_field}" for reg_field.')

        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = svf
        self.references.preint_flow = preint_flow
        self.references.postint_flow = postint_flow
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method="linear"):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_input = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(interp_method=interp_method)(st_input)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict(
            [src, trg, img]
        )


###############################################################################
# original CIDS custom models
###############################################################################


def get_gan_mode(ordered_keys):
    if not isinstance(ordered_keys, list):
        ordered_keys = ordered_keys.core_model.keys()
    if "conditional" in "".join(ordered_keys):
        return "conditional"
    if "auxiliary" in "".join(ordered_keys):
        return "auxiliary"
    return None


def _shorten_name(name, prefix=""):
    if "Subcore" in name:
        return prefix + name.split("_", maxsplit=1)[1]
    return prefix + name


class MeanBalance(tf.keras.metrics.Mean):
    def __init__(self, name="mean_balance", optimal_balance=0.7, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.optimal_balance = optimal_balance

    def update_state(self, values, sample_weight=None):
        return super().update_state(
            tf.abs(values - self.optimal_balance), sample_weight=sample_weight
        )


class GANControl(tf.keras.callbacks.Callback):
    def __init__(self, gan_model, decay_factor=0.99):
        self.gan_model = gan_model
        self.decay_factor = decay_factor
        super().__init__()

    def _decay(self, variable, factor):
        if variable is not None:
            value = tf.keras.backend.get_value(variable)
            value *= factor
            tf.keras.backend.set_value(variable, value)

    def _flip(self, variable, probability):
        if variable is not None:
            if np.random.binomial(1, probability):
                value = tf.keras.backend.get_value(variable)
                value = not value
                tf.keras.backend.set_value(variable, value)

    def on_epoch_begin(self, epoch, logs=None):
        self._decay(self.gan_model.noisy_labels_var, self.decay_factor)
        self._decay(self.gan_model.noisy_image_var, self.decay_factor)
        # self._decay(self.gan_model.noisy_labels_var, self.decay_factor)
        self._flip(self.gan_model.noisy_labels_var, self.gan_model.noisy_labels)


class GANModel(tf.keras.Model):
    def __init__(
        self,
        generator,
        adversary,
        smooth_labels=0.2,
        noisy_image=0.05,
        noisy_labels=0.001,
        decay_factor=0.99,
        optimal_balance=0.7,
    ):
        """A Generative Adversarial Network (GAN) model for keras

        Args:
            generator (keras.Model): generator model
            discriminator (keras.Model): discriminator model
            smooth_labels (float): noise standard deviation to apply to each label
            noisy_labels (float): probability for flipping the labels each batch
            optimal_balance (float): desired target accuracy for binary discrimination
        """
        super().__init__()
        self.generator = generator
        self.adversary = adversary
        self.smooth_labels = smooth_labels
        self.smooth_labels_var = None
        self.noisy_image = noisy_image
        self.noisy_image_var = None
        self.noisy_labels = noisy_labels
        self.noisy_labels_var = None
        self.decay_factor = decay_factor
        if len(adversary.inputs) > 1:
            self.mode = "conditional"
        elif len(adversary.outputs) > 1:
            self.mode = "auxiliary"
        else:
            self.mode = None
        # Average the distance of the accuracy of the binary discrimination from 0.7
        # over the entire training to find models with stable learning
        self.train_balance_tracker = MeanBalance(
            "mean_balance", optimal_balance=optimal_balance
        )
        self.val_balance_tracker = MeanBalance(
            "mean_balance", optimal_balance=optimal_balance
        )

    def build(self, input_shape):
        if self.smooth_labels:
            self.smooth_labels_var = self.add_weight(
                name="smooth_labels", shape=None, dtype=tf.float32, trainable=False
            )
            tf.keras.backend.set_value(self.smooth_labels_var, self.smooth_labels)
        if self.noisy_image:
            self.noisy_image_var = self.add_weight(
                name="noisy_image", shape=None, dtype=tf.float32, trainable=False
            )
            tf.keras.backend.set_value(self.noisy_image_var, self.noisy_image)
        if self.noisy_labels:
            self.noisy_labels_var = self.add_weight(
                name="noisy_labels", shape=None, dtype=tf.bool, trainable=False
            )
            tf.keras.backend.set_value(self.noisy_labels_var, False)
            # self.noisy_labels_var = self.add_weight(
            #     name="noisy_labels", shape=None, dtype=tf.float32, trainable=False
            # )
            # tf.keras.backend.set_value(self.noisy_labels_var, self.noisy_labels)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return self.generator(inputs, training=training)

    def fit(self, *args, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(GANControl(self, decay_factor=self.decay_factor))
        kwargs["callbacks"] = callbacks
        return super().fit(*args, **kwargs)

    def compile(
        self,
        optimizer_gen,
        optimizer_adv,
        loss_gen=None,
        loss_adv=None,
        metrics_gen=None,
        metrics_adv=None,
    ):
        self.generator.compile(optimizer_gen, loss=loss_gen, metrics=metrics_gen)
        self.adversary.compile(optimizer_adv, loss=loss_adv, metrics=metrics_adv)
        super().compile()

    def train_step(self, data):
        # Get data
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Meta architecture assembly
        with tf.GradientTape() as forward_tape, tf.GradientTape() as adversarial_tape:
            # Forward model prediction
            y_ = self.generator(x, training=True)
            # Apply noise to auxiliary labels
            if self.noisy_image:
                y = y + tf.random.normal(tf.shape(y), stddev=self.noisy_image_var)
                y_ = y_ + tf.random.normal(tf.shape(y), stddev=self.noisy_image_var)
            # Adversarial model prediction fake and real input
            if self.mode == "conditional":
                e_ = self.adversary((x, y_), training=True)
                e = self.adversary((x, y), training=True)
            elif self.mode == "auxiliary":
                e_, x_y_ = self.adversary(y_, training=True)
                e, x_y = self.adversary(y, training=True)
            else:
                e_ = self.adversary(y_, training=True)
                e = self.adversary(y, training=True)
            # Compute Masks
            if sample_weight is None:
                mask_e = None
                mask_x_y = None
                double_mask_e = None
            else:
                mask_e = tf.concat([sample_weight] * e_.shape[-1], -1)
                mask_x_y = tf.concat([sample_weight] * x_y_.shape[-1], -1)
                double_mask_e = tf.concat([mask_e] * 2, 0)
            # Losses
            e_true = tf.ones_like(e_)
            e_fake = tf.zeros_like(e_)
            # Apply noise to binary labels
            if self.smooth_labels:
                e_true = e_true - self.smooth_labels_var * tf.random.uniform(
                    tf.shape(e_true)
                )
                e_fake = e_fake + self.smooth_labels_var * tf.random.uniform(
                    tf.shape(e_fake)
                )
            # Flip labels
            if self.noisy_labels:
                # flip = tf.random.uniform([], minval=0.0, maxval=1.0)
                # e_true = tf.cond(
                #     flip > self.noisy_labels_var,
                #     lambda: tf.identity(e_true),
                #     lambda: tf.identity(e_fake),
                # )
                # e_fake = tf.cond(
                #     flip > self.noisy_labels_var,
                #     lambda: tf.identity(e_fake),
                #     lambda: tf.identity(e_true),
                # )
                # self.noisy_labels_var.assign(flip < self.noisy_labels)
                e_true = tf.cond(
                    self.noisy_labels_var,
                    lambda: tf.identity(e_fake),
                    lambda: tf.identity(e_true),
                )
                e_fake = tf.cond(
                    self.noisy_labels_var,
                    lambda: tf.identity(e_true),
                    lambda: tf.identity(e_fake),
                )
            # Forward losses
            if self.mode == "auxiliary":
                l_forward = self.adversary.compiled_loss(
                    (e_true, x),
                    (e_, x_y_),
                    (mask_e, mask_x_y),
                    regularization_losses=self.generator.losses,
                )
            else:
                l_forward = self.adversary.compiled_loss(
                    e_true,
                    e_,
                    mask_e,
                    regularization_losses=self.generator.losses,
                )
            # Adversarial losses
            if self.mode == "auxiliary":
                # # Apply noise to auxiliary labels
                # if self.noisy_labels:
                #     x = x + tf.random.normal(tf.shape(x))
                l_adversarial1 = self.adversary.compiled_loss(
                    (e_fake, x),
                    (e_, x_y_),
                    (mask_e, mask_x_y),
                    regularization_losses=self.adversary.losses,
                )
                l_adversarial2 = self.adversary.compiled_loss(
                    (e_true, x),
                    (e, x_y),
                    (mask_e, mask_x_y),
                    regularization_losses=self.adversary.losses,
                )
            else:
                l_adversarial1 = self.adversary.compiled_loss(
                    e_fake,
                    e_,
                    mask_e,
                    regularization_losses=self.adversary.losses,
                )
                l_adversarial2 = self.adversary.compiled_loss(
                    e_true,
                    e,
                    mask_e,
                    regularization_losses=self.adversary.losses,
                )
            # Combine losses
            l_adversarial = l_adversarial1 + l_adversarial2
        # Compute gradients
        # clipnorms working????
        gradients_forward = forward_tape.gradient(
            l_forward, self.generator.trainable_variables
        )
        gradients_adversarial = adversarial_tape.gradient(
            l_adversarial, self.adversary.trainable_variables
        )
        # Apply gradients
        self.generator.optimizer.apply_gradients(
            zip(gradients_forward, self.generator.trainable_variables)
        )
        self.adversary.optimizer.apply_gradients(
            zip(
                gradients_adversarial,
                self.adversary.trainable_variables,
            )
        )
        # Update metrics
        self.generator.compiled_metrics.update_state(y, y_, sample_weight)
        if self.mode == "auxiliary":
            # Auxiliary metrics for test only measure predictions from true inputs with
            # true outputs
            self.adversary.compiled_metrics.update_state(
                (tf.concat([tf.ones_like(e), tf.zeros_like(e_)], 0), x),
                (tf.concat([e, e_], 0), x_y),
                (double_mask_e, mask_x_y),
            )
        else:
            self.adversary.compiled_metrics.update_state(
                tf.concat([e_true, e_fake], 0), tf.concat([e, e_], 0), double_mask_e
            )
        # Merge metrics
        metrics = {_shorten_name(m.name): m.result() for m in self.generator.metrics}
        metrics.update(
            {
                _shorten_name(m.name, prefix="adv_"): m.result()
                for m in self.adversary.metrics
            }
        )
        # Compute balance metric
        adv_accuracy = metrics.get("adv_accuracy")
        if adv_accuracy is not None:
            self.train_balance_tracker.update_state(  # pylint: disable=not-callable
                adv_accuracy
            )
            short_name = _shorten_name(self.train_balance_tracker.name)
            metrics[
                short_name
            ] = self.train_balance_tracker.result()  # pylint: disable=not-callable
        # Overwrite loss metrics
        metrics["loss"] = l_forward
        metrics["adv_loss"] = l_adversarial
        return metrics

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Meta architecture assembly
        y_ = self.generator(x, training=False)
        # Adversarial model prediction fake and real input
        if self.mode == "conditional":
            e_ = self.adversary((x, y_), training=False)
            e = self.adversary((x, y), training=False)
        elif self.mode == "auxiliary":
            e_, x_y_ = self.adversary(y_, training=False)
            e, x_y = self.adversary(y, training=False)
        else:
            e_ = self.adversary(y_, training=False)
            e = self.adversary(y, training=False)
        # Compute Masks
        if sample_weight is None:
            mask_e = None
            mask_x_y = None
            double_mask_e = None
        else:
            mask_e = tf.concat([sample_weight] * e_.shape[-1], -1)
            mask_x_y = tf.concat([sample_weight] * x_y_.shape[-1], -1)
            double_mask_e = tf.concat([mask_e] * 2, 0)
        # Losses
        e_true = tf.ones_like(e_)
        e_fake = tf.zeros_like(e_)
        if self.mode == "auxiliary":
            l_forward = self.adversary.compiled_loss(
                (e_true, x),
                (e_, x_y_),
                (mask_e, mask_x_y),
                regularization_losses=self.generator.losses,
            )
            l_adversarial1 = self.adversary.compiled_loss(
                (e_fake, x),
                (e_, x_y_),
                (mask_e, mask_x_y),
                regularization_losses=self.adversary.losses,
            )
            l_adversarial2 = self.adversary.compiled_loss(
                (e_true, x),
                (e, x_y),
                (mask_e, mask_x_y),
                regularization_losses=self.adversary.losses,
            )
        else:
            l_forward = self.adversary.compiled_loss(
                e_true, e_, mask_e, regularization_losses=self.generator.losses
            )
            l_adversarial1 = self.adversary.compiled_loss(
                e_fake, e_, mask_e, regularization_losses=self.adversary.losses
            )
            l_adversarial2 = self.adversary.compiled_loss(
                e_true, e, mask_e, regularization_losses=self.adversary.losses
            )
        # Combine losses
        l_adversarial = l_adversarial1 + l_adversarial2
        # Update metrics
        self.generator.compiled_metrics.update_state(y, y_, sample_weight)
        if self.mode == "auxiliary":
            # Auxiliary metrics for test only measure predictions from true inputs with
            # true outputs
            self.adversary.compiled_metrics.update_state(
                (tf.concat([e_true, e_fake], 0), x),
                (tf.concat([e, e_], 0), x_y),
                (double_mask_e, mask_x_y),
            )
        else:
            self.adversary.compiled_metrics.update_state(
                tf.concat([e_true, e_fake], 0), tf.concat([e, e_], 0), double_mask_e
            )
        # Merge metrics
        metrics = {_shorten_name(m.name): m.result() for m in self.generator.metrics}
        metrics.update(
            {
                _shorten_name(m.name, prefix="adv_"): m.result()
                for m in self.adversary.metrics
            }
        )
        # Compute balance metric
        adv_accuracy = metrics.get("adv_accuracy")
        if adv_accuracy is not None:
            self.val_balance_tracker.update_state(  # pylint: disable=not-callable
                adv_accuracy
            )
            short_name = _shorten_name(self.val_balance_tracker.name)
            metrics[
                short_name
            ] = self.val_balance_tracker.result()  # pylint: disable=not-callable
        # Overwrite loss metrics
        metrics["loss"] = l_forward
        metrics["adv_loss"] = l_adversarial
        return metrics

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        y_ = self.generator(x, training=False)
        if self.mode == "auxiliary":
            return y_, self.adversary(y, training=False)
        return y_

    def reset_metrics(self):
        self.generator.reset_metrics()
        self.adversary.reset_metrics()
        super().reset_metrics()

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        filepath_gen = os.path.join(filepath, "gen")
        filepath_adv = os.path.join(filepath, "adv")
        self.generator.save(
            filepath_gen,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.adversary.save(
            filepath_adv,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
