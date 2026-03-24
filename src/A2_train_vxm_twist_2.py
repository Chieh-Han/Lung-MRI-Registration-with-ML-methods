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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import cids

import numpy as np
import tensorflow as tf
import seaborn as sns
import json

from pathlib import Path
from tensorflow import keras
import keras.backend as kbackend
import keras.layers as klayers
import keras.initializers as kinitializers
from cids.tensorflow import layers as clayers
from cids.tensorflow.tuner import SearchResults
from cids.statistics import metrics
from kadi_ai import KadiAIProject

import cids.tensorflow.losses as cus_losses

from matplotlib import pyplot as plt
from sortedcontainers import SortedDict
from kerastuner import HyperParameters
from datetime import datetime
import inspect

# Preamble
# plt.style.use("seaborn-paper")  # seaborn-talk, seaborn-poster, seaborn-paper
# plt.rcParams.update(
#     {
#         "font.family": "sans-serif",
#         "figure.dpi": 300,
#         "savefig.format": "png",
#     }
# )

# Time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

################################################################################
# Controls

CHECK = True
SEARCH = False
USE_BEST_SEARCH_CONFIG = False
TRAIN = False
EXPORT = False
EVAL = True
PLOT = False
ANALYZE = False

TRAIN_CONTINUE = False
BIDIR = False

num_check_samples = 100
num_plot_samples = 20
num_principal_components = 3


################################################################################
# Data paths

project_name = "vxm_twist_time_128"
project_dir = Path("/mnt/data/stud-uexja/DATA") / project_name
project = KadiAIProject(project_name, root=project_dir)
project.seed=1216808702

# Read paths
train_samples, valid_samples, test_samples = project.get_split_datasets(
    shuffle=False, valid_split=0.15, test_split=0.15, split_mode='subject'
)
#################################################################################
# Data definition

data_definition = project.data_definition
data_definition.input_features = ["moving", "fixed"]
if BIDIR:
    data_definition.output_features = ["fixed", "moving"]
else:
    data_definition.output_features = ["fixed"]
print()

#################################################################################
# Neural network

def model_function(hp,
                   data_definition, 
                   int_steps = 7,
                   bidir = BIDIR,
                   reg_field = 'svf',
                   svf_resolution = 1, 
                   int_resolution = 2,
                   ):
    # int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
    #            value is 0.
    # svf_resolution: Resolution (relative voxel size) of the predicted SVF.
    #                 Default is 1.
    # int_resolution: Resolution (relative voxel size) of the flow field during
    #                 vector integration. Default is 2.
    # bidir: Enable bidirectional cost function.


    bidir = True # bad implementation to add inv registration and flow field when Bidir is not initially used

    vol_shape = data_definition.input_shape[1:]
    inshape = data_definition.input_shape[1:-1]
    ndim= len(inshape)
    assert ndim in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndim


    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ]
    unet = clayers.VXM_Unet(inshape=vol_shape, nb_features=nb_features)

    # transform unet results into a flow field.
    Conv = getattr(klayers, 'Conv%dD' % ndim)
    disp_tensor = Conv(ndim, kernel_size=3, padding='same', kernel_initializer=kinitializers.RandomNormal(mean=0.0, stddev=1e-5))(unet.output)
    
    # configure flow field format
    flow = disp_tensor
    pre_svf_size = np.array(flow.shape[1:-1])
    svf_size = np.array([np.round(dim / svf_resolution) for dim in inshape])

    # rescale flow field to target flow field resolution (svf: Stationary Velocity Fields)
    if not np.array_equal(pre_svf_size, svf_size):
        rescale_factor = svf_size[0] / pre_svf_size[0]
        flow = clayers.RescaleTransform(rescale_factor)(flow)

    # direct predicted flow field
    svf = flow

    # rescale field to target integration resolution
    if int_steps > 0 and int_resolution > 1:
        int_size = np.array([np.round(dim / int_resolution) for dim in inshape])
        if not np.array_equal(svf_size, int_size):
            rescale_factor = int_size[0] / svf_size[0]
            flow = clayers.RescaleTransform(rescale_factor)(flow)

    # rescaled flow field
    preint_flow = flow

    # optional bidirectional flow field
    pos_flow = flow
    if bidir:
        neg_flow = clayers.Negate()(flow)

    # integrate to produce diffeomorphic warp
    if int_steps > 0:
        pos_flow = clayers.VecInt(method='ss', int_steps=int_steps)(pos_flow)
        if bidir:
            neg_flow = clayers.VecInt(method='ss', int_steps=int_steps)(neg_flow)

    # intgrated flow field
    postint_flow = pos_flow

    # resize to final resolution
    if int_steps > 0 and int_resolution > 1:
        rescale_factor = inshape[0] / int_size[0]
        pos_flow = clayers.RescaleTransform(rescale_factor)(pos_flow)
        if bidir:
            neg_flow = clayers.RescaleTransform(rescale_factor)(neg_flow)

    # extract the first frame (i.e. the "moving" image) from unet input tensor
    moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)

    # warp the moving image with the transformer
    moved_image_tensor = clayers.SpatialTransformer(name='transformer')([moving_image, pos_flow])
    if bidir:
        reversed_image_tensor = clayers.SpatialTransformer(name='neg_transformer')([moved_image_tensor, neg_flow])

    # initialize the keras model output
    outputs = [moved_image_tensor, reversed_image_tensor] if bidir else [moved_image_tensor]

    # determine regularization output
    if reg_field == 'svf':
        # regularize the immediate, predicted SVF
        outputs.append(svf)
    elif reg_field == 'preintegrated':
        # regularize the rescaled, pre-integrated SVF
        outputs.append(preint_flow)
    elif reg_field == 'postintegrated':
        # regularize the rescaled, integrated field
        outputs.append(postint_flow)
    elif reg_field == 'warp':
        # regularize the final, full-resolution deformation field
        outputs.append(pos_flow)
    else:
        raise ValueError(f'Unknown option "{reg_field}" for reg_field.')
    
    # if int_steps > 0:
    #     outputs.append(pos_flow)
    # if bidir:
    #     outputs.append(neg_flow)

    vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)
    
    return vxm_model

def extract_function_defaults(func):
    sig = inspect.signature(func)
    params = sig.parameters
    defaults = {k: v.default for k, v in params.items() if v.default is not inspect.Parameter.empty}
    return defaults

#################################################################################
# Training schedule

def schedule_function(hp):
    learning_rate = 1e-4 #hp.Choice("learning_rate", [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5], default = 1e-4)
    batch_size = 8 # hp.Choice("batch_size", [128, 256 ,512], default = 256)
    
    schedule = {
        "count": [1, 1001],
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    return schedule

#################################################################################
# Assign losses

# Define your loss type and parameters
loss_type = 'NCC'  # 'TukeyBiweight' or 'NCC', or any other loss type you define
regul_weight = 1.0
'''
!!! MutualInformation as of now has NO BIDIR implementation
!!! NCC, loss weight should be at order of ones (i.e. 1.0)
!!! MSE, loss weight should be at order of hundredths (i.e. 0.02)
!!! NCC BIDIR weight ratio: [1.0, 0.1, regul_weight]
!!! MSE BIDIR weight ratio: [1.0, 1.0, regul_weight]

*** NCC, When BIDIR is applied, DECREASE the inv loss weight since inv flowfield creates an almost perfect original moving image
    and NCC's value ranges only from (-1,1) the perfect inveresed original moving will out-weigh the target loss causing the 
    network to stop optimizing fwd flowfield
*** MSE, also create inv flowfield with an almost perfect original moving image. HOWEVER, since it ranges from 0 to infinite 
    the target loss value is much bigger than the inverse loss value keeping the network still focusing on the target loss value
'''

loss_instance = cus_losses.get_loss_function(loss_type)

if BIDIR:
    losses = [
        cus_losses.get_loss_function(loss_type, direction="fwd").loss,
        cus_losses.get_loss_function(loss_type, direction="inv").loss,
        cus_losses.get_loss_function('Grad', penalty="l2").loss
    ]
    losses_weights = [1.0, 0.5, regul_weight]
else:
    losses = [
        cus_losses.get_loss_function(loss_type).loss,
        # cus_losses.get_loss_function('Grad', penalty="l2").loss
    ]
    losses_weights = [
        1.0, 
        # regul_weight,
        ]

# Get the name of the current loss function
loss_name = loss_instance.get_loss_name()

#################################################################################
# Assign proper name

model_name = "VXM_Twist"

model_name += "--" + "--".join(
    [
        "+".join(list(data_definition.input_features)),
        "+".join(list(data_definition.output_features)),
    ]
)

# get model parameters
model_defaults = extract_function_defaults(model_function)
keys_list = list(model_defaults.keys())
values_list = list(model_defaults.values())
model_id = ''
for k in range(len(model_defaults)-2): # get first 3 parameters
    model_id += f'--{keys_list[k]}:{values_list[k]}'

# get epoch numbers
schedule_name = schedule_function(None) 
count_value = schedule_name["count"]

# !!! When doing TRAIN_CONTINUE make sure the epoch is set to initial epoch since it looks for the same name
# TODO: or just remove epoch number from naming
model_id = f"--{loss_name}--epoch:{count_value[1]}" + model_id
model_name += model_id

#################################################################################
# Construct Model

model = cids.CIDSModel.vxm_network(
    data_definition,
    model_function,
    name=model_name,
    identifier="",
    result_dir=project.result_dir,
    loss = losses,
    loss_weights=losses_weights,
)
# model.metrics.append("accuracy")
# model.monitor = "val_accuracy"
# model.online_normalize = "input"
model.data_reader.prefetch = "cache"
# model.metrics.append("mse")
model.VERBOSITY = 2


if CHECK:
    if PLOT:
        check_samples = model.read_tfrecords(
            train_samples[:num_check_samples], disable_feature_merge=True
        )
        project.log("Plotting feature distributions.")
        model.plot_feature_distribution(check_samples, project.input_dir)


if SEARCH:
    project.log(">> Hyperparameters: searching...")
    hp = model.search(
        train_samples,
        valid_samples,
        schedule=schedule_function,
        executions_per_trial=1,
        # max_epochs=51,  # hyperband only
        overwrite=False,
        objective="val_mse",
        # method="hyperband",
        method="bayes",
        num_trials=100,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
    )
    if PLOT:
        search_results = SearchResults(model)
        search_results.plot_hyperparameter_search()
    model.identifier = "best"
    project.log(">> Hyperparameters: search complete.")
elif USE_BEST_SEARCH_CONFIG:
    try:
        search_results = SearchResults(model)
        hps = search_results.get_best_hyperparameters(print="best")
        hp = hps[0]
        model.identifier = "best"
        project.log(">> Hyperparameters: loaded from previous search.")
    except (FileNotFoundError, PermissionError) as e:
        project.log(">> Hyperparameters: " + str(e))
        project.log(">> Hyperparameters: No search found. Defaults loaded.")
        hp = HyperParameters()
        model.identifier = "default"
else:
    project.log(">> Hyperparameters: Defaults loaded.")
    hp = HyperParameters()
    model.identifier = "default"


if TRAIN:
    project.log(">> Training...")
    if TRAIN_CONTINUE:
        project.log(">> ...continuing from last...")
        checkpoint = "last"
    else:
        project.log(">> ...starting new...")
        checkpoint = None
    model.train(
        train_samples,
        valid_samples,
        schedule=schedule_function,
        checkpoint=checkpoint,
    )
    project.log(">> Training complete.")
    if EXPORT:
        model.export(
            f"/mnt/data/stud-uexja/Documents/MRI_Registration/CIDS_SavedModels/{project_name}{model_id}/vxm_twist.keras"
        )

if EVAL:
    project.log(">> Evaluating...")
    project.log(">>> Metrics...")
    if BIDIR == True:
        X, Y, Y_reg, Y_neg, Y_fwdfield, Y_posfwd = model.infer_data(
            test_samples,
            batch_size=3,
            checkpoint="last_phase",
            postprocess=False,
        )

        test_result_file = Path(model.base_model_dir, "test_results_inferdata.npz")
        np.savez(
            test_result_file,
            X=X,
            Y=Y,
            Y_reg=Y_reg,
            Y_neg = Y_neg,
            Y_fwdfield=Y_fwdfield,
            Y_posfwd = Y_posfwd,
        )
    else:
        X, Y, Y_reg, Y_neg, Y_posfwd,  = model.infer_data(
            test_samples,
            batch_size=3,
            checkpoint="last_phase",
            postprocess=False,
        )

        test_result_file = Path(model.base_model_dir, "test_results_inferdata.npz")
        np.savez(
            test_result_file,
            X=X,
            Y=Y,
            Y_reg=Y_reg,
            Y_neg=Y_neg,
            Y_posfwd = Y_posfwd,
        )
'''
    # # Evaluate metrics
    # total_metrics = SortedDict()
    # total_metrics["mae"] = metrics.mean_absolute_error(Y, Y_)
    # total_metrics["mape"] = metrics.mean_absolute_percentage_error(Y, Y_)
    # total_metrics["smape"] = metrics.symmetric_mean_absolute_percentage_error(Y, Y_)
    # total_metrics["rmse"] = metrics.root_mean_square_error(Y, Y_)
    # total_metrics["nrmse"] = metrics.normalized_root_mean_square_error(Y, Y_)
    # project.log("Total metrics")
    # for k, v in total_metrics.items():
    #     project.log(f"   {k}: {v}")
    # project.log("")

    # # Feature metrics
    # feature_metrics = SortedDict()
    # feature_metrics["mae"] = metrics.mean_absolute_error(Y, Y_, reduction_axes=(0,))
    # feature_metrics["mape"] = metrics.mean_absolute_percentage_error(
    #     Y, Y_, reduction_axes=(0,)
    # )
    # feature_metrics["smape"] = metrics.symmetric_mean_absolute_percentage_error(
    #     Y, Y_, reduction_axes=(0,)
    # )
    # feature_metrics["rmse"] = metrics.root_mean_square_error(Y, Y_, reduction_axes=(0,))
    # feature_metrics["nrmse"] = metrics.normalized_root_mean_square_error(
    #     Y, Y_, reduction_axes=(0,)
    # )
    # project.log("Feature metrics")
    # for k, v in feature_metrics.items():
    #     project.log(f"   {k}: {v}")
    # project.log("")

    # # Sample metrics
    # sample_metrics = SortedDict()
    # sample_metrics["mae"] = metrics.mean_absolute_error(Y, Y_, reduction_axes=(1,))
    # sample_metrics["mape"] = metrics.mean_absolute_percentage_error(
    #     Y, Y_, reduction_axes=(1,)
    # )
    # sample_metrics["smape"] = metrics.symmetric_mean_absolute_percentage_error(
    #     Y, Y_, reduction_axes=(1,)
    # )
    # sample_metrics["rmse"] = metrics.root_mean_square_error(Y, Y_, reduction_axes=(1,))
    # sample_metrics["nrmse"] = metrics.normalized_root_mean_square_error(
    #     Y, Y_, reduction_axes=(1,)
    # )
    # project.log("Sample metrics:")
    # for k, v in sample_metrics.items():
    #     project.log(f"   {k}: {v}")
    # project.log("")

    # # Select worst, best, mean, median samples
    # error_metric = "mae"
    # sample_errors = sample_metrics[error_metric]
    # total_error = total_metrics[error_metric]
    # sorted_sample_ids = np.argsort(sample_errors)
    # sorted_sample_errors = sample_errors[sorted_sample_ids]
    # # nonconvergence_cutoff = 100000.0
    # # converged_sorted_sample_errors = sorted_sample_errors[
    # #     sorted_sample_errors < nonconvergence_cutoff
    # # ]
    # # converged_sorted_sample_ids = sorted_sample_ids[
    # #     sorted_sample_errors < nonconvergence_cutoff
    # # ]
    # converged_sorted_sample_ids = sorted_sample_ids
    # converged_sorted_sample_errors = sorted_sample_errors
    # best_sample_id = converged_sorted_sample_ids[0]
    # worst_sample_id = converged_sorted_sample_ids[-1]
    # median_sample_id = converged_sorted_sample_ids[
    #     len(converged_sorted_sample_ids) // 2
    # ]
    # perc25_sample_id = converged_sorted_sample_ids[
    #     len(converged_sorted_sample_ids) * 1 // 4
    # ]
    # perc75_sample_id = converged_sorted_sample_ids[
    #     len(converged_sorted_sample_ids) * 3 // 4
    # ]
    # mean_sample_id = np.argmin(
    #     np.abs(sample_errors - np.mean(converged_sorted_sample_errors))
    # )
    # project.log(f"{project_name}: Sample quality: sample {error_metric}")
    # project.log(
    #     f"{project_name}:\n"
    #     + f"    mean: {sample_errors[mean_sample_id]}"
    #     + f" ({test_samples[mean_sample_id]})\n"
    #     + f"    best: {sample_errors[best_sample_id]}"
    #     + f" ({test_samples[best_sample_id]})\n"
    #     + f"    worst: {sample_errors[worst_sample_id]}"
    #     + f" ({test_samples[worst_sample_id]})\n"
    #     + f"    median: {sample_errors[median_sample_id]}"
    #     + f" ({test_samples[median_sample_id]})\n"
    #     + f"    perc25: {sample_errors[perc25_sample_id]}"
    #     + f" ({test_samples[perc25_sample_id]})\n"
    #     + f"    perc75: {sample_errors[perc75_sample_id]}"
    #     + f" ({test_samples[perc75_sample_id]})\n"
    # )
'''

# Plot
if PLOT:
    project.log(">> Plotting results...")
    test_result_file = Path(model.base_model_dir, "test_results.npz")
    test_results = np.load(test_result_file)
    X = test_results["X"][:num_plot_samples]
    Y = test_results["Y"][:num_plot_samples]
    Y_ = test_results["Y_"][:num_plot_samples]
    # Plot individual samples
    project.log(">>> Plotting individual samples")
    for i, (x, y, y_) in enumerate(zip(X, Y, Y_)):
        image = np.squeeze(x)
        label = np.argmax(y, axis=-1)
        prediction = np.argmax(y_, axis=-1)
        # Draw
        fig = plt.figure(figsize=(4, 3))
        plt.imshow(image, cmap=sns.color_palette("mako_r", as_cmap=True))
        plt.axis("off")
        fig.suptitle(
            f"label = {label:d}, prediction = {prediction:d} ({y_[prediction]})"
        )
        fig.tight_layout()
        # Save and close
        plot_file = os.path.join(
            model.plot_dir, f"best_to_worst_{i:05d}_label={label:d}"
        )
        plt.savefig(plot_file)
        plt.close()

# Explain
if ANALYZE:
    analze_samples = test_samples[:10]
    X, Y = model.read_tfrecords(test_samples[:num_plot_samples])
    # # PCA analysis
    # Y_, C1, C2 = model.predict(X, return_states=True)
    # for si in range(num_plot_samples):
    #     # Compute PCA on single sample      # TODO: pca over multiple samples?
    #     cov = np.dot(C1[si, ...].T, C1[si, ...])
    #     u, s, v = np.linalg.svd(cov, compute_uv=True)
    #     pc_axes = np.dot(C1[si, ...], u[:, :num_principal_components])
    #     # Plot loading and relaxation
    #     fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6),
    #                              gridspec_kw={"height_ratios": [0.5, 1, 1]})
    #     plt.sca(axes[0])
    #     plt.plot(X[si, :, 0], "C0", label="strain")
    #     plt.ylabel("strain [-]")
    #     plt.sca(axes[1])
    #     plt.plot(Y[si, :, 1], "C2", label="strain_plastic_ref")
    #     plt.plot(Y_[si, :, 1], "C2", linestyle="dashed",
    #              label="strain_plastic")
    #     for pc in range(num_principal_components):
    #         plt.plot(pc_axes[:, pc], "C{:d}".format(3 + pc),
    #                  linestyle="dotted", label="state{:d}".format(pc))
    #     plt.ylabel("history [-]")
    #     plt.sca(axes[2])
    #     plt.plot(Y[si, :, 0], "C1", label="stress_ref")
    #     plt.plot(Y_[si, :, 0], "C1", linestyle="dashed", label="stress")
    #     plt.ylabel("stress [-]")
    #     plt.xlabel("increments [-]")
    #     fig.legend(loc="upper center", ncol=4)
    #     plt.tight_layout()
    #     plt.draw()
    #     plt.savefig(os.path.join(model.plot_dir, "pca_{:06d}".format(si)))
    #     plt.close()
    # Local response propagation
    analysis = model.analyze(
        X[:num_plot_samples, ...],
        method="lrp.epsilon",
        neuron_selection_mode="all",
        # neuron_selection=0,
        plot=True,
        checkpoint="last",
    )
    project.log(">> Evaluation complete.")

    # test_loss = model.eval_data(
################################################################################
# Finished

project.log("done")
