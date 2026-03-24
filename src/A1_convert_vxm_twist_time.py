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
import os, sys
from pathlib import Path

import numpy as np
import math
import glob

import tensorflow as tf
import tqdm

from cids.data import DataDefinition
from cids.data import DataWriter
from cids.data import Feature
from kadi_ai import KadiAIProject
import ants
import scipy.ndimage

################################################################################
# Data paths

project_name = "vxm_twist_time_128_final"
project_dir = Path("/mnt/data/stud-uexja/DATA") / project_name
project = KadiAIProject(project_name, root=project_dir)

# Project creates an `input_dir` in the `project_dir`, which stores converted
#   input data as tfrecords in a subdirectory `tfrecord`
tfrecord_dir = Path(project.input_dir) / "tfrecord"

################################################################################
# Data definition

data_definition = DataDefinition(
    Feature(
        "fixed",
        [None, 64, 128, 128, 1],
        data_format="NXYZF",
        dtype=tf.string,
        decode_str_to=tf.float32,
    ),
    Feature(
        "moving",
        [None, 64, 128, 128, 1],
        data_format="NXYZF",
        dtype=tf.string,
        decode_str_to=tf.float32,
    ),
    dtype=tf.float32,
)

project.data_definition = data_definition
################################################################################

# fix data
def register_image(img_fixed, img_moving, wdir):
    wdir = Path(wdir)
    rdir = wdir / "Registration"
    if not os.path.isdir(rdir):
        os.mkdir(rdir)
    img_fixed = ants.from_numpy(img_fixed)
    img_moving = ants.from_numpy(img_moving)
    os.chdir(rdir)
    reg1 = ants.registration(
        img_fixed,
        img_moving,
        "AffineFast",
        write_composite_transform=True,
        outprefix=str(rdir) + "/",
    )
    img_moving_tf = ants.apply_transforms(
        fixed=img_fixed,
        moving=img_moving,
        transformlist=reg1["fwdtransforms"],
    ).numpy()
    return img_moving_tf

def reshape(img, target_height, target_width):
    original_depth, original_height, original_width = img.shape

    # Calculate scale factor based on the most constraining dimension (smallest ratio)
    scale_factor = min(target_height / original_height, target_width / original_width)

    # Calculate the new dimensions
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)
    new_depth = original_depth  # Assuming depth should not be scaled

    # Calculate zoom factors for each dimension
    zoom_factor_z = new_depth / original_depth
    zoom_factor_y = new_height / original_height
    zoom_factor_x = new_width / original_width

    # Apply the zoom with calculated factors
    reshaped = scipy.ndimage.zoom(img, (zoom_factor_z, zoom_factor_y, zoom_factor_x))
    return reshaped

def NormalizeData(image):
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

################################################################################
# Read data


patients = []

for file in glob.glob('/mnt/data/stud-uexja/Documents/MRI_Registration/TestPatientsRegistration/*.npz'):
    tmp = Path(file).stem.split('Cosyconet')[1]
    patients.append(tmp)

def data_gen(current_patient):
    # Load data
    aff_reg_wdir = "/mnt/data/stud-uexja/Documents/MRI_Registration/TestPatientsRegistration"
    patient_file = np.load(aff_reg_wdir + "/Cosyconet" + current_patient + '.npz')
    t = 1
    moving_list = []
    fixed_list = []
    while t <= 3:
        data_fixed = patient_file['imgsvibecor'].astype(np.float32)

        # normalization
        if data_fixed.shape != data_shape:
            data_fixed = reshape(data_fixed, data_shape[1], data_shape[2])
        data_fixed = NormalizeData(data_fixed)

        # median filter (optional)
        # data_fixed = scipy.ndimage.median_filter(data_fixed, size=4, mode="reflect")
        
        data_moving = patient_file['imgs'].astype(np.float32)
        data_moving = np.transpose(data_moving, [3,1,2,0])
        data_moving = data_moving[...,t]

        # reshape to the size of data definition
        if data_moving.shape != data_fixed.shape:
            data_moving = reshape(data_moving, data_shape[1], data_shape[2])
        data_moving = NormalizeData(data_moving)

        # Affine register moving to fixed
        data_moving_tf = register_image(data_fixed, data_moving, aff_reg_wdir)
        data_moving_tf = NormalizeData(data_moving_tf)
        moving_list.append(data_moving_tf)
        fixed_list.append(data_fixed)
        t += 1
     
    src_sample = [[moving_list[i], fixed_list[i]] for i in range(0,len(moving_list))]
    return src_sample

###############################################################################
# Data processing


def read_and_process(src_sample):
    """Read and process source data."""
    # Do some preprocessing
    moving = src_sample[0]
    fixed = src_sample[1]

    # Pack into dictionary
    sample = {}
    sample["moving"] = moving
    sample["fixed"] = fixed
    return sample


################################################################################
# Start processing

# Create a data converter object
data_writer = DataWriter(data_definition)
data_shape = tuple(data_definition["fixed"].data_shape[1:4])

# Loop over all pairs of source files with a pretty progress bar
n = 0
for i in range(len(patients)):
    j = 1
    current_patient = sorted(patients)[i]
    print(current_patient)
    src_data = data_gen(current_patient)

    subject_name=f'subject_{i+1:02d}'
    for src_sample in tqdm.tqdm(
        src_data,
        total=len(src_data),
        file=project.stream_to_logger(),
        leave=True,
        desc="Conversion",
        unit="sources",
        dynamic_ncols=True,
    ):
        # Process sample
        sample = read_and_process(src_sample)
        out_file = tfrecord_dir / f"{n:06d}_sample_{j:05d}_{subject_name}.tfrecord"
        # Write sample to file
        try:
            data_writer.write_example(out_file, sample)
        except KeyError as e:
            project.warn(f"Missing key {e.args[0]} in: {os.fspath(out_file)}")
            continue

        j += 1
        n += 1

    project.log(f"Done processing: {os.fspath(out_file)}")

# Write the data definition and the features to a human-readable json file
#   The json file can also be loaded directly later-on for training.
project.data_definition = data_definition
project.to_json(write_data_definition=True)

project.log("Done.")
