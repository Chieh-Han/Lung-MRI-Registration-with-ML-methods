import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from cids.tensorflow import layers as clayers

class Transform(tf.keras.Model):
    """
    Simple transform model to apply dense or affine transforms.
    """

    def __init__(self,
                 inshape,
                 affine=False,
                 interp_method='linear',
                 rescale=None,
                 fill_value=None,
                 nb_feats=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            affine: Enable affine transform. Default is False.
            interp_method: Interpolation method. Can be 'linear' or 'nearest'. Default is 'linear'.
            rescale: Transform rescale factor. Default is None.
            fill_value: Fill value for SpatialTransformer. Default is None.
            nb_feats: Number of source image features. Default is 1.
        """

        # configure inputs
        ndims = len(inshape)
        scan_input = tf.keras.Input((*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = tf.keras.Input((ndims, ndims + 1), name='trf_input')
        else:
            trf_shape = inshape if rescale is None else [int(d / rescale) for d in inshape]
            trf_input = tf.keras.Input((*trf_shape, ndims), name='trf_input')

        trf_scaled = trf_input if rescale is None else clayers.RescaleTransform(rescale)(trf_input)

        # transform and initialize the keras model
        trf_layer = clayers.SpatialTransformer(interp_method=interp_method,
                                              name='transformer',
                                              fill_value=fill_value)
        y_source = trf_layer([scan_input, trf_scaled])
        super().__init__(inputs=[scan_input, trf_input], outputs=y_source)


def fwd(img, flow):
    volshape = img[:,:,:].shape
    trf = Transform(volshape)
    flow = flow[:,:,:]
    Y_fwd = trf([img[None, ..., None], flow[None]])
    Y_fwd = Y_fwd[0,:, :, :, 0]
    return Y_fwd

patients = []

for file in glob.glob('/mnt/data/stud-uexja/Documents/MRI_Registration/RegisteredImages_Elastix/*.npz'):
    tmp = Path(file).stem.split('TestSample')[1]
    patients.append(tmp)

img_list = []
inv_list = []
flow_list = []
invflow_list = []


for i in range(48):
    current_patient = sorted(patients)[i]
    wdir = "/mnt/data/stud-uexja/Documents/MRI_Registration/RegisteredImages_Elastix"
    patient_file = Path(wdir, "TestSample" + current_patient + ".npz")
    regimg = np.load(patient_file)["regimg"]
    flowfield = np.load(patient_file)["flowfield"]
    flowfield_inv = np.load(patient_file)["flowfield_inv"]
    invimg = fwd(regimg, flowfield_inv)

    img_list.append(regimg)
    inv_list.append(invimg)
    flow_list.append(flowfield)
    invflow_list.append(flowfield_inv)

Y_reg = np.asarray(img_list)
Y_neg = np.asarray(inv_list)
Y_fwdfield = np.asarray(flow_list)
Y_invfield = np.asarray(invflow_list)

np.savez_compressed(f'/mnt/data/stud-uexja/Documents/MRI_Registration/ANTsReg/ElastixData', 
                    Y_reg=Y_reg,
                    Y_neg=Y_neg,
                    Y_fwdfield = Y_fwdfield,
                    Y_invfield = Y_invfield,
                )