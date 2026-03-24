import numpy as np
import matplotlib.pyplot as plt
import pystrum

import scipy
import tensorflow as tf
from cids.tensorflow import layers as clayers

loss_type = "MutualInformation"

result1 = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--{loss_type}--epoch:1001--int_steps:0--bidir:False--reg_field:svf/test_results_inferdata.npz')
result2 = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--{loss_type}--epoch:1001--int_steps:7--bidir:False--reg_field:svf/test_results_inferdata.npz')
result3 = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed+moving--{loss_type}--epoch:1001--int_steps:0--bidir:True--reg_field:svf/test_results_inferdata.npz')

# def load_data(data):
#     X = data['X'][:,:,:,:,0]
#     Y = data['Y'][:,:,:,:,0]
#     Y_reg = data['Y_reg'][:,:,:,:,0]
#     Y_neg = data['Y_neg'][:,:,:,:,0]
#     # Y_fwdfield = data['Y_fwdfield'][:,:,:,:,:]
#     # Y_post = data['Y_posfwd'][:,:,:,:,:]
#     return X, Y, Y_reg, Y_neg #, Y_fwdfield, Y_post

# X, Y, Y_reg_1, Y_neg_1 = load_data(result1)
# X, Y, Y_reg_2, Y_neg_2 = load_data(result2)
# X, Y, Y_reg_3, Y_neg_3 = load_data(result3)

Y_flow_1= result1['Y_fwdfield'][:,:,:,:,0::2]
Y_flow_2= result2['Y_posfwd'][:,:,:,:,0::2]
Y_flow_3= result3['Y_fwdfield'][:,:,:,:,0::2]

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

def grid_slider(img1, img2, img3, batch=36, slice_num=32):
    plt.rcParams.update({
    'font.size': 14,      # Global font size for all elements
    'axes.titlesize': 20, # Title font size
    'axes.labelsize': 14, # Label font size
    'xtick.labelsize': 12, # X-tick labels font size
    'ytick.labelsize': 12, # Y-tick labels font size
    'legend.fontsize': 12, # Legend font size
    })

    volshape = img1[batch, slice_num, :, :, 0].shape

    warp1 = img1[batch, slice_num, :, :]
    warp2 = img2[batch, slice_num, :, :]
    warp3 = img3[batch, slice_num, :, :]

    
    sr = 2
    srvolshape = [sr * f for f in volshape]

    warp1 = scipy.ndimage.zoom(warp1, [sr, sr, 1]) * sr
    warp2 = scipy.ndimage.zoom(warp2, [sr, sr, 1]) * sr
    warp3 = scipy.ndimage.zoom(warp3, [sr, sr, 1]) * sr


    grid = pystrum.pynd.ndutils.bw_grid(vol_shape=srvolshape, spacing=11)
    trf = Transform(srvolshape)

    warped_grid1 = trf([grid[None, ..., None], warp1[None]])
    warped_grid2 = trf([grid[None, ..., None], warp2[None]])
    warped_grid3 = trf([grid[None, ..., None], warp3[None]])

    f2, axarr = plt.subplots(1,3,figsize=(16,16))
    axarr[0].imshow(warped_grid1[0, ..., 0], cmap="gray")
    axarr[0].set_title("MI No Mod")
    axarr[0].set_axis_off()
    axarr[1].imshow(warped_grid2[0, ..., 0], cmap="gray")
    axarr[1].set_title("MI VecInt") 
    axarr[1].set_axis_off()
    axarr[2].imshow(warped_grid3[0, ..., 0], cmap="gray")
    axarr[2].set_title("MI Bidir") 
    axarr[2].set_axis_off()
    plt.savefig('/mnt/data/stud-uexja/Documents/MRI_Registration/Thesis_Img/MI_gridviews', dpi = 300)  # Save as PNG image

    # Optionally, you can also display the plot
    plt.show()

grid_slider(Y_flow_1, Y_flow_2, Y_flow_3)