import numpy as np
import matplotlib.pyplot as plt
import statistics

loss_type = "MutualInformation"

# results = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--{loss_type}--epoch:1001--int_steps:0--bidir:False--reg_field:svf/test_results_inferdata.npz')
# results = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--{loss_type}--epoch:1001--int_steps:7--bidir:False--reg_field:preintegrated/test_results_inferdata.npz')
# results = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--{loss_type}--epoch:5001--int_steps:7--bidir:False--reg_field:preintegrated/test_results_inferdata.npz')

results = np.load(f'/mnt/data/stud-uexja/Documents/MRI_Registration/ANTsReg/ElastixData.npz')
# X = results['X'][:,:,:,:,0]
# Y = results['Y'][:,:,:,:,0]
# Y_reg = results['Y_reg'][:,:,:,:,0]

# flow = results['Y_fwdfield'][:,:,:,:,:]
flow = results['Y_fwdfield'][:,:,:,:,:]

def compute_jacobian_determinant_3d(flow_field):
    """
    Compute the Jacobian determinant of a 3D flow field.

    Args:
        flow_field (np.array): The flow field of shape (H, W, D, 3), where the last dimension represents
                               the displacement in the x, y, and z directions.

    Returns:
        np.array: The Jacobian determinant field of shape (H, W, D).
    """
    flow_x = flow_field[..., 0]
    flow_y = flow_field[..., 1]
    flow_z = flow_field[..., 2]

    # Compute the gradients of the displacement components
    dFx_dx = np.gradient(flow_x, axis=2)
    dFx_dy = np.gradient(flow_x, axis=1)
    dFx_dz = np.gradient(flow_x, axis=0)

    dFy_dx = np.gradient(flow_y, axis=2)
    dFy_dy = np.gradient(flow_y, axis=1)
    dFy_dz = np.gradient(flow_y, axis=0)

    dFz_dx = np.gradient(flow_z, axis=2)
    dFz_dy = np.gradient(flow_z, axis=1)
    dFz_dz = np.gradient(flow_z, axis=0)

    # Construct the Jacobian matrix at each point
    J = np.zeros(flow_x.shape + (3, 3))

    J[..., 0, 0] = 1 + dFx_dx
    J[..., 0, 1] = dFx_dy
    J[..., 0, 2] = dFx_dz
    J[..., 1, 0] = dFy_dx
    J[..., 1, 1] = 1 + dFy_dy
    J[..., 1, 2] = dFy_dz
    J[..., 2, 0] = dFz_dx
    J[..., 2, 1] = dFz_dy
    J[..., 2, 2] = 1 + dFz_dz

    # Compute the Jacobian determinant for each point
    jacobian_determinant = np.linalg.det(J)

    return jacobian_determinant

def find_non_diffeomorphic_regions_3d(jacobian_determinant, tolerance=1e-6):
    """
    Identify regions where the Jacobian determinant is <= 0 (non-diffeomorphic regions) in 3D.

    Args:
        jacobian_determinant (np.array): The Jacobian determinant field of shape (H, W, D).

    Returns:
        np.array: A binary mask of shape (H, W, D), where 1 indicates non-diffeomorphic regions.
    """
    non_diffeomorphic_mask = (jacobian_determinant <= tolerance).astype(np.uint8)
    return non_diffeomorphic_mask

def quantify_non_diffeomorphic_regions(non_diffeomorphic_mask):
    """
    Quantify the percentage of non-diffeomorphic regions in the 3D mask.

    Args:
        non_diffeomorphic_mask (np.array): The 3D mask of non-diffeomorphic regions.

    Returns:
        float: The percentage of non-diffeomorphic voxels.
    """
    total_voxels = non_diffeomorphic_mask.size
    non_diffeomorphic_voxels = np.sum(non_diffeomorphic_mask)
    percentage_non_diffeomorphic = (non_diffeomorphic_voxels / total_voxels) * 100
    return percentage_non_diffeomorphic, non_diffeomorphic_voxels

def find_non_diffeomorphic_slices(non_diffeomorphic_mask):
    """
    Visualize only the slices from 3D data that contain non-diffeomorphic regions.

    Args:
        jacobian_determinant (np.array): The 3D Jacobian determinant field.
        non_diffeomorphic_mask (np.array): The 3D binary mask of non-diffeomorphic regions.
    """
    # Find slices that contain non-diffeomorphic regions
    slice_indices = np.where(np.any(non_diffeomorphic_mask, axis=(1, 2)))[0]

    return slice_indices

def plot_slices_with_comparison(jacobian_determinant, non_diffeomorphic_mask):
    """
    Visualize 2D slices of the Jacobian determinant and non-diffeomorphic regions from 3D data,
    along with a comparison of both on the same plot, but only if non-diffeomorphic slices are found.

    Args:
        jacobian_determinant (np.array): The 3D Jacobian determinant field.
        non_diffeomorphic_mask (np.array): The 3D binary mask of non-diffeomorphic regions.
        num_slices (int): Number of slices to visualize across the depth dimension.
    """
    # Find slices that contain non-diffeomorphic regions
    non_diffeomorphic_slices = find_non_diffeomorphic_slices(non_diffeomorphic_mask)
    
    # If no non-diffeomorphic regions are found, print a message and exit
    if len(non_diffeomorphic_slices) == 0:
        print("No non-diffeomorphic regions found in any slice. No comparison plots will be shown.")
        return
    
    # Plot all non-diffeomorphic slices
    print(f"number of non-diffeomorphic slices: {len(non_diffeomorphic_slices)}")
    num_slices_to_plot = min(len(non_diffeomorphic_slices), 5)
    
    # If more than 5 slices, select a subset of them
    if len(non_diffeomorphic_slices) > 5:
        slice_indices = np.linspace(0, len(non_diffeomorphic_slices) - 1, num_slices_to_plot, dtype=int)
        slice_indices = [non_diffeomorphic_slices[idx] for idx in slice_indices]
    else:
        slice_indices = non_diffeomorphic_slices

    # Plot the selected slices
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(15, 3 * len(slice_indices)))

    for i, slice_idx in enumerate(slice_indices):
        # Plot Jacobian determinant slice
        axes[i, 0].imshow(jacobian_determinant[slice_idx, :, :], cmap='viridis')
        axes[i, 0].set_title(f'Jacobian Determinant Slice {slice_idx}')
        axes[i, 0].axis('off')

        # Plot Non-Diffeomorphic mask slice
        axes[i, 1].imshow(non_diffeomorphic_mask[slice_idx, :, :], cmap='gray')
        axes[i, 1].set_title(f'Non-Diffeomorphic Mask Slice {slice_idx}')
        axes[i, 1].axis('off')

        # Overlay the mask on the determinant
        axes[i, 2].imshow(jacobian_determinant[slice_idx, :, :], cmap='viridis')
        axes[i, 2].imshow(non_diffeomorphic_mask[slice_idx, :, :], cmap='Reds', alpha=0.5)
        axes[i, 2].set_title(f'Overlay: Determinant + Mask Slice {slice_idx}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


percent_list = []
number_list = []
for batch in range (48):
    if batch in [11,12,13]:
        continue

    jacobian_determinant = compute_jacobian_determinant_3d(flow[batch,:,:,:,:])
    non_diffeomorphic_mask = find_non_diffeomorphic_regions_3d(jacobian_determinant)
    percentage_non_diffeomorphic, number_of_non_diffeomorphic_voxels = quantify_non_diffeomorphic_regions(non_diffeomorphic_mask)
    percent_list.append(percentage_non_diffeomorphic)
    number_list.append(int(number_of_non_diffeomorphic_voxels))

avg_percent = statistics.mean(percent_list)
avg_number = statistics.mean(number_list)

print(f"Percentage of non-diffeomorphic regions: {avg_percent:.2f}%")
print(f"Number of non-diffeomorphic regions: {avg_number}")
# plot_slices_with_comparison(jacobian_determinant, non_diffeomorphic_mask)

# Print statistics about the Jacobian determinant field
# print(f"Jacobian determinant min: {jacobian_determinant.min()}")
# print(f"Jacobian determinant max: {jacobian_determinant.max()}")
# print(f"Jacobian determinant mean: {jacobian_determinant.mean()}")
