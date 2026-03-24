import ants
import numpy as np
import matplotlib.pyplot as plt

import time

start_time = time.time()

data = np.load("/mnt/data/groups/public/ChiehHan2Julian/X/Test_data.npy.npz")
moving = data['moving']
fixed = data['fixed']

def register_image(img_fixed, img_moving):

    img_f = ants.from_numpy(img_fixed)
    img_m = ants.from_numpy(img_moving)

    reg1 = ants.registration(
        img_f, 
        img_m, 
        'SyN', 
        syn_sampling=32, 
        reg_iterations=(40,30,20,10),
        grad_step=0.25,
        aff_shrink_factors=(6, 4, 2, 1))
    
    forward_transform = reg1['fwdtransforms']
    inverse_transform = reg1['invtransforms']

    fwd_warped_image = ants.apply_transforms(fixed=img_f, moving=img_m, transformlist=forward_transform)
    fwd_warped_image_np = fwd_warped_image.numpy()

    inv_warped_image = ants.apply_transforms(fixed=img_m, moving=fwd_warped_image, transformlist=inverse_transform)
    inv_warped_image_np = inv_warped_image.numpy()
    
    fwd_disp_field = ants.image_read(forward_transform[0])
    fwd_disp_field_np = fwd_disp_field.numpy()

    inv_disp_field = ants.image_read(inverse_transform[1])
    inv_disp_field_np = inv_disp_field.numpy()

    return fwd_warped_image_np, inv_warped_image_np, fwd_disp_field_np, inv_disp_field_np


fwd_tf_list = []
inv_tf_list = []
fwd_field_list = []
inv_field_list = []

for i in range(48):
    m = moving[i,...]
    f = fixed[i,...]
    fwd_tf, inv_tf , fwd_field, inv_field= register_image(f,m)
    fwd_tf_list.append(fwd_tf)
    inv_tf_list.append(inv_tf)
    fwd_field_list.append(fwd_field)
    inv_field_list.append(inv_field)

fwd_tf_list = np.array(fwd_tf_list)
inv_tf_list = np.array(inv_tf_list)
fwd_field_list = np.array(fwd_field_list)
inv_field_list = np.array(inv_field_list)

np.savez_compressed(f'/mnt/data/stud-uexja/Documents/MRI_Registration/ANTsReg/ANTsData', 
                    Y_reg=fwd_tf_list,
                    Y_neg=inv_tf_list,
                    Y_fwdfield = fwd_field_list,
                    Y_invfield = inv_field_list,
                )

for i in range(1000000):
    pass
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")