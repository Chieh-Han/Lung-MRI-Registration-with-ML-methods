import numpy as np
import statistics
import skimage

result1 = np.load('/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--NCC--epoch:1001--int_steps:7--bidir:False--reg_field:preintegrated/test_results_inferdata.npz')
result2 = np.load('/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--NCC--epoch:5001--int_steps:7--bidir:False--reg_field:preintegrated/test_results_inferdata_1000.npz')

X = result1['X'][:,:,:,:,0]
Y = result1['Y'][:,:,:,:,0]
Y_reg_1 = result1['Y_reg'][:,:,:,:,0]
Y_neg_1 = result1['Y_neg'][:,:,:,:,0]
# Y_fwdfield_1 = result1['Y_fwdfield'][:,:,:,:,:]
# Y_post_1 = result1['Y_posfwd'][:,:,:,:,:]

Y_reg_2 = result2["Y_reg"][:,:,:,:,0]
Y_neg_2 = result2["Y_neg"][:,:,:,:,0]
# Y_fwdfield_2 = result2["Y_fwdfield"]
# Y_invfield_2 = result2["Y_invfield"]


def metric_avg(img1,img2, img3, img4,img5, img6):
    nmi_og_list = []
    nmi_reg_1_list = []
    nmi_reg_2_list = []
    nmi_inv_1_list = []
    nmi_inv_2_list = []

    ssim_og_list = []
    ssim_reg_1_list = []
    ssim_reg_2_list = []
    ssim_inv_1_list = []
    ssim_inv_2_list = []

    for batch in range (48):
        if batch in [11,12,13]:
            continue
        tmpimg1 = np.squeeze(img1[batch,:,:,:])
        tmpimg2 = np.squeeze(img2[batch,:,:,:])

        tmpimg3 = np.squeeze(img3[batch,:,:,:])
        tmpimg4 = np.squeeze(img4[batch,:,:,:])
        
        tmpimg5 = np.squeeze(img5[batch,:,:,:])
        tmpimg6 = np.squeeze(img6[batch,:,:,:])

        nmi_original = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg1)
        nmi_1_reg = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg5)
        nmi_2_reg = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg6)
        nmi_1_inv = skimage.metrics.normalized_mutual_information(tmpimg1, tmpimg3)
        nmi_2_inv = skimage.metrics.normalized_mutual_information(tmpimg1, tmpimg4)

        ssim_original = skimage.metrics.structural_similarity(tmpimg2, tmpimg1, data_range = tmpimg1.max() - tmpimg1.min())
        ssim_1_reg = skimage.metrics.structural_similarity(tmpimg2, tmpimg5, data_range = tmpimg5.max() - tmpimg5.min())
        ssim_2_reg = skimage.metrics.structural_similarity(tmpimg2, tmpimg6, data_range = tmpimg6.max() - tmpimg6.min())
        ssim_1_inv = skimage.metrics.structural_similarity(tmpimg1, tmpimg3, data_range = tmpimg3.max() - tmpimg3.min())
        ssim_2_inv = skimage.metrics.structural_similarity(tmpimg1, tmpimg4, data_range = tmpimg4.max() - tmpimg4.min())
        
        nmi_og_list.append(nmi_original)
        nmi_reg_1_list.append(nmi_1_reg)
        nmi_reg_2_list.append(nmi_2_reg)
        nmi_inv_1_list.append(nmi_1_inv)
        nmi_inv_2_list.append(nmi_2_inv)

        ssim_og_list.append(ssim_original)
        ssim_reg_1_list.append(ssim_1_reg)
        ssim_reg_2_list.append(ssim_2_reg)
        ssim_inv_1_list.append(ssim_1_inv)
        ssim_inv_2_list.append(ssim_2_inv)

    nmi_avg_og = statistics.mean(nmi_og_list)
    nmi_avg_reg1 = statistics.mean(nmi_reg_1_list)
    nmi_avg_reg2 = statistics.mean(nmi_reg_2_list)
    nmi_avg_inv1 = statistics.mean(nmi_inv_1_list)
    nmi_avg_inv2 = statistics.mean(nmi_inv_2_list)

    ssmi_avg_og = statistics.mean(ssim_og_list)
    ssmi_avg_reg1 = statistics.mean(ssim_reg_1_list)
    ssmi_avg_reg2 = statistics.mean(ssim_reg_2_list)
    ssmi_avg_inv1 = statistics.mean(ssim_inv_1_list)
    ssmi_avg_inv2 = statistics.mean(ssim_inv_2_list)

    print(f"OG NMI {nmi_avg_og} SSMI {ssmi_avg_og}")

    print(f"Normalized Mutual Information:\nReg 1: {nmi_avg_reg1}, 2: {nmi_avg_reg2}")
    print(f"Inv 1: {nmi_avg_inv1}, 2: {nmi_avg_inv2}")
    print()
    print(f"Structural Similarity:\nReg  1: {ssmi_avg_reg1}, 2: {ssmi_avg_reg2}")
    print(f"Inv 1: {ssmi_avg_inv1}, 2: {ssmi_avg_inv2}")

metric_avg(
    img1 = X,
    img2 = Y,
    img3 = Y_neg_1,
    img4 = Y_neg_2,
    img5 = Y_reg_1,
    img6 = Y_reg_2,
)