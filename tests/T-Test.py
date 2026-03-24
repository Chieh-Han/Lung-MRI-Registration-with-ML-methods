import numpy as np
import scipy.stats as stats 
import skimage
# loss_type = "MutualInformation"
result2 = np.load(f'/mnt/data/stud-uexja/DATA/vxm_twist_time_128/RESULTS/VXM_Twist--moving+fixed--fixed--NCC--epoch:5001--int_steps:7--bidir:False--reg_field:preintegrated/test_results_inferdata.npz')
result1 = np.load(f'/mnt/data/stud-uexja/Documents/MRI_Registration/ANTsReg/ElastixData.npz')

X = result2['X'][:,:,:,:,0]
Y = result2['Y'][:,:,:,:,0]
Y_reg_1 = result1['Y_reg'][:,:,:,:]
Y_neg_1 = result1['Y_neg'][:,:,:,:]
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

        # nmi_original = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg1)
        nmi_1_reg = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg5)
        nmi_2_reg = skimage.metrics.normalized_mutual_information(tmpimg2, tmpimg6)
        nmi_1_inv = skimage.metrics.normalized_mutual_information(tmpimg1, tmpimg3)
        nmi_2_inv = skimage.metrics.normalized_mutual_information(tmpimg1, tmpimg4)

        # ssim_original = skimage.metrics.structural_similarity(tmpimg2, tmpimg1, data_range = tmpimg1.max() - tmpimg1.min())
        ssim_1_reg = skimage.metrics.structural_similarity(tmpimg2, tmpimg5, data_range = tmpimg5.max() - tmpimg5.min())
        ssim_2_reg = skimage.metrics.structural_similarity(tmpimg2, tmpimg6, data_range = tmpimg6.max() - tmpimg6.min())
        ssim_1_inv = skimage.metrics.structural_similarity(tmpimg1, tmpimg3, data_range = tmpimg3.max() - tmpimg3.min())
        ssim_2_inv = skimage.metrics.structural_similarity(tmpimg1, tmpimg4, data_range = tmpimg4.max() - tmpimg4.min())
        
        # nmi_og_list.append(nmi_original)
        nmi_reg_1_list.append(nmi_1_reg)
        nmi_reg_2_list.append(nmi_2_reg)
        nmi_inv_1_list.append(nmi_1_inv)
        nmi_inv_2_list.append(nmi_2_inv)

        # ssim_og_list.append(ssim_original)
        ssim_reg_1_list.append(ssim_1_reg)
        ssim_reg_2_list.append(ssim_2_reg)
        ssim_inv_1_list.append(ssim_1_inv)
        ssim_inv_2_list.append(ssim_2_inv)

    print(len(ssim_reg_2_list))

    nmi_p_reg = stats.ttest_rel(nmi_reg_1_list, nmi_reg_2_list)
    nmi_p_inv = stats.ttest_rel(nmi_inv_1_list, nmi_inv_2_list)

    ssim_p_reg = stats.ttest_rel(ssim_reg_1_list, ssim_reg_2_list)
    ssim_p_inv = stats.ttest_rel(ssim_inv_1_list, ssim_inv_2_list) 


    print(f"nmi p-value between 1 & 2 registrations:{nmi_p_reg}")
    print(f"nmi p-value between 1 & 2 inverse:{nmi_p_inv}")

    print(f"ssim p-value between 1 & 2 registrations:{ssim_p_reg}")
    print(f"ssim p-value between 1 & 2 inverse:{ssim_p_inv}")


metric_avg(
    img1 = X,
    img2 = Y,
    img3 = Y_neg_1,
    img4 = Y_neg_2,
    img5 = Y_reg_1,
    img6 = Y_reg_2,
)