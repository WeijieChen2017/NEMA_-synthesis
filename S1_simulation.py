import os
import glob
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def maxmin_norm(data):
    MAX = np.amax(data)
    MIN = np.amin(data)
    data = (data - MIN)/(MAX-MIN)
    return data

import nibabel
from nibabel import processing

def nib_smooth(file_mri, data, fwhm, tag, save_path):
    nii_file = nibabel.Nifti1Image(data, file_mri.affine, file_mri.header)
    smoothed = processing.smooth_image(nii_file, fwhm=idx_fwhm, mode='nearest')
    smoothed_data = maxmin_norm(np.asanyarray(smoothed.dataobj))
    smoothed_file = nibabel.Nifti1Image(smoothed_data, file_mri.affine, file_mri.header)
#     print(np.amax(smoothed_file.get_fdata()))
    nibabel.save(smoothed_file, save_path+"fwhm_"+str(idx_fwhm)+"_"+tag+".nii")
    print("fwhm_"+str(idx_fwhm)+"_"+tag+".nii")

name_dataset = "stick"

for folder_name in ["trainA", "trainB", "testA", "testB"]:
    path = "./pytorch-CycleGAN-and-pix2pix/datasets/"+name_dataset+"/"+folder_name+"/"
    if not os.path.exists(path):
        os.makedirs(path)
        
blur_path = "./data/"+name_dataset+"/blur/"
if not os.path.exists(blur_path):
    os.makedirs(blur_path)
    
pure_path = "./data/"+name_dataset+"/pure/"
if not os.path.exists(pure_path):
    os.makedirs(pure_path)

import nibabel
from skimage.transform import radon, iradon

fwhm_hub = [8]
# gau_sigma_hub = [1e-3,3e-3,5e-3,7e-3,9e-3]
# poi_sigma_hub = [1,3,5,7,9]
# gau_sigma_hub=[1e-3, 5e-3]
# poi_sigma_hub=[1, 5]
gau_sigma_hub=[1*1e-2,3*1e-2]
poi_sigma_hub=[1e2]
# gau_sigma_hub=[]
# poi_sigma_hub=[]
flag_Radon = True
# fwhm_hub = [0, 0.5, 1, 1.5, 2, 2.5]
theta = np.linspace(0., 360., 28*4, endpoint=False) # max(image.shape)

print("Gau noise: ", gau_sigma_hub)
print("Poi noise: ", poi_sigma_hub)
print("Radon: ", flag_Radon)


list_ori = glob.glob(pure_path+"*.nii")
list_ori.sort()
for path_ori in list_ori:
    print(path_ori)
    file_mri = nibabel.load(path_ori)
    data_mri = np.asanyarray(file_mri.dataobj)
    file_name = os.path.basename(path_ori)
#     nibabel.save(file_mri, pure_path+file_name)
    print(data_mri.shape)

    for idx_fwhm in fwhm_hub:
        tag = file_name[:-4]+""
        nib_smooth(file_mri, data_mri, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

        # gaussian noise
        for idx_gau_sigma in gau_sigma_hub:
            noise = np.random.normal(0, idx_gau_sigma*np.var(data_mri), data_mri.shape)
            noisy_img = data_mri + noise
            tag = file_name[:-4]+"_gs_"+'{:.0e}'.format(idx_gau_sigma)
            nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

        # poisson noise
        for idx_poi_sigma in poi_sigma_hub:
            noise = np.random.poisson(size=data_mri.shape, lam=np.mean(data_mri)*idx_poi_sigma)
            noisy_img = data_mri + noise
            tag = file_name[:-4]+"_ps_"+'{:.0e}'.format(idx_poi_sigma)
            nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)
    
    if flag_Radon:
        # radon transform, https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html
        radon_img = np.zeros(data_mri.shape)
        for idx_slice in range(data_mri.shape[2]):
            orginal_img = data_mri[:, :, idx_slice]
            sinogram = radon(orginal_img, theta=theta, circle=False)
            reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
            radon_img[:, :, idx_slice] = reconstruction_fbp

        for idx_fwhm in fwhm_hub:
            tag = file_name[:-4]+"_radon"
            nib_smooth(file_mri, radon_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

            # gaussian noise
            for idx_gau_sigma in gau_sigma_hub:
                noise = np.random.normal(0, idx_gau_sigma*np.var(data_mri), data_mri.shape)
                noisy_img = radon_img + noise
                tag = file_name[:-4]+"_radon_gs_"+'{:.0e}'.format(idx_gau_sigma)
                nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)

            # poisson noise
            for idx_poi_sigma in poi_sigma_hub:
                noise = np.random.poisson(size=data_mri.shape, lam=np.mean(data_mri)*idx_poi_sigma)
                noisy_img = radon_img + noise
                tag = file_name[:-4]+"_radon_ps_"+'{:.0e}'.format(idx_poi_sigma)
                nib_smooth(file_mri, noisy_img, fwhm=idx_fwhm, tag=tag, save_path=blur_path)