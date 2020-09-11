from lungmask import mask
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from img_viz.medical import MedicalImageVisualizer
from os.path import join
import os
from inout.io_common import save_image
from Preproc.UtilsPreproc import resample_to_reference_itk
from preproc.UtilsItk import copyItkImage

# -------------- Test example -----------------
# input_image = sitk.ReadImage("/data/UM/COVID/Kaggle_Mosmed/studies/study_0001.nii")
# segmentation = mask.apply(input_image)  # default model is U-net(R231)
# viz_obj = MedicalImageVisualizer(disp_images=False, output_folder="/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/LungSegmentationSoftware/OUTPUT")
# viz_obj.plot_img_and_ctrs_np(sitk.GetArrayFromImage(input_image), [segmentation],
#                               title="Test Lung seg")

# Visualizing Mosmed data
data_folder = "/data/UM/COVID/Kaggle_Mosmed"
output_folder = "/data/UM/COVID/Kaggle_Mosmed/lung_masks"

viz_obj = MedicalImageVisualizer(output_folder=join(data_folder, "output_imgs"), disp_images=False)
mask_files = os.listdir(join(data_folder, "masks"))
mask_files.sort()
for c_mask_file_name in mask_files:
    c_case = int(c_mask_file_name.split("_")[1])
    c_img_file_name = F"study_0{c_case:03d}.nii"
    c_img_file = join(data_folder, "studies", c_img_file_name)
    print(F"--- Working with {c_img_file} ---")
    itk_img = sitk.ReadImage(c_img_file)

    np_lung_mask= mask.apply(itk_img)  # default model is U-net(R231)
    itk_lung_mask = copyItkImage(itk_img, np_lung_mask)
    viz_obj.plot_img_and_ctrs_itk(itk_img, [itk_lung_mask], title=c_img_file_name, draw_only_ctrs=True,
                                  file_name_prefix=c_img_file_name)

    save_image(itk_lung_mask, output_folder, c_mask_file_name)