from os.path import join
import os
import SimpleITK as sitk
from img_viz.medical import *
from img_viz.constants import SliceMode, PlaneTypes
from proj_inout.utils import printImageSummary, utilsSplitMask

_disp_imgs = False

def vizMosmedData():
    # Visualizing Mosmed data
    data_folder = "/data/UM/COVID/Kaggle_Mosmed"

    viz_obj = MedicalImageVisualizer(output_folder=join(data_folder, "output_imgs"), disp_images=_disp_imgs)
    mask_files = os.listdir(join(data_folder, "masks"))
    for c_mask_file_name in mask_files:
        c_case = int(c_mask_file_name.split("_")[1])
        c_mask_file = join(data_folder, "masks", c_mask_file_name)
        c_img_file_name = F"study_0{c_case:03d}.nii"
        c_img_file = join(data_folder, "studies", c_img_file_name)
        print(F"--- Working with {c_img_file} -- {c_mask_file}")
        itk_mask = sitk.ReadImage(c_mask_file)
        itk_img = sitk.ReadImage(c_img_file)
        np_mask = sitk.GetArrayFromImage(itk_mask)

        viz_obj.plot_img_and_ctrs_itk(itk_img, [itk_mask], title=c_img_file_name, draw_only_ctrs=True,
                                      file_name_prefix=c_img_file_name)
        printImageSummary(itk_img)

def vizMedSeg1():
    data_folder = "/data/UM/COVID/Medicalsegmentation_V1"
    viz_obj = MedicalImageVisualizer(output_folder=join(data_folder, "output_imgs"), disp_images=_disp_imgs)
    itk_img = sitk.ReadImage(join(data_folder, "tr_im.nii"))
    itk_mask = sitk.ReadImage(join(data_folder, "tr_mask.nii"))
    itk_mask_1, itk_mask_2, itk_mask_3 = utilsSplitMask(itk_mask)
    printImageSummary(itk_img)

    print("Plotting....")
    viz_obj.plot_img_and_ctrs_itk(itk_img, [itk_mask_1, itk_mask_2, itk_mask_3],
                                  title="All", draw_only_ctrs=True,
                                  labels=["Ground-glass", "Consolidation", "Pleural effusion"],
                                  file_name_prefix="All")
    print("Done....")

def vizMedSeg2():
    data_folder = "/data/UM/COVID/Medicalsegmentation_V2"

    viz_obj = MedicalImageVisualizer(output_folder=join(data_folder,"output_imgs"), disp_images=_disp_imgs)
    for c_id in range(1,10):
        c_mask_file = join(data_folder, "rp_msk", F"{c_id}.nii")
        c_img_file = join(data_folder, "rp_im", F"{c_id}.nii")
        c_lung_file = join(data_folder, "rp_lung_msk", F"{c_id}.nii")
        itk_img = sitk.ReadImage(c_img_file)
        itk_mask = sitk.ReadImage(c_mask_file)
        itk_lung = sitk.ReadImage(c_lung_file)
        np_mask = sitk.GetArrayFromImage(itk_mask)
        itk_mask_1, itk_mask_2, itk_mask_3 = utilsSplitMask(itk_mask)

        printImageSummary(itk_img)

        viz_obj.plot_img_and_ctrs_itk(itk_img, [itk_mask_1, itk_mask_2, itk_mask_3, itk_lung],
                                      title=c_id, draw_only_ctrs=True,
                                      labels=["Ground-glass", "Consolidation", "Pleural effusion", "Lung"],
                                      file_name_prefix=c_id)

    print("Done....")

def vizRadiopaedia():
    data_folder = "/data/UM/COVID/Kaggle_1_Radiopedia"

    viz_obj = MedicalImageVisualizer(output_folder=join(data_folder,"output_imgs"), disp_images=_disp_imgs)
    file_names = listdir(join(data_folder,"ct_scans"))
    for id, c_file_name in enumerate(file_names):
        c_img_file = join(data_folder, "ct_scans", c_file_name)
        if c_file_name.find("radiopaedia") != -1:
            c_file_name = c_file_name.replace("org_covid-19-pneumonia-","")
            c_file_name = c_file_name.replace("-dcm","")
            c_mask_file = join(data_folder, "infection_mask", c_file_name)
            c_lung_file = join(data_folder, "lung_mask", c_file_name)
        else:
            c_file_name = c_file_name.replace("org_","")
            c_mask_file = join(data_folder, "infection_mask", c_file_name)
            c_lung_file = join(data_folder, "lung_mask", c_file_name)

        itk_img = sitk.ReadImage(c_img_file)
        itk_mask = sitk.ReadImage(c_mask_file)
        itk_lung = sitk.ReadImage(c_lung_file)
        np_mask = sitk.GetArrayFromImage(itk_mask)
        np_lung = sitk.GetArrayFromImage(itk_lung)

        printImageSummary(itk_img)

        viz_obj.plot_img_and_ctrs_itk(itk_img, [itk_mask, itk_lung],
                                      slices=SliceMode.MIDDLE,
                                      title=c_file_name, draw_only_ctrs=True,
                                      labels=["Infection", "Lungs"],
                                      file_name_prefix=id)

    print("Done!")

if __name__ == '__main__':
    _disp_imgs = False
    # vizMosmedData()
    # vizMedSeg1()
    # vizMedSeg2()
    vizRadiopaedia()


