from os.path import join
from constants.AI_params import *
import os
import SimpleITK as sitk
from img_viz.medical import *
from img_viz.constants import SliceMode, PlaneTypes
from proj_inout.utils import printImageSummary, utilsSplitMask
from Preproc.UtilsPreproc import normalize_to_percentiles, resample_imgs_and_ctrs, resample_img_itk, resample_to_reference_itk, crop_to_specific_dimensions_from_center
from inout.io_common import save_image
from config.MainConfig import get_training_3d
from preproc.UtilsItk import copyItkImage
import cv2

def readMosmed(all_imgs, all_lesion_ctrs, all_lung_ctrs):
    # Visualizing Mosmed data
    print("Reading mosmed...")
    data_folder = "/data/UM/COVID/Kaggle_Mosmed"
    mask_files = os.listdir(join(data_folder, "masks"))
    mask_files.sort()
    for count, c_mask_file_name in enumerate(mask_files):
        c_case = int(c_mask_file_name.split("_")[1])
        c_mask_file = join(data_folder, "masks", c_mask_file_name)
        c_img_file_name = F"study_0{c_case:03d}.nii"
        c_img_file = join(data_folder, "studies", c_img_file_name)
        c_lung_mask_file = join(data_folder, "lung_masks", c_mask_file_name)

        itk_mask = sitk.ReadImage(c_mask_file)
        itk_img = sitk.ReadImage(c_img_file)
        itk_lung_mask = sitk.ReadImage(c_lung_mask_file)
        all_imgs.append(itk_img)
        all_lesion_ctrs.append(itk_mask)
        all_lung_ctrs.append(itk_lung_mask)

    print("Done!")
    return all_imgs, all_lesion_ctrs, all_lung_ctrs

def readMedSeg2(all_imgs, all_lesion_ctrs, all_lung_ctrs):
    data_folder = "/data/UM/COVID/Medicalsegmentation_V2"

    for c_id in range(1,10):
        c_mask_file = join(data_folder, "rp_msk", F"{c_id}.nii")
        c_img_file = join(data_folder, "rp_im", F"{c_id}.nii")
        c_lung_file = join(data_folder, "rp_lung_msk", F"{c_id}.nii")
        itk_img = sitk.ReadImage(c_img_file)
        itk_mask = sitk.ReadImage(c_mask_file)
        itk_lung = sitk.ReadImage(c_lung_file)

        np_mask = sitk.GetArrayFromImage(itk_mask)
        # np_mask[np_mask == 3] = 0
        np_mask[np_mask >= 1] = 1

        all_imgs.append(itk_img)
        all_lesion_ctrs.append(copyItkImage(itk_img, np_mask))
        all_lung_ctrs.append(itk_lung)

    print("Done!")
    return all_imgs, all_lesion_ctrs, all_lung_ctrs

def readRadiopaedia(all_imgs, all_lesion_ctrs, all_lung_ctrs):
    data_folder = "/data/UM/COVID/Kaggle_1_Radiopedia"

    file_names = listdir(join(data_folder,"ct_scans"))
    file_names.sort()
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
        np_mask[np_mask >= .01] = 1 # Not sure why but it has decimals

        all_imgs.append(itk_img)
        all_lesion_ctrs.append(copyItkImage(itk_img, np_mask))
        all_lung_ctrs.append(itk_lung)

    return all_imgs, all_lesion_ctrs, all_lung_ctrs
    print("Done!")

def debugShowImages(imgs, proc_imgs, ctrs, title, viz_obj, idxs):
    for id, c_img in enumerate(imgs):
        if id in idxs:
            c_ctr = ctrs[id]
            comp_img = proc_imgs[id]
            print(F"Size for {id} are orig: {c_img.GetSize()} vs proc: {comp_img.GetSize()}")
            print(F"Spacing for {id} are orig: {c_img.GetSpacing()} vs proc: {comp_img.GetSpacing()}")
            viz_obj.plot_imgs_and_ctrs_itk([c_img, comp_img], [c_ctr], title=title,
                                       draw_only_ctrs=False, slices=SliceMode.MIDDLE,
                                       file_name_prefix=F"{id}_{title}")



if __name__ == '__main__':
    config = get_training_3d()

    input_folder = config[TrainingParams.input_folder]
    output_folder = config[TrainingParams.output_folder]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    img_names = config[TrainingParams.image_file_names]
    model_name_user = config[TrainingParams.config_name]
    ctr_names = config[TrainingParams.ctr_file_names]
    optimizer = config[TrainingParams.optimizer]

    output_folder_3d = "/data/UM/COVID/PREPROC/3D/data"
    output_folder_2d = "/data/UM/COVID/PREPROC/2D/data"
    viz_obj = MedicalImageVisualizer(output_folder_3d="/data/UM/COVID/PREPROC/3D/output_imgs", disp_images=False)

    resampling = [.9, .9, 4.5]
    roi_dims = [320, 320, 80]

    case_start_id = 1
    all_imgs = []
    all_lesion_ctrs = [] # Lesionevs
    all_lung_ctrs = [] # Lesions
    print("------------ Reading images --------------")
    print("--- Mosmed ---")
    all_imgs, all_lesion_ctrs, all_lung_ctrs = readMosmed(all_imgs, all_lesion_ctrs, all_lung_ctrs)
    print("--- MedSeg2 ---")
    all_imgs, all_lesion_ctrs, all_lung_ctrs = readMedSeg2(all_imgs, all_lesion_ctrs, all_lung_ctrs)
    print("--- Radiopaedia ---")
    all_imgs, all_lesion_ctrs, all_lung_ctrs = readRadiopaedia(all_imgs, all_lesion_ctrs, all_lung_ctrs)

    id_2d = 0
    for id, c_img in enumerate(all_imgs):
        print(F"=============== Working with id: {id} ========================")
        print("------------ Normalizing image ----------")
        c_img_norm = normalize_to_percentiles([c_img])[0]
        print("Done!")
        # Just for debugging
        # debugShowImages([all_imgs[id]], [c_img_norm], [all_lesion_ctrs[id], all_lung_ctrs[id]], "original_vs_norm", viz_obj, [0])

        print("------------ Resampling image --------------")
        c_imgs_res, c_ctrs_res = resample_imgs_and_ctrs([c_img_norm], [all_lesion_ctrs[id], all_lung_ctrs[id]], resampling)
        # np_mask = sitk.GetArrayFromImage(c_ctrs_res[0])
        # midlay = int(np_mask.shape[0]/2)
        # plt.imshow(np_mask[midlay, :, :])
        # plt.show()
        print("Done!")

        print("------------ Cropping image --------------")
        c_img_crop_np, cropped_ctrs_np, pos_res, pos_crop, new_origin = crop_to_specific_dimensions_from_center(c_imgs_res[0],  c_ctrs_res, roi_dims, img_center_of_mass=False)
        c_img_crop = sitk.GetImageFromArray(c_img_crop_np)
        # np_mask = cropped_ctrs_np[0]
        # midlay = int(np_mask.shape[0]/2)
        # plt.imshow(np_mask[midlay, :, :])
        # plt.show()

        # Setting the proper metadata to the image
        c_img_crop.SetOrigin(new_origin)
        c_img_crop.SetDirection(c_imgs_res[0].GetDirection())
        c_img_crop.SetSpacing(c_imgs_res[0].GetSpacing())

        # Copying metadata to lesion contour and setting its value to 1
        c_ctr_crop_np = cropped_ctrs_np[0]
        # Making final correction of decimal values
        c_ctr_crop_np[c_ctr_crop_np > .001] = 1

        tmp_ctr_lesion_np = c_ctr_crop_np.copy() # Just a temporal variable used to save the lungs separated from the lesion in 2D
        tmp_lung_ctr_crop_np = cropped_ctrs_np[1]
        tmp_lung_ctr_crop_np[tmp_lung_ctr_crop_np > .001] = 1 # Setiting the two lungs to a value of 1

        c_ctr_crop_np += tmp_lung_ctr_crop_np
        c_ctr_crop = copyItkImage(c_img_crop, c_ctr_crop_np)
        # np_mask = tmp_lung_ctr_crop_np
        # midlay = int(np_mask.shape[0]/2)
        # plt.imshow(np_mask[midlay, :, :])
        # plt.show()
        print("Done!")

        print("------------ Saving images 3D --------------")
        # save_image(c_img_crop, join(output_folder_3d, F"Case-{id:04d}"), "img.nrrd")
        # save_image(c_ctr_crop, join(output_folder_3d, F"Case-{id:04d}"), "ctr.nrrd")

        print("------------ Saving images 2D --------------")
        tot_slices = c_img_crop_np.shape[0]
        for c_slice in range(tot_slices):
            # Only saving slices where there is a contour
            if np.amax(c_ctr_crop_np[c_slice, :, :]) > 1:
                n_folder =join(output_folder_2d,F"Case-{id_2d:04d}")
                if not os.path.exists(n_folder):
                    os.makedirs(n_folder)
                cv2.imwrite(join(n_folder, "img.png"), cv2.convertScaleAbs(c_img_crop_np[c_slice, :, :], alpha=(255.0)))
                cv2.imwrite(join(n_folder, "ctr_lesion.png"), cv2.convertScaleAbs(tmp_ctr_lesion_np[c_slice, :, :], alpha=(255.0)))
                cv2.imwrite(join(n_folder, "ctr_lung.png"), cv2.convertScaleAbs(tmp_lung_ctr_crop_np[c_slice, :, :], alpha=(255.0)))
                # For debugging
                # n_folder =join(output_folder_2d) # cv2.imwrite(join(n_folder, F"{id_2d:04d}_img.png"), cv2.convertScaleAbs(c_img_crop_np[c_slice, :, :], alpha=(255.0)))
                # cv2.imwrite(join(n_folder, F"{id_2d:04d}_ctr.png"), cv2.convertScaleAbs(c_ctr_crop_np[c_slice, :, :], alpha=(255.0)))
                id_2d += 1
