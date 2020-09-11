import os
from pandas import DataFrame
import pandas as pd
import time
from constants.AI_params import *
from os.path import join
import SimpleITK as sitk

from img_viz.medical import MedicalImageVisualizer
from config.MainConfig import get_segmentation_3d_config
from inout.io_common import create_folder, select_cases_from_folder
from preproc.UtilsItk import copyItkImage
from preproc.UtilsPreproc import binaryThresholdImage, getLargestConnectedComponents
from inout.readDataPreproc import read_preproc_imgs_and_ctrs_itk
from AI_proj.data_generation.utilsDataFormat import *
from models.modelSelector import select_3d_model
from img_viz.constants import SliceMode, PlaneTypes
from AI_proj.metrics import numpy_dice

ORIGINAL_TXT = 'Original'
def main():
    config = get_segmentation_3d_config()
    cases = config[ClassificationParams.cases]
    save_segmented_ctrs = config[ClassificationParams.save_segmented_ctrs]

    input_folder = config[ClassificationParams.input_folder]
    input_img_names = config[ClassificationParams.input_img_file_names]
    output_folder = config[ClassificationParams.output_folder]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]
    output_file_name = config[ClassificationParams.output_file_name]
    model_weights_file = config[ClassificationParams.model_weights_file]
    compute_metrics = config[ClassificationParams.compute_metrics]

    save_imgs = config[ClassificationParams.save_imgs]
    if save_imgs:
        save_imgs_planes = config[ClassificationParams.save_img_planes]
        save_imgs_slices = config[ClassificationParams.save_img_slices]

    # Builds the visualization object
    viz_obj = MedicalImageVisualizer(disp_images=config[ClassificationParams.show_imgs],
                                     output_folder=output_imgs_folder)

    if compute_metrics:
        output_ctr_file_names = config[ClassificationParams.output_ctr_file_names]
    else:
        output_ctr_file_names = []
    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_3d_model(config)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    examples = select_cases_from_folder(input_folder, cases)
    create_folder(output_imgs_folder)

    # *********** Makes a dataframe to contain the DSC information **********
    metrics_params = config[ClassificationParams.metrics]
    metrics_dict = {met.name: met.value for met in metrics_params}

    # Check if the output files already exist, in that case read the df from it.
    if os.path.exists(join(output_imgs_folder, output_file_name)):
        data = pd.read_csv(join(output_imgs_folder, output_file_name), index_col=0)
    else:
        data_columns = list(metrics_dict.values())
        data = DataFrame(index=examples, columns=data_columns)

    # *********** Iterates over each case *********
    for id_folder, current_folder in enumerate(examples):
        print(F'******* Computing folder {current_folder} ************')
        t0 = time.time()
        try:
            # -------------------- Reading data -------------
            print('\t Reading data....')
            # All these names are predefined, for any other 3d segmentation we will need to create a different configuration
            imgs_itk, ctrs_itk, size_roi, start_roi, _ = read_preproc_imgs_and_ctrs_itk(input_folder,
                                                                                      folders_to_read=[current_folder],
                                                                                      img_names=input_img_names,
                                                                                      ctr_names=output_ctr_file_names)

            imgs_np = [sitk.GetArrayFromImage(c_img) for c_img in imgs_itk[0]] # The 0 is because we read a single fold
            ctrs_np = [sitk.GetArrayFromImage(c_img) for c_img in ctrs_itk[0]]
            ctrs_lesion_np = ctrs_np[0].copy()  # We only have one contour in our system
            ctrs_lungs_np = ctrs_np[0].copy()  # We only have one contour in our system
            ctrs_lungs_np[ctrs_lungs_np > 1] = 1  # Settings lungs to 1
            ctrs_lesion_np[ctrs_lesion_np == 1] = 0  # Setting lungs to 0
            ctrs_lesion_np[ctrs_lesion_np == 2] = 1  # Setting lesions = 1
            ctr_lung_itk = copyItkImage(imgs_itk[0][0], ctrs_lesion_np)

            # If we want to visualize the input images
            # viz_obj.plot_imgs_and_ctrs_itk(imgs_itk[0], ctrs_itk=ctrs_itk[0])

            # ------------------- Making prediction -----------
            print('\t Making prediction....')
            input_array = format_for_nn_classification(imgs_np)
            output_nn_all = model.predict(input_array, verbose=1)
            output_nn_np = output_nn_all[0,:,:,:,0]
            output_nn_np[ctrs_lungs_np == 0] = 0  # Making the prediction 0 outside the lungs
            # For visualizing the output of the network
            # viz_obj.plot_img_and_ctrs_np(output_nn_np, np_ctrs=[], file_name_prefix=F"NN_Raw_{id_folder}", slices=SliceMode.MIDDLE)

            # ------------------- Postprocessing -----------
            print('\t Postprocessing prediction....')
            threshold = .5
            output_nn_itk = copyItkImage(imgs_itk[0][0], output_nn_np)
            print(F'\t\t Threshold NN output to {threshold} ....')
            output_nn_itk = binaryThresholdImage(output_nn_itk, threshold)

            if save_segmented_ctrs:
                print('\t Saving Prediction...')
                create_folder(join(output_folder, current_folder))
                # TODO at some point we will need to see if we can output more than one ctr
                sitk.WriteImage(output_nn_itk, join(output_folder, current_folder, output_ctr_file_names[0]))

            if compute_metrics:
                # Compute metrics
                print('\t Computing metrics....')
                for c_metric in metrics_params:  # Here we can add more metrics
                    if c_metric == ClassificationMetrics.DSC_3D:
                        metric_value = numpy_dice(output_nn_np, ctrs_lesion_np)
                        data.loc[current_folder][c_metric.value] = metric_value
                        print(F'\t\t ----- DSC: {metric_value:.3f} -----')

                # Saving the results every 10 steps
                if id_folder % 10 == 0:
                    save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
                    data.to_csv(join(output_folder, output_file_name))

            if save_imgs:
                print('\t Plotting images....')
                plot_intermediate_results(current_folder, data_columns, imgs_itk=imgs_itk[0],
                                          gt_ctr_itk=ctr_lung_itk, nn_ctr_itk=output_nn_itk, data=data,
                                          viz_obj=viz_obj, slices=save_imgs_slices, compute_metrics=compute_metrics)


        except Exception as e:
            print("---------------------------- Failed {} error: {} ----------------".format(current_folder, e))
        print(F'\t Done! Elapsed time {time.time()-t0:0.2f} seg')

    if compute_metrics:
        save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
        data.to_csv(join(output_folder, output_file_name))


def plot_intermediate_results(current_folder, metric_names, imgs_itk, gt_ctr_itk, nn_ctr_itk,
                              data, viz_obj, slices, compute_metrics=True, prefix_name=''):
    """
    This function is in charge of plotting the segmentation as images.
    :param current_folder:
    :param metric_names:
    :param img_itk:
    :param gt_ctr_itk:
    :param nn_ctr_itk:
    :param data:
    :param viz_obj:
    :param slices:
    :param compute_metrics:
    :return:
    """
    title = F'{current_folder}  '
    if compute_metrics:
        for c_metric_name in metric_names:  # Here we can add more metrics
            title += F'{c_metric_name}: {data.loc[current_folder][c_metric_name]:.3f}   '

    file_name = F'{prefix_name}_{current_folder}' if prefix_name != '' else F'{current_folder}'

    viz_obj.plot_imgs_and_ctrs_itk(itk_imgs=imgs_itk, itk_ctrs=[gt_ctr_itk, nn_ctr_itk],
                                   slices=slices, file_name_prefix=file_name,
                                   title=title,
                                   labels=['GT', 'NN'])


def save_metrics_images(data, metric_names, viz_obj: MedicalImageVisualizer):
    # ************** Plot and save Metrics for ROI *****************
    data.loc['AVG'] = data.mean() # Compute all average values
    title = ''
    input_dics = []
    for c_metric in metric_names:
        title += F"{c_metric}: {data.loc['AVG'][c_metric]:.3f}"
        input_dics.append(data[c_metric].dropna().to_dict())

    viz_obj.plot_multiple_bar_plots(input_dics,
                                    title=title,
                                    legends=metric_names,
                                    file_name='Mean_Performance.png')
if __name__ == '__main__':
    main()
