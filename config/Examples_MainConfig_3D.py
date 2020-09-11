from tensorflow.keras.optimizers import *
from os.path import join
import os

from AI_proj.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
_data_folder = '/media/osz1/DATA/DATA/PX/'  # Where the data is stored and where the preproc folder will be saved
_preproc_folder = 'Preproc'  # Name to save preprocessed data
_run_name = F'Prostate_MultiStream_{_preproc_folder}'  # Name of the model, for training and classification
_output_folder = '/media/osz1/DATA/DATA/DELETE'  # Where to save the models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        # ModelParams.MODEL: AiModels.UNET_3D_3_STREAMS,
        ModelParams.MODEL: AiModels.UNET_3D_SINGLE,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [168, 168, 168],
        ModelParams.START_NUM_FILTERS: 8,
        ModelParams.NUMBER_LEVELS: 3,
        ModelParams.FILTER_SIZE: 3
    }
    return {**cur_config, **model_config}


def get_training_3d():
    cur_config = {
        TrainingParams.input_folder: F'{join(_data_folder,_preproc_folder)}',
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        # TrainingParams.image_file_names: ['roi_tra.nrrd', 'roi_sag.nrrd', 'roi_cor.nrrd'],
        TrainingParams.image_file_names: ['roi_tra.nrrd'],
        TrainingParams.ctr_file_names: ['roi_ctr_pro.nrrd'],
        TrainingParams.evaluation_metrics: [real_dice_coef],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: dice_coef_loss,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 2,
        TrainingParams.epochs: 1000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False
    }
    return append_model_params(cur_config)


def get_segmentation_3d_config():
    cur_config = {
        ClassificationParams.input_folder: F'{join(_data_folder, _preproc_folder)}',
        ClassificationParams.output_folder: F"{join(_output_folder, 'Segmentation', _run_name)}",
        ClassificationParams.output_imgs_folder: F"{join(_output_folder, 'Segmentation_Images', _run_name)}",
        ClassificationParams.cases: 'all',
        ClassificationParams.model_weights_file: '/media/osz1/DATA/DATA/DELETE/Training/Prostate_MultiStream_Preproc/models/Prostate_MultiStream_Preproc_2019_08_21_20_01-11--0.83.hdf5',
        ClassificationParams.save_segmented_ctrs: True,
        ClassificationParams.output_file_name: 'Classification_DSC.csv',
        ClassificationParams.segmentation_type: SegmentationTypes.PROSTATE,
        ClassificationParams.input_img_file_names: ['roi_tra.nrrd', 'roi_sag.nrrd', 'roi_cor.nrrd'],
        ClassificationParams.output_ctr_file_names: ['roi_ctr_pro.nrrd'],
        # *********** These configurations depend on computing the original resolution ********
        ClassificationParams.compute_original_resolution: True,
        ClassificationParams.resampled_resolution_image_name: 'hr_tra.nrrd',
        ClassificationParams.original_resolution_image_name: 'img_tra.nrrd',
        ClassificationParams.original_resolution_ctr_name: 'ctr_pro.nrrd',
        # *********** These configurations depend on computing metrics ********
        ClassificationParams.compute_metrics: True,
        ClassificationParams.metrics: [ClassificationMetrics.DSC_3D],
        # *********** These configurations depend ont saving the images ********
        ClassificationParams.save_imgs: True,
        ClassificationParams.show_imgs: False,
        ClassificationParams.save_img_planes: PlaneTypes.ALL,
        ClassificationParams.save_img_slices: SliceMode.MIDDLE
        # ClassificationParams.save_img_slices: range(70,90,2)
    }
    return append_model_params(cur_config)
