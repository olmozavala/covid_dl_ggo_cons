from tensorflow.keras.optimizers import *
from os.path import join
import os

from AI_proj.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

# ===============================================================================================
# ==================================== 3D =======================================================
# ===============================================================================================

def append_model_params_3d(cur_config):
    model_config = {
        # ModelParams.MODEL: AiModels.UNET_3D_3_STREAMS,
        ModelParams.MODEL: AiModels.UNET_3D_SINGLE,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [80, 320, 320],
        ModelParams.START_NUM_FILTERS: 8,
        ModelParams.NUMBER_LEVELS: 3,
        ModelParams.FILTER_SIZE: 3
    }
    return {**cur_config, **model_config}


def get_training_3d():
    _data_folder = '/data/UM/COVID/PREPROC/3D/data'  # Where the data is stored and where the preproc folder will be saved
    _run_name = F'COVID_SingleStream_3DUNet'  # Name of the model, for training and classification
    _output_folder = '/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT'
    cur_config = {
        TrainingParams.input_folder: _data_folder,
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.image_file_names: ['img.nrrd'],
        TrainingParams.ctr_file_names: ['ctr.nrrd'],
        TrainingParams.evaluation_metrics: [dice_coef_lesion_loss, real_dice_coef_lesion],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: dice_coef_lesion_loss,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 2,
        TrainingParams.epochs: 1000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: True
    }
    return append_model_params_3d(cur_config)


def get_segmentation_3d_config():
    _data_folder = '/data/UM/COVID/PREPROC/3D/data'  # Where the data is stored and where the preproc folder will be saved
    _run_name = F'COVID_SingleStream_3DUNet'  # Name of the model, for training and classification
    _output_folder = '/data/UM/COVID/PREPROC/3D/output'  # Where to save the models
    cur_config = {
        ClassificationParams.input_folder: _data_folder,
        ClassificationParams.output_folder: F"{join(_output_folder, 'Segmentation', _run_name)}",
        ClassificationParams.output_imgs_folder: F"{join(_output_folder, 'Segmentation_Images', _run_name)}",
        ClassificationParams.cases: 'all',
        ClassificationParams.model_weights_file: '/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_3DUNet/models/COVID_SingleStream_3DUNet_2020_09_09_19_01-28--0.56671.hdf5',
        ClassificationParams.save_segmented_ctrs: True,
        ClassificationParams.output_file_name: 'Classification_DSC.csv',
        ClassificationParams.input_img_file_names: ['img.nrrd'],
        ClassificationParams.output_ctr_file_names: ['ctr.nrrd'],
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
    return append_model_params_3d(cur_config)

# ===============================================================================================
# ==================================== 2D =======================================================
# ===============================================================================================

def append_model_params_2d(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.UNET_2D_SINGLE,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [320, 320],
        ModelParams.START_NUM_FILTERS: 16,
        ModelParams.NUMBER_LEVELS: 3,
        ModelParams.FILTER_SIZE: 3
    }
    return {**cur_config, **model_config}


def get_training_2d():
    _data_folder = '/data/UM/COVID/PREPROC/2D/data'  # Where the data is stored and where the preproc folder will be saved
    _run_name = F'COVID_SingleStream_2DUNet'  # Name of the model, for training and classification
    _output_folder = '/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/'  # Where to save the models
    cur_config = {
        TrainingParams.input_folder: _data_folder,
        TrainingParams.output_folder: F"{join(_output_folder,'Training', _run_name)}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.image_file_names: ['img.png'],
        TrainingParams.ctr_file_names: ['ctr_lung.png','ctr_lesion.png'],
        TrainingParams.evaluation_metrics: [dice_coef_lesion_loss, real_dice_coef_lesion],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: dice_coef_lesion_loss,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 20, # Number of samples per gradient update
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: True
    }
    return append_model_params_2d(cur_config)

def get_segmentation_2d_config():
    _data_folder = '/data/UM/COVID/PREPROC/2D/data'  # Where the data is stored and where the preproc folder will be saved
    _run_name = F'COVID_SingleStream_2DUNet'  # Name of the model, for training and classification
    _output_folder = '/data/UM/COVID/PREPROC/2D/output'  # Where to save the models
    cur_config = {
        ClassificationParams.input_folder: _data_folder,
        ClassificationParams.output_folder: F"{join(_output_folder, 'Segmentation', _run_name)}",
        ClassificationParams.output_imgs_folder: F"{join(_output_folder, 'Segmentation_Images', _run_name)}",
        ClassificationParams.cases: 'all',
        ClassificationParams.model_weights_file: '/home/olmozavala/Dropbox/MyProjects/UM/Covid19_Prognosis/COVID_DL_Segmentation/OUTPUT/Training/COVID_SingleStream_2DUNet/models/COVID_SingleStream_2DUNet_2020_09_10_18_53-131--0.81194.hdf5',
        ClassificationParams.save_segmented_ctrs: True,
        ClassificationParams.output_file_name: 'Classification_DSC.csv',
        ClassificationParams.input_img_file_names: ['img.png'],
        ClassificationParams.output_ctr_file_names: ['ctr_lung.png', 'ctr_lesion.png'],
        # *********** These configurations depend on computing metrics ********
        ClassificationParams.metrics: [ClassificationMetrics.DSC_2D],
        # *********** These configurations depend ont saving the images ********
        ClassificationParams.save_imgs: True,
        ClassificationParams.show_imgs: False,
    }
    return append_model_params_2d(cur_config)