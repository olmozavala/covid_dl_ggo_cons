from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os

from AI_proj.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
_output_folder = '/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Template/OUTPUT'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.DENSE_CNN,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: False,
        ModelParams.INPUT_SIZE: [1200, 1920],
        ModelParams.START_NUM_FILTERS: 1,
        ModelParams.FILTER_SIZE: 3,
        ModelParams.NUMBER_DENSE_LAYERS: 1, # In this case are 'DENSE' CNN
    }
    return {**cur_config, **model_config}


def get_model_viz_config():
    cur_config = {
        ClassificationParams.model_weights_file: '/home/olmozavala/Dropbox/TutorialsByMe/TensorFlow/Examples/TestBedCNNs/OUTPUT/Training/sobel_from_mnist/models/sobel_from_mnist_2020_03_05_20_54-61-0.00000.hdf5',
        TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
    }
    return append_model_params(cur_config)

