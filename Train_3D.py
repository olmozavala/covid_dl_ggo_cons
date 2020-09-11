from datetime import datetime

from config.MainConfig import get_training_3d
from AI_proj.data_generation.Generators3D import *

from inout.io_common import create_folder, select_cases_from_folder

from constants.AI_params import *
from models.modelSelector import select_3d_model
import AI_proj.trainingutils as utilsNN

from tensorflow.keras.utils import plot_model
import tensorflow as tf
from os.path import join

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

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

    nn_input_size = config[ModelParams.INPUT_SIZE]
    model_type = config[ModelParams.MODEL]

    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    folders_to_read = select_cases_from_folder(input_folder, config[TrainingParams.cases])
    tot_examples = len(folders_to_read)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    print("Train examples (total:{}) :{}".format(len(train_ids), folders_to_read[train_ids]))
    print("Validation examples (total:{}) :{}:".format(len(val_ids), folders_to_read[val_ids]))
    print("Test examples (total:{}) :{}".format(len(test_ids), folders_to_read[test_ids]))

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{now}'

    # ******************* Selecting the model **********************
    model = select_3d_model(config)
    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, folders_to_read=folders_to_read,
                        train_idx=train_ids, val_idx=val_ids, test_idx=test_ids)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Training ...")
    my_generator = Generator3D()

    # Decide which generator to use
    batch_size = config[TrainingParams.batch_size]
    data_augmentation = config[TrainingParams.data_augmentation]
    if (model_type == AiModels.UNET_3D_3_STREAMS) or (model_type == AiModels.UNET_3D_SINGLE):
        train_generator = my_generator.unet_3d_single_stream(input_folder=input_folder,
                                                             folders_to_read=folders_to_read[train_ids],
                                                             stream_file_names=img_names,
                                                             ctr_file_name=ctr_names[0], # Single contour
                                                             data_augmentation=data_augmentation,
                                                             batch_size=batch_size)

        val_generator = my_generator.unet_3d_single_stream(input_folder=input_folder,
                                                           folders_to_read=folders_to_read[val_ids],
                                                           stream_file_names=img_names,
                                                           ctr_file_name=ctr_names[0], # Single contour
                                                           data_augmentation=data_augmentation,
                                                           batch_size=batch_size)

    model.fit_generator(train_generator, steps_per_epoch=min(100, len(train_ids)),
                        validation_data=val_generator,
                        validation_steps=min(20, len(val_ids)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])
