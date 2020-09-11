import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *
from preproc.constants import NormParams
from pandas import DataFrame

from AI_proj.metrics import dice_coef_loss, real_dice_coef
from os.path import join
import numpy as np

def getAllCallbacksLesion(model_name, early_stopping_func, weights_folder, logs_folder):
    logdir = "{}/run-{}/".format(logs_folder, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.2f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model,monitor=early_stopping_func, save_best_only=True,
                                              mode='max',save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping_func, min_delta=.001, patience=150, mode='max')
    return [logger, save_callback, stop_callback]


def get_all_callbacks(model_name, early_stopping_func, weights_folder, logs_folder):
    logdir = "{}/run-{}/".format(logs_folder, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.5f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model, monitor=early_stopping_func,
                                              save_best_only=True,
                                              save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping_func, min_delta=.0001, patience=200, mode='min')
    return [logger, save_callback, stop_callback]


def runAndSaveModel(model, X, Y, model_name, epochs, is3d, val_per, early_stopping_func) :

    # plot_model(model, to_file='../images/{}.png'.format(model_name), show_shapes=True)

    [logger, save_callback, stop_callback] = get_all_callbacks(model_name, early_stopping_func)

    if is3d:
        model.fit(
            x = X,
            y = Y,
            epochs=epochs,
            shuffle=True,
            batch_size=1,
            verbose=2,
            validation_split=val_per,
            callbacks=[logger, save_callback, stop_callback]
        )
    else:
        model.fit(
            x = X,
            y = Y,
            epochs=epochs,
            shuffle=True,
            batch_size=4,
            verbose=2,
            validation_split=val_per,
            callbacks=[logger]
        )


def split_train_and_test(num_examples, test_percentage):
    """
    Splits a number into training and test randomly
    :param num_examples: int of the number of examples
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples * test_percentage))
    # Train and validation indexes
    train_val_idx = all_samples_idx[0:len(all_samples_idx) - test_examples]
    test_idx = all_samples_idx[len(all_samples_idx) - test_examples:len(all_samples_idx)]

    return [train_val_idx, test_idx]


def split_train_validation_and_test(num_examples, val_percentage, test_percentage):
    """
    Splits a number into training, validation, and test randomly
    :param num_examples: int of the number of examples
    :param val_percentage: int of the percentage desired for validation
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idx = np.arange(num_examples)
    np.random.shuffle(all_samples_idx)
    test_examples = int(np.ceil(num_examples * test_percentage))
    val_examples = int(np.ceil(num_examples * val_percentage))
    # Train and validation indexes
    train_idx = all_samples_idx[0:len(all_samples_idx) - test_examples - val_examples]
    val_idx = all_samples_idx[len(all_samples_idx) - test_examples - val_examples:len(all_samples_idx) - test_examples]
    test_idx = all_samples_idx[len(all_samples_idx) - test_examples:]
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    return [train_idx, val_idx, test_idx]


def save_splits(file_name, folders_to_read, train_idx, val_idx, test_idx):
    print("Saving splits....")
    file = open(file_name,'w')
    file.write("\n\nTrain examples (total:{}) :{}".format(len(train_idx), folders_to_read[train_idx]))
    file.write("\n\nValidation examples (total:{}) :{}:".format(len(val_idx), folders_to_read[val_idx]))
    file.write("\n\nTest examples (total:{}) :{}".format(len(test_idx), folders_to_read[test_idx]))
    file.close()

def save_norm_params(file_name, norm_type, scaler):
    print("Saving normalization parameters....")

    if norm_type == NormParams.min_max:
        file = open(file_name, 'w')
        min_val = scaler.data_min_
        max_val = scaler.data_max_
        scale = scaler.scale_
        range = scaler.data_range_

        file.write(F"Normalization type: {norm_type}, min: {min_val}, max: {max_val}, range: {range}, scale: {scale}")
        file.close()
    else:
        print(F"WARNING! The normalization type {norm_type} is unknown!")


