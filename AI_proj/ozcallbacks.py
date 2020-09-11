from os.path import join

import keras.callbacks as callbacks


def get_all_callbacks(model_name, early_stopping, weights_folder='.', logs_folder='.', min_delta=.001, patience=70):
    root_log_dir = join(logs_folder, 'logs')
    log_dir = "{}/run-{}/".format(root_log_dir, model_name)

    logger = callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    file_path_model = join(weights_folder, model_name + '-{epoch:02d}-{val_loss:.2f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(file_path_model, monitor=early_stopping, save_best_only=True,
                                              mode='max', save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping, min_delta=min_delta, patience=patience, mode='max')
    return [logger, save_callback, stop_callback]
