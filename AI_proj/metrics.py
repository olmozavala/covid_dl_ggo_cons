from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as k

k.set_image_data_format('channels_last')


def real_dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = k.flatten(y_true)
    # y_true_f = tf.minimum(K.flatten(y_true), 1)
    y_pred_f = tf.minimum(k.flatten(y_pred), 1)
    intersection = k.sum(y_true_f * y_pred_f)
    # return (2. * intersection + smooth) / ( K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
    # The squared is not necessary when the GT is 0 or 1
    return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


def real_dice_coef_lesion(y_true, y_pred, smooth=1.0):
    # Flatten all the arrays
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)

    # Take into account only the kidney area
    # Multiply by 0 everything outside the kidney
    temp = y_true_f / 2.0  # temp have 1 for lesion and .5 for lungs
    y_pred_c = y_pred_f * tf.ceil(temp - .001)  # Make 0 the prediction outside the lungs
    y_pred_c = tf.minimum(k.flatten(y_pred_c), 1)  # Make 1 everything > 1 (for non-sigmoid networks)
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + .001)
    intersection = k.sum(y_true_c * y_pred_c)
    return (2. * intersection + smooth) / (k.sum(y_true_c) + k.sum(y_pred_c) + smooth)


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    # y_pred_f = tf.minimum(K.flatten(y_pred), 1)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


def dice_coef_lesion(y_true, y_pred, smooth=1.0):
    eps = .001
    # Flatten all the arrays
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)

    # y_true_f MUST have 2 for lesion and 1 for kidney
    temp = y_true_f / 2.0  # temp have 1 for lesion and .5 for kidney

    y_pred_c = y_pred_f * tf.ceil(temp - eps)  # Make 0 the prediction outside the kidney
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + eps)
    intersection = k.sum(y_true_c * y_pred_c)
    return (2. * intersection + smooth) / (k.sum(y_true_c) + k.sum(y_pred_c) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def dice_coef_lesion_loss(y_true, y_pred):
    return -dice_coef_lesion(y_true, y_pred)


def numpy_dice(y_true, y_pred, smooth=1.0):
    intersection = y_true.flatten() * y_pred.flatten()

    return (2. * intersection.sum() + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def numpy_dice_copy(y_true, y_pred):
    smooth = 1.0
    y_true_f = y_true.flatten()
    y_pred_f = np.minimum(y_pred.flatten(), 1)
    intersection = y_true_f * y_pred_f
    return (2. * intersection.sum() + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def mse(y_true, y_pred, smooth=1.0):
    eps = .001
    # Flatten all the arrays
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)

    # y_true_f MUST have 2 for lesion and 1 for kidney
    temp = y_true_f / 2.0  # temp have 1 for lesion and .5 for kidney
    y_pred_c = y_pred_f * tf.ceil(temp - eps)  # Make 0 the prediction outside the kidney
    # Keep only the lesion with values of 1
    y_true_c = tf.floor(temp + eps)
    return tf.squared_difference(y_pred_c, y_true_c)


# ============= LAST ATTEMPT ============

def dice_coef_combined(y_true, y_pred, smooth=1.0):
    # y_true_f MUST have 2 for lesion and 1 for kidney
    eps = .01
    # Flatten all the arrays
    y_true_both = k.flatten(y_true)
    y_pred_k = k.flatten(y_pred[:,:,:,:,0])
    # y_pred_t = k.flatten(y_pred[:,:,:,:,1])

    # ---------- Kidney -----------
    temp = y_true_both / 2.0  # temp have 1 for lesion and .5 for kidney
    y_true_kidney = tf.ceil(temp - eps)  # 0 outside kidney 1 in kidney
    intersection = k.sum(y_true_kidney * y_pred_k)
    dsc_kidney = (2. * intersection + smooth) / (k.sum(y_true_kidney) + k.sum(y_pred_k) + smooth)
    return dsc_kidney

    # # ---------- Tumor -----------
    # y_true_tumor = tf.floor(temp + eps)  # 0 outside kidney 1 in kidney
    # intersection = k.sum(y_true_tumor * y_pred_t)
    # dsc_tumor = (2. * intersection + smooth) / (k.sum(y_true_tumor) + k.sum(y_pred_t) + smooth)
    #
    # return (dsc_kidney + dsc_tumor)/2

def dice_coef_combined_loss(y_true, y_pred):
    return -dice_coef_combined(y_true, y_pred)
