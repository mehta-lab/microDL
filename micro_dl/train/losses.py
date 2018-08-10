"""Custom losses"""
from keras import backend as K
import tensorflow as tf

import micro_dl.train.metrics as metrics
import keras.losses as keras_loss

def mse_binary_wtd(n_channels):
    """Masked loss function

    https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    https://github.com/keras-team/keras/issues/3270
    https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred

    nested functions -> closures
    A Closure is a function object that remembers values in enclosing
    scopes even if they are not present in memory. Read only access!!

    :mask_image: a binary image (assumes foreground / background classes)
    :return: weighted loss
    """

    def mse_wtd(y_true, y_pred):
        try:
            y_true, mask_image = tf.split(y_true, [n_channels, 1], axis=1)
        except Exception as e:
            print('cannot separate mask and y_true' + str(e))
        keras_loss.kullback_leibler_divergence()
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        weights = K.batch_flatten(mask_image)
        weights = K.cast(weights, 'float32')
        # loss = K.square(y_pred - y_true)
        loss = K.abs(y_pred - y_true)
        
        fg_count = K.sum(weights, axis=1)
        total_count = K.cast(K.shape(y_true)[1], 'float32')
        fg_vol_frac = tf.div(fg_count, total_count)
        bg_vol_frac = 1 - fg_vol_frac
        # fg_vf is a tensor
        fg_weights = tf.where(fg_vol_frac >= 0.5, fg_vol_frac, bg_vol_frac)
        fg_mask = weights * K.expand_dims(fg_weights, axis=1)
        bg_mask = (1 - weights) * K.expand_dims(1 - fg_weights, axis=1)
        
        mask = fg_mask + bg_mask
        modified_loss = K.mean(loss * mask, axis=1)
        # modified_loss = tf.Print(modified_loss, [modified_loss], message='modified_loss', summarize=16)
        return modified_loss
    return mse_wtd


def dice_coef_loss(y_true, y_pred):
    """
    The Dice loss function is defined by 1 - DSC
    since the DSC is in the range [0,1] where 1 is perfect overlap
    and we're looking to minimize the loss.

    :param y_true: true values
    :param y_pred: predicted values
    :return: Dice loss
    """
    return 1. - metrics.dice_coef(y_true, y_pred)


def mae_kl(loss_wts):
    """Weighted sum of mae and kl losses

    :param list loss_wts: weights for individual losses
    :return: weighted loss of mae and kl
    """

    def loss_fn(y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        total_count = K.cast(K.shape(y_true)[1], 'float32')
        mae = keras_loss.mean_absolute_error(y_true, y_pred)

        y_true_range = [K.min(y_true), K.max(y_true)]
        y_true_hist = tf.histogram_fixed_width(y_true, y_true_range,
                                               nbins=256, dtype=tf.int32)
        y_pred_hist = tf.histogram_fixed_width(y_pred, y_true_range,
                                               nbins=256, dtype=tf.int32)
        # KL is for probability distributions, normalize hist
        y_true_hist = K.cast(y_true_hist, 'float32')
        y_true_hist = y_true_hist / total_count
        y_pred_hist = K.cast(y_pred_hist, 'float32')
        y_pred_hist = y_pred_hist / total_count
        kl = keras_loss.kullback_leibler_divergence(y_true_hist, y_pred_hist)
        loss = (K.cast(loss_wts[0], 'float32') * mae +
                K.cast(loss_wts[1], 'float32') * kl)
        return loss
    return loss_fn
