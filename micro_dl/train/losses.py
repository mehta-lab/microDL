"""Custom losses"""
from keras import backend as K
import tensorflow as tf

import micro_dl.train.metrics as metrics


def mae_loss(y_true, y_pred):
    """Mean absolute error

    Keras losses by default calculate metrics along axis=-1, which works with
    image_format='channels_last'. The arrays do not seem to batch flattened,
    change axis if using 'channels_first
    """

    if K.image_data_format() == 'channels_last':
        return K.mean(K.abs(y_pred - y_true), axis=-1)
    else:
        return K.mean(K.abs(y_pred - y_true), axis=1)


def mse_loss(y_true, y_pred):
    """Mean squared loss"""

    if K.image_data_format() == 'channels_last':
        return K.mean(K.square(y_pred - y_true), axis=-1)
    else:
        return K.mean(K.square(y_pred - y_true), axis=1)


def kl_divergence_loss(y_true, y_pred):
    """KL divergence loss"""

    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    if K.image_data_format() == 'channels_last':
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    else:
        return K.sum(y_true * K.log(y_true / y_pred), axis=1)


def split_ytrue_mask(y_true, n_channels):
    """Split the mask concatenated with y_true

    :param keras.tensor y_true: if channels_first, ytrue has shape [batch_size,
     n_channels, y, x]. mask is concatenated as the n_channels+1, shape:
     [[batch_size, n_channels+1, y, x].
    :param int n_channels: number of channels in y_true
    :return:
     keras.tensor ytrue_split - ytrue with the mask removed
     keras.tensor mask_image - bool mask
    """

    try:
        if K.image_data_format() == 'channels_first':
            split_axis = 1
        else:
            split_axis = -1

        y_true_split, mask_image = tf.split(y_true, [n_channels, 1],
                                            axis=split_axis)
        return y_true_split, mask_image
    except Exception as e:
        print('cannot separate mask and y_true' + str(e))


def generate_vf_wtd_mask(mask_image):
    """Mask with values of vf and 1-vf in FG & BG, if vf>=0.5

    Else 1-vf and vf in FG & BG. FG-foreground, BF-background. binary mask!
    :param keras.tensor mask_image: with shape [batch_size, 1, y, x]
    :return keras.tensor mask: flatten mask with shape [batch_size, y*x] with
     values vf and 1-vf
    """

    weights = K.batch_flatten(mask_image)
    weights = K.cast(weights, 'float32')

    fg_count = K.sum(weights, axis=1)
    total_count = K.cast(K.shape(mask_image)[1], 'float32')
    fg_vol_frac = tf.div(fg_count, total_count)
    bg_vol_frac = 1 - fg_vol_frac
    # fg_vf is a tensor
    fg_weights = tf.where(fg_vol_frac >= 0.5, fg_vol_frac, bg_vol_frac)
    fg_mask = weights * K.expand_dims(fg_weights, axis=1)
    bg_mask = (1 - weights) * K.expand_dims(1 - fg_weights, axis=1)
    mask = fg_mask + bg_mask
    return mask


def mse_masked(n_channels):
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

    def mse_masked_loss(y_true, y_pred):
        y_true, mask_image = split_ytrue_mask(y_true, n_channels)
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        loss = mse_loss(y_true, y_pred)
        mask = generate_vf_wtd_mask(mask_image)
        modified_loss = K.mean(loss * mask, axis=1)
        # modified_loss = tf.Print(modified_loss, [modified_loss], message='modified_loss', summarize=16)
        return modified_loss
    return mse_masked_loss


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

    def mae_kl_loss(y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        total_count = K.cast(K.shape(y_true)[1], 'float32')
        mae = mae_loss(y_true, y_pred)

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
        kl = kl_divergence_loss(y_true_hist, y_pred_hist)
        loss = (K.cast(loss_wts[0], 'float32') * mae +
                K.cast(loss_wts[1], 'float32') * kl)
        return loss
    return mae_kl_loss
