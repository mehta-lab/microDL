"""Custom metrics"""
import keras.backend as K
import tensorflow as tf


def coeff_determination(y_true, y_pred):
    """
    R^2 Goodness of fit, using as a proxy for accuracy in regression

    :param y_true: Ground truth
    :param y_pred: Prediction
    :return float r2: Coefficient of determination
    """

    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


def mask_coeff_determination(n_channels):
    """split y_true into y_true and mask

    For masked_loss there's an added function/method to convert split
    y_true and pass to loss, metrics and callbacks.

    :param int n_channels: Number of channels
    """
    def coeff_deter(y_true, y_pred):
        """
        Coefficient of determination (R2)

        :param y_true: Ground truth
        :param y_pred: Prediction
        :return float r2: Coefficient of determination
        """
        if K.image_data_format() == "channels_last":
            split_axis = -1
        else:
            split_axis = 1
        y_true_split, mask = tf.split(y_true, [n_channels, 1], axis=split_axis)
        r2 = coeff_determination(y_true_split, y_pred)
        return r2
    return coeff_deter


def dice_coef(y_true, y_pred, smooth=1.):
    """
    This is a global non-binary Dice similarity coefficient (DSC)
    with smoothing.
    It computes an approximation of Dice but over the whole batch,
    and it leaves predicted output as continuous. This might help
    alleviate potential discontinuities a binary image level Dice
    might introduce.
    DSC = 2 * |A union B| /(|A| + |B|) = 2 * |ab| / (|a|^2 + |b|^2)
    where a, b are binary vectors
    smoothed DSC = (2 * |ab| + s) / (|a|^2 + |b|^2 + s)
    where s is smoothing constant.
    Although y_pred is not binary, it is assumed to be near binary
    (sigmoid transformed) so |y_pred|^2 is approximated by sum(y_pred).

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :param float smooth: Constant added for smoothing and to avoid
       divide by zeros
    :return float dice: Smoothed non-binary Dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def flip_dimensions(func):
    """
    Decorator to convert channels first tensor to channels last.

    :param func: Function to be decorated
    """
    def wrap_function(y_true, y_pred, max_val=6):
        if K.image_data_format() == 'channels_first':
            if K.ndim(y_true) > 4:
                y_true = tf.transpose(y_true, [0, 2, 3, 4, 1])
                y_pred = tf.transpose(y_pred, [0, 2, 3, 4, 1])
            else:
                y_true = tf.transpose(y_true, [0, 2, 3, 1])
                y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
        return func(y_true, y_pred, max_val)
    return wrap_function


@flip_dimensions
def ssim(y_true, y_pred, max_val=6):
    """Structural similarity
    Uses a default max_val=6 to approximate maximum of normalized images.
    Tensorflow does not support SSIM for 3D images. Need a different
    way to compute SSIM for 5D tensor (e.g. skimage).

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :param float max_val: The dynamic range of the images (i.e., the
        difference between the maximum the and minimum allowed values).
    :return float K.mean(ssim): mean SSIM over images in the batch
    """
    ssim = K.mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    return ssim


@flip_dimensions
def ms_ssim(y_true, y_pred, max_val=None):
    """
    MS-SSIM for 2D images over batches.
    Use max_val=6 to approximate maximum of normalized images.
    Tensorflow uses average pooling for each scale, so your tensor
    has to be relatively large (>170 pixel in x and y) for this to work.

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :param float max_val: The dynamic range of the images (i.e., the
        difference between the maximum the and minimum allowed values).
    :return float ms_ssim: Mean SSIM over images in the batch
    """
    ssim = K.mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val))
    # If you're getting nans
    if ssim != ssim:
        ssim = 0.
    return ssim


def pearson_corr(y_true, y_pred):
    """Pearson correlation
    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted label,
    :return float r: Pearson over all images in the batch
    """
    covariance = K.mean((y_pred - K.mean(y_pred)) *
                        (y_true - K.mean(y_true)))
    r = covariance / (K.std(y_pred) * K.std(y_true) + K.epsilon())
    return r
