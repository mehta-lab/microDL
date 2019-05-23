"""Custom metrics"""
import keras.backend as K
import numpy as np
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
    ssim_val = K.mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    return ssim_val


def _tf_fspecial_gauss(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1,
                              -size // 2 + 1:size // 2 + 1]

    x_data = x_data[:, np.newaxis, np.newaxis]
    y_data = y_data[:, np.newaxis, np.newaxis]

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    gauss = tf.exp(-((x ** 2 + y ** 2)/(2.0 * sigma ** 2)))
    return gauss / tf.reduce_sum(gauss)


def tf_ssim(img1, img2, im_range=255, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    Tensorflow implementation of SSIM

    :param img1:
    :param img2:
    :param cs_map:
    :param mean_metric:
    :param size:
    :param sigma:
    :return:
    """
    # window shape [size, size]
    window = _tf_fspecial_gauss(size, sigma)
    C1 = (0.01 * im_range) ** 2
    C2 = (0.03 * im_range) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(im1, im2, mean_metric=True, level=5):
    """
    Tensorflow implementation of MS-SSIM
    :param im1:
    :param im2:
    :param mean_metric:
    :param level:
    :return:
    """
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(im1, im2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        im1 = tf.nn.avg_pool(im1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        im2 = tf.nn.avg_pool(im2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


@flip_dimensions
def ms_ssim(y_true, y_pred, max_val=None):
    """
    MS-SSIM for 2D images over batches.
    Use max_val=6 to approximate maximum of normalized images.
    Tensorflow uses average pooling for each scale, so your tensor
    has to be relatively large (>170 pixel in x and y) for this to work.
    Warning: when using normalized images you often get nans since you'll
    compute small values to the power of small weights,
    so this implementation moves all images values up to positives in a
    hacky way by adding 255 to your prediction and target.

    :param tensor y_true: Labeled ground truth
    :param tensor y_pred: Predicted labels, potentially non-binary
    :param float max_val: The dynamic range of the images (i.e., the
        difference between the maximum the and minimum allowed values).
    :return float ms_ssim: Mean SSIM over images in the batch
    """
    # Re-normalize images to avoid nans when doing the weights exponential
    y_t = y_true + 255.
    y_p = y_pred + 255.
    msssim = K.mean(tf.image.ssim_multiscale(y_t, y_p, max_val=max_val))
    # If you're getting nans
    msssim = tf.where(tf.is_nan(msssim), 0., msssim)
    if msssim is None:
        msssim = K.constant(0.)
    return msssim


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
