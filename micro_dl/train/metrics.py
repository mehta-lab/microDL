"""Custom metrics"""
import keras.backend as K


def coeff_determination(y_true, y_pred):
    """R^2 Goodness of fit, using as a proxy for accuracy in regression"""

    SS_res = K.sum(K.square(y_true - y_pred ))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient with smoothing.
    Maybe try Jaccard too?
    https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
