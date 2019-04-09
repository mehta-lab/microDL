"""Metrics for model evaluation"""
import functools
import numpy as np
import pandas as pd
from skimage.measure import compare_ssim as ssim
from scipy.stats import pearsonr


def mask_decorator(metric_function):
    """Decorator for masking the metrics"""

    @functools.wraps(metric_function)
    def wrapper_metric_function(**kwargs):
        """Expected inputs cur_target, prediction, mask"""

        metric = metric_function(target=kwargs['target'],
                                 prediction=kwargs['prediction'])
        if 'mask' in kwargs:
            mask = kwargs['mask']
            cur_target = kwargs['target']
            cur_pred = kwargs['prediction']
            masked_metric = metric_function(target=cur_target[mask],
                                            prediction=cur_pred[mask])
            return [metric, masked_metric]

        return metric
    return wrapper_metric_function


@mask_decorator
def mse_metric(target, prediction):
    """MSE of target and prediction"""

    return np.mean((target - prediction) ** 2)


@mask_decorator
def mae_metric(target, prediction):
    """MAE of target and prediction"""

    return np.mean(np.abs(target - prediction))


@mask_decorator
def r2_metric(target, prediction):
    """Coefficient of determination of target and prediction"""

    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    cur_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return cur_r2


@mask_decorator
def corr_metric(target, prediction):
    """Pearson correlation of target and prediction"""

    cur_corr = pearsonr(target.flatten(), prediction.flatten())[0]
    return cur_corr


def ssim_metric(target, prediction, mask=None):
    """SSIM of target and prediction"""

    if mask is None:
        cur_ssim = ssim(target, prediction,
                        data_range=target.max() - target.min())
        return cur_ssim
    else:
        cur_ssim, cur_ssim_img = ssim(target, prediction,
                                      data_range=target.max() - target.min(),
                                      full=True)
        cur_ssim_masked = np.mean(cur_ssim_img[mask])
        return [cur_ssim, cur_ssim_masked]


class MetricsEstimator:
    """Estimate metrics for evaluating a trained model"""

    def __init__(self, metrics_list, masked_metrics, len_data_split):
        """Init

        :param list metrics_list: list of strings with name of metrics
        :param bool masked_metrics: get the metrics for the masked region
        """

        available_metrics = {'ssim', 'corr', 'r2', 'mse', 'mae'}
        assert available_metrics.issubset(metrics_list), \
            'only ssim, r2, correlation, mse and mae are currently supported'
        self.metrics_list = metrics_list
        self.pd_col_names = metrics_list

        self.masked_metrics = masked_metrics
        if masked_metrics:
            self.pd_col_names.append('vol_frac')
            for metric in metrics_list:
                cur_col_name = '{}_masked'.format(metric)
                self.pd_col_names.append(cur_col_name)

        self.pd_col_names.append('tar_fname')
        df_metrics = pd.DataFrame(
            index=range(len_data_split),
            columns=self.pd_col_names
        )
        self.df_metrics = df_metrics
        self.row_idx = 0

    def get_metrics_df(self):
        """Return self.df_metrics"""

        return self.df_metrics

    def estimate_metrics(self, target,
                         prediction,
                         pred_fname,
                         mask=None):
        """Estimate metrics for the current input, target pair

        :param np.array target:
        :param np.array prediction:
        :param str pred_fname:
        :param np.array mask:
        """

        assert isinstance(pred_fname, str), \
            'more than one pred_fname is passed. Only one target-pred pair ' \
            'is handled per function call'
        assert target.shape == prediction.shape, \
            'The shape of target and prediction are not same: {}, {}'.format(
                target.shape, prediction.shape
            )
        assert target.dtype == prediction.dtype, \
            'The dtype of target and prediction are not same: {}, {}'.format(
                target.dtype, prediction.dtype
            )

        if mask is not None:
            assert target.shape == mask.shape, \
                'The shape of target and mask are not same: {}, {}'.format(
                    target.shape, mask.shape
                )
            assert mask.dtype == 'bool', 'mask is not boolean'

        fn_mapping = {'mae_metric': mae_metric,
                      'mse_metric': mse_metric,
                      'r2_metric': r2_metric,
                      'corr_metric': corr_metric,
                      'ssim_metric': ssim_metric}

        self.df_metrics.loc[self.row_idx]['tar_fname'] = pred_fname
        for cur_metric in self.metrics_list:
            metric_fn_name = '{}_metric'.format(cur_metric)
            metric_fn = fn_mapping[metric_fn_name]
            if self.masked_metrics:
                cur_metric_list = metric_fn(target=target,
                                            prediction=prediction,
                                            mask=mask)
                vol_frac = np.mean(mask)
            else:
                cur_metric = metric_fn(target=target,
                                       prediction=prediction)
            if self.masked_metrics:
                self.df_metrics.loc[self.row_idx]['vol_frac'] = vol_frac
                self.df_metrics.loc[self.row_idx][cur_metric] = \
                    cur_metric_list[0]
                metric_name = '{}_masked'.format(cur_metric)
                self.df_metrics.loc[self.row_idx][metric_name] = \
                    cur_metric_list[1]
            else:
                self.df_metrics.loc[self.row_idx][cur_metric] = cur_metric
        self.row_idx += 1
