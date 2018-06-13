import numpy as np

from keras.callbacks import Callback
from keras import backend as K


class CyclicLearning(Callback):
    """
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLearning(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLearning(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    """

    def __init__(self,
                 base_lr=0.001,
                 max_lr=0.006,
                 step_size=20.,
                 gamma=1.,
                 scale_mode='iterations',
                 ):
        """

        :param base_lr:
        :param max_lr:
        :param step_size:
        :param gamma:
        :param scale_mode:

            # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        """

        super(CyclicLearning, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        assert scale_mode in {"cycle", "iterations"}, \
            "Scale mode ({}) must be cycle or iterations".format(scale_mode)
        self.scale_mode = scale_mode
        self._reset()

    def _reset(self,
               new_base_lr=None,
               new_max_lr=None,
               new_step_size=None):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        scale_factor = (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        if self.scale_mode == 'cycle':
            return self.base_lr + scale_factor * (self.gamma ** cycle)
        else:
            return self.base_lr + scale_factor * (self.gamma ** self.clr_iterations)

    def on_train_begin(self, logs=None):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.trn_iterations += 1
        self.clr_iterations += 1
        print("clr", self.clr())
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs):
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)
