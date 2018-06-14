import numpy as np

from keras.callbacks import Callback
from keras import backend as K


class CyclicLearning(Callback):
    """
    Custom Callback implementing cyclical learning rate (CLR)
    as in the paper: https://arxiv.org/abs/1506.01186.

    Learning rate is increased then decreased in a repeated
    triangular pattern over time. One triangle = one cycle.
    Initial amplitude is scaled by gamma ** iterations.
    https://keras.io/callbacks/
    https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 base_lr=0.001,
                 max_lr=0.006,
                 step_size=2000.,
                 gamma=1.,
                 scale_mode="cycle",
                 ):
        """
        :param float base_lr: Base (minimum) learning rate
        :param float max_lr: Maximum learning rate
        :param float step_size: The number of iterations per half cycle
        :param float gamma: Constant factor for max_lr exponential
            decrease over time (gamma ** iterations) gamma [0, 1]
        :param str scale_mode: Evaluate scaling on cycle number
            ('cycle' - default) or cycle iteration ('iterations')
        """
        super(CyclicLearning, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        assert gamma >= 0. and gamma <=1., \
            "Gamma is {}, should be [0, 1]".format(gamma)
        self.gamma = gamma
        self.iterations = 0.
        assert scale_mode in {"cycle", "iterations"}, \
            "Scale mode ({}) must be cycle or iterations".format(scale_mode)
        self.scale_mode = scale_mode

    def clr(self):
        """
        Updates the cyclic learning rate with exponential decline.

        :return float clr: Learning rate as a function of iterations
        """
        cycle = np.floor(1 + self.iterations / (2 * self.step_size))
        x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
        scale_factor = (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        if self.scale_mode == 'cycle':
            return self.base_lr + scale_factor * (self.gamma ** cycle)
        else:
            return self.base_lr + scale_factor * (self.gamma ** self.iterations)

    def on_train_begin(self, logs=None):
        """
        Set base learning rate at the beginning of training.

        :param logs: Logging from super class
        """
        logs = logs or {}

        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        """
        Updates the learning rate at the end of each batch. Prints
        learning rate along with other metrics during training.

        :param batch: Batch number from Callback super class
        :param logs: Log from super class (required but not used here)
        """
        logs = logs or {}

        self.iterations += 1
        print(" - clr: {:0.5f}".format(self.clr()))
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs):
        """
        Log learning rate at the end of each epoch for
        Tensorboard and CSVLogger.

        :param epoch: Epoch number from Callback super class
        :param logs: Log from super class
        """
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)
