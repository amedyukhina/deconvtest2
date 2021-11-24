from __future__ import print_function, unicode_literals, absolute_import, division

import os

import matplotlib.pyplot as plt
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict, plot_history
from csbdeep.utils.tf import limit_gpu_memory


def care_train(data_file: str, fn_output: str = 'test',
               train_epochs: int = 10, train_steps_per_epoch: int = 40, train_batch_size: int = 16,
               limit_gpu: float = 0.5, validation_split: float = 0.2,
               **kwargs):
    """

    Parameters
    ----------
    data_file : str
        File name for training data in ``.npz`` format (full path)
    fn_output : str
        Folder name to save the model (full path)
    train_epochs : int, optional
        Number of training epochs.
        Default: 10
        Should be set to ~100 for real training
    train_steps_per_epoch : int, optional
        Number of parameter update steps per epoch.
        Default: 40
        Should be set to ~400 for a real training
    train_batch_size : int, optional
        Batch size for training.
        Default: 16
    limit_gpu : float, optional
        Fraction of the GPU memory to use.
        Default: 0.5
    validation_split : float, optional
        Fraction of images to use as validation set during training.
        Default: 0.2
    kwargs : dict
        Configuration attributes (see below).

    Attributes
    ----------
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable.
        Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
        Default: ``{'factor': 0.5, 'patience': 10, 'min_delta': 0}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """
    print(data_file, fn_output)
    (X, Y), (X_val, Y_val), axes = load_training_data(data_file,
                                                      validation_split=validation_split,
                                                      verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    limit_gpu_memory(fraction=limit_gpu)
    config = Config(axes, n_channel_in, n_channel_out,
                    train_epochs=train_epochs, train_steps_per_epoch=train_steps_per_epoch,
                    train_batch_size=train_batch_size, **kwargs)
    model = CARE(config, fn_output.split('/')[-1], basedir=os.path.dirname(fn_output))
    history = model.train(X, Y, validation_data=(X_val, Y_val))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
    plt.savefig(os.path.join(fn_output, 'history.png'))
