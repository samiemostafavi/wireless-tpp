import numpy as np

from wireless_tpp.utils.const import PredOutputIndex
from wireless_tpp.utils.metrics import MetricsHelper


@MetricsHelper.register(name='rmse_2d', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_2d_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    # input prediction shape is [1, 2, batch_size, seq_len, 2]
    # we pick the last element of the sequence and remove the first 2 dims
    # result is [batch_size, seq_len, 2]
    predictions = np.array(predictions)[0,0,:,-1,:]

    # input label shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first dim, then we transpose the matrix
    labels = np.array(labels)[0,:,:,-1].T

    # Calculate RMSE for X and Y separately
    rmse_dtime = np.sqrt(np.mean((labels[:, 0] - predictions[:, 0]) ** 2))
    rmse_event = np.sqrt(np.mean((labels[:, 1] - predictions[:, 1]) ** 2))

    # Overall RMSE considering both X and Y together
    overall_rmse = np.sqrt(np.mean(np.sum((labels - predictions) ** 2, axis=1)))

    return overall_rmse

@MetricsHelper.register(name='rmse_2d_dtime', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_2d_dtime_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    # input prediction shape is [1, 2, batch_size, seq_len, 2]
    # we pick the last element of the sequence and remove the first 2 dims
    # result is [batch_size, seq_len, 2]
    predictions = np.array(predictions)[0,0,:,-1,:]

    # input label shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first dim, then we transpose the matrix
    labels = np.array(labels)[0,:,:,-1].T

    # Calculate RMSE for X and Y separately
    rmse_dtime = np.sqrt(np.mean((labels[:, 0] - predictions[:, 0]) ** 2))

    return rmse_dtime

@MetricsHelper.register(name='rmse_2d_event', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_2d_event_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    # input prediction shape is [1, 2, batch_size, seq_len, 2]
    # we pick the last element of the sequence and remove the first 2 dims
    # result is [batch_size, seq_len, 2]
    predictions = np.array(predictions)[0,0,:,-1,:]

    # input label shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first dim, then we transpose the matrix
    labels = np.array(labels)[0,:,:,-1].T

    # Calculate RMSE for X and Y separately
    rmse_event = np.sqrt(np.mean((labels[:, 1] - predictions[:, 1]) ** 2))

    return rmse_event

@MetricsHelper.register(name='rmse_dtime', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_dtime_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of 1D time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    # input prediction shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first 2 dims
    # result is [batch_size]
    predictions = np.array(predictions)[0,0,:,-1]

    # input label shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first dim, then we transpose the matrix\
    # result is [batch_size]
    labels = np.array(labels)[0,0,:,-1]

    # Overall RMSE considering both X and Y together
    return np.sqrt(np.mean((labels - predictions) ** 2))


@MetricsHelper.register(name='rmse_event', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_event_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of 1D event predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    # input prediction shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first 2 dims
    # result is [batch_size]
    predictions = np.array(predictions)[0,1,:,-1]

    # input label shape is [1, 2, batch_size, seq_len]
    # we pick the last element of the sequence and remove the first dim, then we transpose the matrix\
    # result is [batch_size]
    labels = np.array(labels)[0,1,:,-1]

    # Overall RMSE considering both X and Y together
    return np.sqrt(np.mean((labels - predictions) ** 2))


@MetricsHelper.register(name='rmse', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    seq_mask = kwargs.get('seq_mask')

    # apply the mask and remove the first element since it is repeated
    if len(seq_mask) > 0:
        pred = np.array(predictions[PredOutputIndex.TimePredIndex][seq_mask])
        label = np.array(labels[PredOutputIndex.TimePredIndex][seq_mask])
    else:
        pred = np.array(predictions[PredOutputIndex.TimePredIndex])[:, 1:]
        label = np.array(labels[PredOutputIndex.TimePredIndex])[:, 1:]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.sqrt(np.mean((pred - label) ** 2))

@MetricsHelper.register(name='acc', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def acc_metric_function(predictions, labels, **kwargs):
    """Compute accuracy ratio metrics of the type predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: accuracy ratio of the type predictions.
    """
    seq_mask = kwargs.get('seq_mask')

    # apply the mask and remove the first element since it is repeated
    if len(seq_mask) > 0:
        pred = np.array(predictions[PredOutputIndex.TypePredIndex][seq_mask])
        label = np.array(labels[PredOutputIndex.TypePredIndex][seq_mask])
    else:
        pred = np.array(predictions[PredOutputIndex.TypePredIndex])[:, 1:]
        label = np.array(labels[PredOutputIndex.TypePredIndex])[:, 1:]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.mean(pred == label)