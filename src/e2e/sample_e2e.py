import copy
import numpy as np
from wireless_tpp.utils import logger

NUM_RBS_PADDING = 106
NUM_SYMBOLS_PADDING = 14
MRETX_PADDING = 4
RFAILED_PADDING = 2

def predict_scheduling(sched_runner, sched_history_sequence, segment_num, mcs_index, exclude_link_quality, exp_config) -> np.ndarray:
    """
    sched_history_sequence: [history_size, mcs_size, sched_sequence_length]
    For a certain arrival_history_sequence which is a batch of sequences of scheduling events,
    returns a list of predicted scheduling events (multiple samples) for the next segment_num segment.
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    history_size = sched_history_sequence.shape[0]
    mcs_size = sched_history_sequence.shape[1]
    history_sequence_length = sched_history_sequence.shape[2]

    # input dim are [history_size, mcs_size, history_sequence_length]
    # output dim are [history_size, mcs_size, history_sequence_length+1]
    sched_history = np.empty((history_size, mcs_size, history_sequence_length+1), dtype=object)
    no_scheduling_mask = np.empty((history_size, mcs_size), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idx, idy] in [history_size, mcs_size]
    for idx in range(history_size):
        for idy in range(mcs_size):
            no_scheduling_mask[idx, idy] = segment_is_not_needed(sched_history_sequence[idx,idy])
            # here we only copy the last model_history_sequence_length-1 events in the history because that is what the model needs
            sched_history[idx, idy, :] = np.append(
                copy.deepcopy(sched_history_sequence[idx, idy, :]),
                {
                    'idx_event' : 0, 'type_event': -1, 'slot' : 0, 'len' : 0, 'mcs_index' : 0, 
                    'mretx' : 0, 'rfailed' : 0, 'num_rbs' : 0, 'num_symbols' : 0, 'time_since_start' : 0, 
                    'time_since_last_event' : 0, 'timestamp' : 0, 'num_rbs': 0, 'packet_id': 0, 'segment': -1, 'depart_timestamp': 0
                }
            )
            for pos in range(history_sequence_length+1):
                sched_history[idx,idy,pos]['idx_event'] = pos

    # combine sched_history first two dimensions
    # input is [history_size, mcs_size, history_sequence_length+1]
    # result is [history_size*mcs_size, history_sequence_length+1]
    sched_history = sched_history.reshape(-1, history_sequence_length+1)

    # predict the next segment
    result = sched_runner.run(
        batch_size=sched_history.shape[0],
        source_data=sched_history,
        data_specs={
            "num_event_types": 4, #FIXME
            "pad_token_id": 4, #FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_dtime = []
    p_len = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_len.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=1)
    cp_len = np.concatenate(p_len, axis=1)

    num_samples = cp_dtime.shape[0]
    cp_dtime = cp_dtime[:, :, 0] # [num_samples_dtime, history_size*mcs_size]
    cp_len = cp_len[:, :, 0] # [num_samples_len, history_size*mcs_size]
    last_history_events = sched_history_sequence[:, :, -1] # [history_size, mcs_size]

    cp_dtime = cp_dtime.reshape(num_samples, history_size, mcs_size) # [num_samples_len, history_size, mcs_size]
    cp_len = cp_len.reshape(num_samples, history_size, mcs_size) # [num_samples_len, history_size, mcs_size]
    assert cp_dtime.shape[1] == last_history_events.shape[0] # history_size should be the same
    assert cp_dtime.shape[2] == last_history_events.shape[1] # mcs_size should be the same
    assert cp_len.shape[0] == cp_dtime.shape[0] # sample numbers should be the same
    
    len_bytes = np.empty((num_samples, history_size, mcs_size), dtype=object)
    predictions = np.empty((num_samples, history_size, mcs_size), dtype=object)
    for idx in range(history_size):
        for idy in range(mcs_size):
            for idz in range(num_samples):
                pred_sched_dtime = cp_dtime[idz, idx, idy]
                pred_sched_len = int(cp_len[idz, idx, idy])
                prev_sched_event = last_history_events[idx, idy]

                # complete the prediction for this sample
                pred_segment_time_since_start = (prev_sched_event['time_since_start'] + pred_sched_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
                pred_segment_slot = (prev_sched_event['slot'] + pred_sched_dtime/slots_duration_ms) % (num_slots_per_frame)
                pred_timestamp = np.float64(prev_sched_event['timestamp']) + np.float64(pred_sched_dtime/1000.0)

                predictions[idz, idx, idy] = {
                    'idx_event' : -1,
                    'type_event': segment_num + 1,
                    'slot' : pred_segment_slot,
                    'len' : pred_sched_len,
                    'mcs_index' : mcs_index[idy],
                    'mretx' : 0 if exclude_link_quality else -1,
                    'rfailed' : 0 if exclude_link_quality else -1,
                    'num_rbs' : -1,
                    'num_symbols' : 3, # FIXME
                    'time_since_start' : pred_segment_time_since_start,
                    'time_since_last_event' : pred_sched_dtime,
                    'timestamp' : pred_timestamp
                }

                len_bytes[idz,idx,idy] = pred_sched_len

    # input is [history_size, mcs_size]
    # output is [num_samples, history_size, mcs_size]
    # repeat no_scheduling_mask to match the shape of predictions
    no_scheduling_mask = np.repeat(no_scheduling_mask[np.newaxis, :, :], num_samples, axis=0)
    return predictions, no_scheduling_mask, len_bytes


def predict_retx(retx_runner, retx_history_sequence, mcs_index : np.ndarray, num_rbs : np.ndarray, exp_config) -> np.ndarray:
    """
    input dims:
    retx_history_sequence dims: [retx_history_size, mcs_size, history_sequence_length]
    mcs_index dims: [mcs_size]
    num_rbs dims: [num_sched_samples, sched_history_size, mcs_size]

    NOTE: retx_history_size and sched_history_size are equal and correspond to the same history

    For a certain arrival_history_sequence which is a batch of sequences of scheduling events,
    returns a list of predicted scheduling events (multiple samples) for the next segment_num segment.
    """

    assert retx_history_sequence.shape[1] == mcs_index.shape[0]
    assert retx_history_sequence.shape[1] == num_rbs.shape[2]
    assert retx_history_sequence.shape[0] == num_rbs.shape[1]

    history_size = retx_history_sequence.shape[0]
    history_sequence_length = retx_history_sequence.shape[2]
    mcs_size = mcs_index.shape[0]
    assert mcs_size == num_rbs.shape[2]
    assert mcs_size == retx_history_sequence.shape[1]
    num_sched_samples = num_rbs.shape[0]

    # input dimensions are [history_size, mcs_size, history_sequence_length]
    # output dimensions are [num_sched_samples, history_size, mcs_size, history_sequence_length]
    cond_retx_history = np.empty((num_sched_samples, history_size, mcs_size, history_sequence_length+1), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idx, idy, idz] in [history_size, num_sched_samples, mcs_size]
    for idx in range(history_size):
        for idy in range(num_sched_samples):
            for idz in range(mcs_size):
                cond_retx_history[idy, idx, idz, :] = np.append(
                    copy.deepcopy(retx_history_sequence[idx, idz, :]),
                    {
                        'idx_event': 0, # we will fix it later
                        'type_event': 0, #block attempt event
                        'timestamp': 0,
                        'time_since_start': 0,
                        'time_since_last_event': 0,
                        'rfailed': 0,
                        'mretx': 0,
                        'mcs_index': mcs_index[idz], # conditionally set
                        'num_rbs': num_rbs[idy, idx, idz], # conditionally set
                    }
                )
            for pos in range(history_sequence_length+1):
                cond_retx_history[idy, idx, idz, pos]['idx_event'] = pos

    # combine cond_retx_history first two dimensions
    # input is [num_sched_samples, history_size, mcs_size, history_sequence_length+1]
    # result is [num_sched_samples*history_size*mcs_size, history_sequence_length+1]
    cond_retx_history = cond_retx_history.reshape(-1, history_sequence_length+1)

    # predict retx
    result = retx_runner.run(
        batch_size=cond_retx_history.shape[0],
        source_data=cond_retx_history,
        data_specs={
            "num_event_types": 2, #FIXME
            "pad_token_id": 2, #FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_retx = []
    p_rfailed = []
    for batch in result['pred']:
        p_retx.append(batch[0])
        p_rfailed.append(batch[1])
    cp_retx = np.concatenate(p_retx, axis=1)
    cp_rfailed = np.concatenate(p_rfailed, axis=1)
    cp_retx = cp_retx[:, :, 0] # [num_retx_samples, history_size*num_sched_samples*mcs_size]
    cp_rfailed = cp_rfailed[:, :, 0] # [num_retx_samples, history_size*num_sched_samples*mcs_size]
    num_retx_samples = cp_retx.shape[0]

    # input is [num_retx_samples, num_sched_samples*history_size*mcs_size]
    # result is [num_retx_samples, num_sched_samples, history_size, mcs_size]
    predictions_retx = cp_retx.reshape(num_retx_samples, num_sched_samples, history_size, mcs_size)
    predictions_rfailed = cp_rfailed.reshape(num_retx_samples, num_sched_samples, history_size, mcs_size)
    return predictions_retx, predictions_rfailed


def predict_mcs(mcs_runner, mcs_history_sequence, mcs_eval_interval_ms) -> np.ndarray:
    """
    In this function, first we check if it is time for a new MCS prediction.
    Since MCS prediction is done every 100ms typically, we check first that
    at the time of the new arrival prediction, the time for new MCS index evaluation has come.
    Then we make a prediction and return the result.
    input dims:
    mcs_history_sequence dims: [upd_mcs_history_size, mcs_size, history_sequence_length]
    output dims:
    predictions dims: [num_samples, upd_mcs_history_size, mcs_size]
    """

    upd_mcs_history_size = mcs_history_sequence.shape[0]
    mcs_size = mcs_history_sequence.shape[1]
    history_sequence_length = mcs_history_sequence.shape[2]

    # NOTE: assumption: packet interarrival times are always less than 2*mcs_eval_interval_ms and more than mcs_eval_interval_ms

    # input dimensions are [upd_mcs_history_size, mcs_size, history_sequence_length]
    # output dimensions are [upd_mcs_history_size, mcs_size, history_sequence_length+1]
    next_mcs_eval_ts = np.empty((upd_mcs_history_size, mcs_size), dtype=object)
    mcs_history = np.empty((upd_mcs_history_size, mcs_size, history_sequence_length+1), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idy, idz] in [upd_mcs_history_size, mcs_size]
    for idy in range(upd_mcs_history_size):
        for idz in range(mcs_size):
            next_mcs_eval_ts[idy,idz] = find_next_mcs_eval_ts(
                mcs_history_sequence[idy, idz], mcs_eval_interval_ms
            )
            mcs_history[idy, idz, :] = np.append(
                copy.deepcopy(mcs_history_sequence[idy, idz, :]),
                { # dummy event for prediction only
                    'idx_event': 0, # we will fix it later
                    'type_event': 1, # MCS event
                    'timestamp' : 0, # does not matter
                    'time_since_start':0, # does not matter
                    'time_since_last_event': 0, # does not matter
                    'mcs_index' : 0, # does not matter
                    'rfailed' : RFAILED_PADDING,
                    'mretx' : MRETX_PADDING,
                    'num_rbs': NUM_RBS_PADDING
                }
            )
            for pos in range(history_sequence_length+1):
                mcs_history[idy, idz, pos]['idx_event'] = pos

    # combine cond_retx_history first two dimensions
    # input is [upd_mcs_history_size, mcs_size, history_sequence_length+1]
    # result is [upd_mcs_history_size*mcs_size, history_sequence_length+1]
    mcs_history = mcs_history.reshape(-1, history_sequence_length+1)
    
    # predict retx
    result = mcs_runner.run(
        batch_size=mcs_history.shape[0],
        source_data=mcs_history,
        data_specs={
            "num_event_types": 2, #FIXME
            "pad_token_id": 2, #FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_mcs = []
    for batch in result['pred']:
        p_mcs.append(batch[0])
    cp_mcs = np.concatenate(p_mcs, axis=1)
    cp_mcs = cp_mcs[:,:,0] # [num_mcs_samples, upd_mcs_history_size*mcs_size]
    num_mcs_samples = cp_mcs.shape[0]

    # input is [num_mcs_samples, upd_mcs_history_size*mcs_size]
    # result is [num_mcs_samples, upd_mcs_history_size, mcs_size]
    predictions_mcs = cp_mcs.reshape(num_mcs_samples, upd_mcs_history_size, mcs_size)

    # next_mcs_eval_ts dims: [upd_mcs_history_size, mcs_size]
    return predictions_mcs, next_mcs_eval_ts

def predict_arrival(arrival_history_sequence, arrival_runner, exp_config) -> np.ndarray:
    """
    arrival_history_sequence: [history_size, sequence_length]
    For a batch of arrival_history_sequence which is a batch of sequences of packet arrival events,
    returns predicted packet arrival events (multiple samples) for the next packet arrival.
    """
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    history_size = arrival_history_sequence.shape[0]
    history_sequence_length = arrival_history_sequence.shape[1]

    # input dimensions are [history_size, history_sequence_length]
    # output dimensions are [history_size, history_sequence_length+1]
    arrival_history = np.empty((history_size, history_sequence_length+1), dtype=object)
    # append a dummy event to the end of each sequence to act as the label
    # [idx] in [history_size]
    for idx in range(history_size):
        arrival_history[idx, :] = np.append(
            copy.deepcopy(arrival_history_sequence[idx, :]),
            {
                'idx_event': history_sequence_length, # we will fix it later
                'type_event': 0,
                'timestamp': 0,
                'time_since_start': 0,
                'time_since_last_event': 0,
            }
        )
        for pos in range(history_sequence_length+1):
            arrival_history[idx,pos]['idx_event'] = pos

    logger.info(f"packet arrival event prediction with sequence of shape: {arrival_history.shape}")
    result = arrival_runner.run(
        batch_size=arrival_history.shape[0],
        source_data=arrival_history,
        data_specs={
            "num_event_types": 10000, # a very large number
            "pad_token_id": 10000, # a very large number
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )
    p_dtime = []
    p_event_type = []
    for batch in result['pred']:
        p_dtime.append(batch[0])
        p_event_type.append(batch[1])
    cp_dtime = np.concatenate(p_dtime, axis=1)
    cp_event_type = np.concatenate(p_event_type, axis=1)

    cp_dtime = cp_dtime[:, :, 0] # [num_samples_dtime, batch_size]
    cp_event_type = cp_event_type[:, :, 0] # [num_samples_len, batch_size]
    last_history_events = arrival_history_sequence[:,-1] # [batch_size]
    batch_size = cp_dtime.shape[1]
    assert cp_dtime.shape[1] == last_history_events.shape[0] # batch sizes should be the same
    assert cp_event_type.shape[0] == cp_dtime.shape[0] # sample numbers should be the same
    num_samples = cp_dtime.shape[0]

    predictions = np.empty((num_samples, batch_size), dtype=object)
    for idx in range(batch_size):
        for idy in range(num_samples):
            # set the samples to proceed with
            pred_arrival_dtime = cp_dtime[idy, idx]
            pred_arrival_len = int(cp_event_type[idy, idx])
            prev_arrival_event = last_history_events[idx]

            pred_time_since_start = (prev_arrival_event['time_since_start'] + pred_arrival_dtime) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
            pred_timestap = prev_arrival_event['timestamp'] + pred_arrival_dtime/1000
            predictions[idy,idx] = {
                'idx_event': history_sequence_length,
                'type_event' : int(pred_arrival_len),
                'time_since_start' : pred_time_since_start,
                'time_since_last_event' : pred_arrival_dtime,
                'timestamp' : pred_timestap
            }
    # output dimensions are [num_samples, batch_size]
    return predictions

def calc_num_rbs(mcs_index : np.ndarray, len_bytes : np.ndarray):
    """
    mcs_index: [ mcs_size ]
    len_bytes: [num_sched_samples, sched_history_size, mcs_size]
    returns num_rbs: [num_sched_samples, sched_history_size, mcs_size]
    """
    mcs_size = mcs_index.shape[0]
    len_size = len_bytes.shape[0]
    history_size = len_bytes.shape[1]
    num_rbs = np.zeros((len_size, history_size, mcs_size), dtype=int)
    for idx in range(len_size):
        for idy in range(history_size):
            for idz in range(mcs_size):
                if len_bytes[idx, idy, idz] <= 20:
                    num_rbs[idx, idy, idz] = 5
                elif len_bytes[idx, idy, idz] > 20:
                    num_rbs[idx, idy, idz] = 26
    return num_rbs

def find_next_mcs_eval_ts(mcs_history_sequence, mcs_eval_interval_ms):
    """
    mcs_history_sequence: [mcs_sequence_length]
    """
    # iterate backwards to find the last mcs decision event
    for idx in range(mcs_history_sequence.shape[0]-1, -1, -1):
        if int(mcs_history_sequence[idx]['type_event']) == 1:
            # MCS decision event
            return mcs_history_sequence[idx]['timestamp'] + (mcs_eval_interval_ms/1000.0)
    logger.error("mcs_eval_is_not_needed: no MCS decision event found in the history sequence")
    return None

def segment_is_not_needed(sched_history_sequence):
    """
    sched_history_sequence: [sched_sequence_length]
    """
    departed_bytes = 0
    # iterate backwards to find the last packet arrival
    for idx in range(sched_history_sequence.shape[0]-1, -1, -1):
        if int(sched_history_sequence[idx]['type_event']) > 0:
            # segment event
            if sched_history_sequence[idx]['rfailed'] == 0:
                departed_bytes += int(sched_history_sequence[idx]['len'])
        elif int(sched_history_sequence[idx]['type_event']) == 0:
            # packet arrival event
            if int(departed_bytes) >= int(sched_history_sequence[idx]['len']):
                return True
            else:
                return False
    logger.error("segment_is_not_needed: no packet arrival event found in the history sequence")
    return None

def project_segment_predictions(sched_predictions, predictions_retx, predictions_rfailed, num_rbs, no_sched_mask):
    # now set the mretx and rfailed for the segment predictions
    # sched_predictions dims: [num_sched_samples, sched_history_size, mcs_size]
    # predictions_retx dims: [num_retx_samples, num_sched_samples, retx_history_size, mcs_size]
    # predictions_rfailed dims: [num_retx_samples, num_sched_samples, retx_history_size, mcs_size]
    # num_rbs dims: [num_sched_samples, sched_history_size, mcs_size]
    num_retx_samples = predictions_retx.shape[0]
    num_sched_samples = sched_predictions.shape[0]
    retx_history_size = predictions_retx.shape[2]
    mcs_size = sched_predictions.shape[2]
    assert num_sched_samples == num_rbs.shape[0]
    assert num_sched_samples == sched_predictions.shape[0]
    assert num_sched_samples == predictions_rfailed.shape[1]
    assert num_sched_samples == predictions_retx.shape[1]

    num_rfailed, num_mretx, len_bytes = 0,0,0
    segment_predictions = np.empty((num_retx_samples, num_sched_samples, retx_history_size, mcs_size), dtype=object)
    for idx in range(num_retx_samples):
        for idy in range(num_sched_samples):
            for idz in range(retx_history_size):
                for idk in range(mcs_size):
                    segment_predictions[idx,idy,idz,idk] = copy.deepcopy(sched_predictions[idy,idz,idk])
                    segment_predictions[idx,idy,idz,idk]['mretx'] = predictions_retx[idx,idy,idz,idk]
                    segment_predictions[idx,idy,idz,idk]['rfailed'] = predictions_rfailed[idx,idy,idz,idk]
                    segment_predictions[idx,idy,idz,idk]['num_rbs'] = num_rbs[idy,idz,idk]
                    num_rfailed += int(predictions_rfailed[idx,idy,idz,idk])
                    num_mretx += int(predictions_retx[idx,idy,idz,idk] > 0)
                    len_bytes += int(sched_predictions[idy,idz,idk]['len'])


    # repeat no_sched_mask to match the shape of segment_predictions
    # input dims: [num_sched_samples, sched_history_size, mcs_size]
    # output dims: [num_retx_samples, num_sched_samples, sched_history_size, mcs_size]
    no_sched_mask = np.repeat(no_sched_mask[np.newaxis, :, :], num_retx_samples, axis=0)

    # size becomes [ segment_pred_size = num_retx_samples*num_sched_samples, retx_history_size, mcs_size]
    segment_predictions = segment_predictions.reshape(-1, retx_history_size, mcs_size)
    no_sched_mask = no_sched_mask.reshape(-1, retx_history_size, mcs_size)

    return num_rfailed, num_mretx, len_bytes, segment_predictions, no_sched_mask

def append_arrival_predictions_to_arrival_history(arrival_predictions, arrival_history_sequence):
    # inputs:
    # arrival_predictions: [num_arrival_samples, arrival_history_size]
    # arrival_history_sequence: [arrival_history_size, arrival_sequence_length]
    # outputs:
    # upd_arrival_history: [num_arrival_samples*arrival_history_size, arrival_sequence_length]
    num_arrival_samples = arrival_predictions.shape[0]
    arrival_history_size = arrival_history_sequence.shape[0]
    arrival_sequence_length = arrival_history_sequence.shape[1]

    # append the arrival predictions to the arrival history sequences
    upd_arrival_history = np.empty((num_arrival_samples, arrival_history_size, arrival_sequence_length), dtype=object)
    for idx in range(num_arrival_samples):
        for idy in range(arrival_history_size):
            upd_arrival_history[idx,idy,:] = np.append(
                copy.deepcopy(arrival_history_sequence[idy, 1:]),
                copy.deepcopy(arrival_predictions[idx,idy]) # append the new event
            )
            for pos in range(arrival_sequence_length):
                upd_arrival_history[idx,idy,pos]['idx_event'] = pos
    upd_arrival_history = upd_arrival_history.reshape(-1, arrival_sequence_length)
    # upd_arrival_history dim: [upd_arrival_history_size = num_arrival_samples*arrival_history_size, arrival_sequence_length]
    logger.info(f"updated arrival history shape: {upd_arrival_history.shape}")
    return upd_arrival_history


def append_mcs_predictions_to_history(mcs_predictions, upd_mcs_history, upd_sched_history, upd_retx_history, next_mcs_eval_ts, exp_config):
    # inputs:
    # mcs_predictions: [num_mcs_samples, upd_mcs_history_size, mcs_size]
    # mcs_history_sequence: [upd_mcs_history_size, mcs_size, mcs_sequence_length]
    # next_mcs_eval_ts: [upd_mcs_history_size, mcs_size]
    # outputs:
    # upd2_mcs_history: [upd_mcs_history_size, num_mcs_samples, mcs_size, mcs_sequence_length]
    num_mcs_samples = mcs_predictions.shape[0]
    upd_mcs_history_size = upd_mcs_history.shape[0]
    upd_retx_history_size = upd_retx_history.shape[0]
    upd_sched_history_size = upd_sched_history.shape[0]
    assert upd_mcs_history_size == upd_retx_history_size
    assert upd_retx_history_size == upd_sched_history_size
    mcs_size = upd_mcs_history.shape[1]
    mcs_sequence_length = upd_mcs_history.shape[2]
    sched_sequence_length = upd_sched_history.shape[2]
    retx_sequence_length = upd_retx_history.shape[2]

    max_num_frames = exp_config['max_num_frames']
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    # append the mcs predictions to the mcs history sequences
    upd2_mcs_history = np.empty((upd_mcs_history_size, num_mcs_samples, mcs_size, mcs_sequence_length), dtype=object)
    upd_mcs_index = np.empty((num_mcs_samples, mcs_size), dtype=int)
    for idx in range(num_mcs_samples):
        for idy in range(upd_mcs_history_size):
            for idz in range(mcs_size):
                prev_mcs_event = upd_mcs_history[idy,idz, -1]
                pred_timestamp = next_mcs_eval_ts[idy,idz]
                pred_dtime_ms = (pred_timestamp - prev_mcs_event['timestamp'])*1000
                pred_time_since_start = (prev_mcs_event['time_since_start'] + pred_dtime_ms) % (max_num_frames*num_slots_per_frame*slots_duration_ms)
                upd2_mcs_history[idy,idx,idz] = np.append(
                    copy.deepcopy(upd_mcs_history[idy,idz,1:]),
                    { # dummy event for prediction only
                        'idx_event': mcs_sequence_length, # we will fix it later
                        'type_event': 1, # MCS event
                        'timestamp' : pred_timestamp,
                        'time_since_start': pred_time_since_start,
                        'time_since_last_event' : pred_dtime_ms,
                        'mcs_index' : int(mcs_predictions[idx,idy,idz]),
                        'rfailed' : RFAILED_PADDING,
                        'mretx' : MRETX_PADDING,
                        'num_rbs': NUM_RBS_PADDING
                    }
                )
                upd_mcs_index[idx,idz] = int(mcs_predictions[idx,idy,idz])
                for pos in range(mcs_sequence_length):
                    upd2_mcs_history[idy,idx,idz,pos]['idx_event'] = pos


    # repeat upd_sched_history and upd_retx_history to match the size of upd2_mcs_history
    upd2_sched_history = np.repeat(upd_sched_history[:, np.newaxis, :, :], num_mcs_samples, axis=1)
    upd2_retx_history = np.repeat(upd_retx_history[:, np.newaxis, :, :], num_mcs_samples, axis=1)
    # now all history sequences have the same size:
    # [upd_mcs_history_size, num_mcs_samples, mcs_size, mcs_sequence_length]

    # reshape them to be used in the next step: [upd_mcs_history_size, num_mcs_samples*mcs_size, mcs_sequence_length]

    upd2_sched_history = upd2_sched_history.reshape(upd_mcs_history_size, num_mcs_samples * mcs_size, sched_sequence_length)
    upd2_retx_history = upd2_retx_history.reshape(upd_mcs_history_size, num_mcs_samples * mcs_size, retx_sequence_length)
    upd2_mcs_history = upd2_mcs_history.reshape(upd_mcs_history_size, num_mcs_samples * mcs_size, mcs_sequence_length)
    upd_mcs_index = upd_mcs_index.reshape(-1)

    return upd2_sched_history, upd2_retx_history, upd2_mcs_history, upd_mcs_index


def append_arrival_predictions_to_history(upd_arrival_history, upd2_sched_history, upd2_retx_history, upd2_mcs_history, exp_config):
    # first update sched history sequence, then repeat the other two histories as they won't change
    # inputs: [upd_sched_history_size, upd_mcs_size, sched_sequence_length]
    # upd_arrival_history: [upd_arrival_history_size, arrival_sequence_length]
    # upd2_sched_history: [upd2_sched_history_size, upd_mcs_size, sched_sequence_length]
    # upd2_retx_history: [upd2_retx_history_size, upd_mcs_size, retx_sequence_length]
    # upd2_mcs_history: [upd2_mcs_history_size, upd_mcs_size, mcs_sequence_length]
    # outputs:
    # upd3_sched_history: [upd_arrival_history_size*upd_sched_history_size, upd_mcs_size, sched_sequence_length]
    # upd3_retx_history: [upd_arrival_history_size*upd_retx_history_size, upd_mcs_size, retx_sequence_length]
    # upd3_mcs_history: [upd_arrival_history_size*upd_mcs_history_size, upd_mcs_size, mcs_sequence_length]
    upd_arrival_history_size = upd_arrival_history.shape[0]
    upd_mcs_size = upd2_sched_history.shape[1]
    sched_sequence_length = upd2_sched_history.shape[2]
    retx_sequence_length = upd2_retx_history.shape[2]
    mcs_sequence_length = upd2_mcs_history.shape[2]
    upd2_sched_history_size = upd2_sched_history.shape[0]

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    # append the arrival predictions to the scheduling history sequences
    # input dims upd_sched_history: [upd_sched_history_size, upd_mcs_size, sched_sequence_length]
    # output dims upd2_sched_history: [upd_arrival_history_size, upd_sched_history_size, upd_mcs_size, sched_sequence_length]
    upd3_sched_history = np.empty((upd_arrival_history_size, upd2_sched_history_size, upd_mcs_size, sched_sequence_length), dtype=object)
    for idx in range(upd_arrival_history_size):
        for idy in range(upd2_sched_history_size):
            for idz in range(upd_mcs_size):
                pred_arrival_event = upd_arrival_history[idx,-1] # newly predicted arrival event
                prev_sched_event = upd2_sched_history[idy, idz, -1]
                pred_sched_dtime = (pred_arrival_event['timestamp'] - prev_sched_event['timestamp'])*1000
                pred_segment_slot = (prev_sched_event['slot'] + pred_sched_dtime/slots_duration_ms) % (num_slots_per_frame)
                upd3_sched_history[idx,idy,idz,:] = np.append(
                    copy.deepcopy(upd2_sched_history[idy, idz, 1:]),
                    {
                        'idx_event' : -1, # will be fixed later
                        'type_event': 0, # arrival event in scheduling history has type zero
                        'slot' : pred_segment_slot, # will be fixed later
                        'len' : int(pred_arrival_event['type_event']),
                        'mcs_index' : 16, #pred_arrival_event['mcs_index'], FIXME
                        'mretx' : MRETX_PADDING,
                        'rfailed' : RFAILED_PADDING,
                        'num_rbs' : NUM_RBS_PADDING,
                        'num_symbols' : NUM_SYMBOLS_PADDING,
                        'time_since_start' : pred_arrival_event['time_since_start'],
                        'time_since_last_event' : pred_sched_dtime,
                        'timestamp' : pred_arrival_event['timestamp']
                    }
                )
                for pos in range(sched_sequence_length):
                    upd3_sched_history[idx, idy, idz, pos]['idx_event'] = pos

    # repeat retx, and mcs histories to match the size of upd2_sched_history
    upd3_retx_history = np.repeat(upd2_retx_history[np.newaxis, ...], upd_arrival_history_size, axis=0)
    upd3_mcs_history = np.repeat(upd2_mcs_history[np.newaxis, ...], upd_arrival_history_size, axis=0)
    # now all history sequences have the same size:
    # [upd_arrival_history_size, upd_sched_history, upd_mcs_size, sched_sequence_length]

    # reshape them to be used in the next prediction
    upd3_sched_history = upd3_sched_history.reshape(-1, upd_mcs_size, sched_sequence_length)
    upd3_retx_history = upd3_retx_history.reshape(-1, upd_mcs_size, retx_sequence_length)
    upd3_mcs_history = upd3_mcs_history.reshape(-1, upd_mcs_size, mcs_sequence_length)

    logger.info(f"arrival updated sched history shape: {upd3_sched_history.shape}")
    logger.info(f"arrival updated retx history shape: {upd3_retx_history.shape}")
    logger.info(f"arrival updated mcs history shape: {upd3_mcs_history.shape}")

    return upd3_sched_history, upd3_retx_history, upd3_mcs_history


def append_segment_predictions_to_history(
        segment_predictions, sched_history_sequence, retx_history_sequence, mcs_history_sequence, no_sched_mask,
        filter_successful_attempts_for_mcs
    ):
    # only append if the departed bytes are smaller than packet size (use no_sched_mask)
    # otherwise, keep the latest event as the last event
    # inputs: 
    # sched_history_sequence: [sched_history_size, mcs_size, sched_sequence_length]
    # retx_history_sequence: [retx_history_size, mcs_size, retx_sequence_length]
    # mcs_history_sequence: [mcs_history_size, mcs_size, mcs_sequence_length]
    # segment predictions: [ segment_pred_size, retx_history_size, mcs_size]
    # no_sched_mask: [segment_pred_size, sched_history_size, mcs_size]
    # output:
    # upd_sched_history: [segment_pred_size*sched_history_size, mcs_size, sched_sequence_length]
    # upd_retx_history: [segment_pred_size*sched_history_size, mcs_size, sched_sequence_length]
    # upd_mcs_history: [segment_pred_size*sched_history_size, mcs_size, sched_sequence_length]
    segment_pred_size = segment_predictions.shape[0]
    sched_history_size = sched_history_sequence.shape[0]
    retx_history_size = retx_history_sequence.shape[0]
    mcs_history_size = mcs_history_sequence.shape[0]
    mcs_size = sched_history_sequence.shape[1]
    sched_sequence_length = sched_history_sequence.shape[2]
    retx_sequence_length = retx_history_sequence.shape[2]
    mcs_sequence_length = mcs_history_sequence.shape[2]

    upd_sched_history = np.empty((segment_pred_size, sched_history_size, mcs_size, sched_sequence_length), dtype=object)
    upd_retx_history = np.empty((segment_pred_size, retx_history_size, mcs_size, retx_sequence_length), dtype=object)
    upd_mcs_history = np.empty((segment_pred_size, mcs_history_size, mcs_size, mcs_sequence_length), dtype=object)
    for idx in range(sched_history_size):
        for idy in range(segment_pred_size):
            for idz in range(mcs_size):
                # don't append if there is no new segment event
                if no_sched_mask[idy, idx, idz]:
                    # keep the latest event as the last event
                    upd_sched_history[idy,idx,idz,:] = copy.deepcopy(sched_history_sequence[idx, idz, :])
                    upd_retx_history[idy,idx,idz,:] = copy.deepcopy(retx_history_sequence[idx, idz, :])
                    upd_mcs_history[idy,idx,idz,:] = copy.deepcopy(mcs_history_sequence[idx, idz, :])
                    continue

                # append to sched history
                upd_sched_history[idy,idx,idz,:] = np.append(
                    copy.deepcopy(sched_history_sequence[idx, idz, 1:]), # remove the oldest event
                    copy.deepcopy(segment_predictions[idy, idx, idz]) # append the new event
                )
                for pos in range(sched_sequence_length):
                    upd_sched_history[idy,idx,idz,pos]['idx_event'] = pos

                # append to retx history
                block_event = copy.deepcopy(segment_predictions[idy, idx, idz])
                block_event['type_event'] = 0
                block_event['time_since_last_event'] = (block_event['timestamp'] - retx_history_sequence[idx, idz, -1]['timestamp'])*1000
                del block_event['len'], block_event['slot'], block_event['num_symbols']
                upd_retx_history[idy,idx,idz,:] = np.append(
                    copy.deepcopy(retx_history_sequence[idx, idz, 1:]), # remove the oldest event
                    block_event # append the new event
                )
                for pos in range(retx_sequence_length):
                    upd_retx_history[idy,idx,idz,pos]['idx_event'] = pos

                # append to mcs history
                if filter_successful_attempts_for_mcs and (not (segment_predictions[idy, idx, idz]['rfailed'] > 0 or segment_predictions[idy, idx, idz]['mretx'] > 0)):
                    upd_mcs_history[idy,idx,idz,:] = copy.deepcopy(mcs_history_sequence[idx, idz, :]) # repeat the same history
                else:
                    block_event = copy.deepcopy(segment_predictions[idy, idx, idz])
                    block_event['type_event'] = 0
                    block_event['time_since_last_event'] = (block_event['timestamp'] - mcs_history_sequence[idx, idz, -1]['timestamp'])*1000
                    del block_event['len'], block_event['slot'], block_event['num_symbols']
                    upd_mcs_history[idy,idx,idz,:] = np.append(
                        copy.deepcopy(mcs_history_sequence[idx, idz, 1:]), # remove the oldest event
                        block_event # append the new event
                    )
                    for pos in range(mcs_sequence_length):
                        upd_mcs_history[idy,idx,idz,pos]['idx_event'] = pos

    upd_sched_history = upd_sched_history.reshape(-1, mcs_size, sched_sequence_length)
    upd_retx_history = upd_retx_history.reshape(-1, mcs_size, retx_sequence_length)
    upd_mcs_history = upd_mcs_history.reshape(-1, mcs_size, mcs_sequence_length)

    logger.info(f"updated sched history shape: {upd_sched_history.shape}")
    logger.info(f"updated retx history shape: {upd_retx_history.shape}")
    logger.info(f"updated mcs history shape: {upd_mcs_history.shape}")

    return upd_sched_history, upd_retx_history, upd_mcs_history

def reduce_history_dimension(
        sched_history_sequence, retx_history_sequence, mcs_history_sequence, history_dimension_limit, mcs_dimension_limit
    ):

    segment_history_size = mcs_history_sequence.shape[0]
    mcs_size = mcs_history_sequence.shape[1]

    # dimension reduction for history sequences
    # randomly select history_dimension_limit samples out of segment_history_size
    # if segment_history_size is less than history_dimension_limit, then use all
    if segment_history_size > history_dimension_limit:
        selected_indices = np.random.choice(segment_history_size, history_dimension_limit, replace=False)
        sched_history_sequence = sched_history_sequence[selected_indices, ...]
        retx_history_sequence = retx_history_sequence[selected_indices, ...]
        mcs_history_sequence = mcs_history_sequence[selected_indices, ...]

    if mcs_size > mcs_dimension_limit:
        selected_indices = np.random.choice(mcs_size, mcs_dimension_limit, replace=False)
        mcs_index = mcs_index[selected_indices]
        sched_history_sequence = sched_history_sequence[:, selected_indices, :]
        retx_history_sequence = retx_history_sequence[:, selected_indices, :]
        mcs_history_sequence = mcs_history_sequence[:, selected_indices, :]

    # segment predictions have dim: [segment_pred_size, sched_history_size, mcs_size]
    logger.info(f"reduced dim sched history sequences shape: {sched_history_sequence.shape}")
    logger.info(f"reduced dim retx history sequences shape: {retx_history_sequence.shape}")
    logger.info(f"reduced dim mcs history sequences shape: {mcs_history_sequence.shape}")
    
    return sched_history_sequence, retx_history_sequence, mcs_history_sequence


def sample_based_e2e_prediction(
        data,
        arrival_runner, 
        mcs_runner, 
        retx_runner,
        sched_runner, 
        exp_config,
        num_packets,
        mcs_eval_interval_ms = 100,
        filter_successful_attempts_for_mcs = True,
        mcs_dimension_limit = 20,
        history_dimension_limit = 100,
        exclude_link_quality = True,
        max_num_segments = 5
    ):

    final_predictions = []

    arrival_history_sequence = np.array(
        [ data['arrival'] ]
    ) # [arrival_history_size, arrival_sequence_length]

    sched_history_sequence = np.array([
        [ data['scheduling' ] ]
    ]) # [segment_history_size, mcs_size, sched_sequence_length]

    mcs_history_sequence = np.array([
        [ data['mcs' ] ] 
    ]) # [segment_history_size, mcs_size, mcs_sequence_length]

    retx_history_sequence = np.array([
        [ data['retx' ] ]
    ] ) # [segment_history_size, mcs_size, retx_sequence_length]

    packet0_arrival = data['label'][0]
    mcs_index = np.array([ packet0_arrival['mcs_index'] ]) # [mcs_size]

    prediction_counter = 0

    for packet_num in range(num_packets):
        for segment_num in range(max_num_segments):
            
            # figure all sizes
            sched_history_size = sched_history_sequence.shape[0]
            retx_history_size = retx_history_sequence.shape[0]  
            mcs_history_size = mcs_history_sequence.shape[0]
            mcs_size = mcs_index.shape[0]

            # these three hisory variables should have the same size
            # becuase they are all related to the same segment predictions
            # these samples correspond to the same predictions
            # so they have to treated together
            assert mcs_history_size == retx_history_size == sched_history_size
            
            sched_history_sequence, retx_history_sequence, mcs_history_sequence = reduce_history_dimension(
                sched_history_sequence, retx_history_sequence, mcs_history_sequence, history_dimension_limit, mcs_dimension_limit
            )
            # update the shapes after dimension reduction
            sched_history_size = sched_history_sequence.shape[0]
            retx_history_size = retx_history_sequence.shape[0]    
            mcs_history_size = mcs_history_sequence.shape[0]

            logger.info(f"Predicting packet {packet_num}, segment: {segment_num}, history sequence dims - mcs: {mcs_history_sequence.shape}, retx: {retx_history_sequence.shape}, sched: {sched_history_sequence.shape}")

            # predict the scheduling event
            # the result is the segment's scheduling event, but mretx, rfailed, mcs, and segment number are not set
            # we either use link quality prediction for that or set them to 0
            # the prediction gives a list of samples
            # input: sched_history_sequence with dims [sched_history_size, mcs_size, sched_sequence_length]
            # output: [num_sched_samples, sched_history_size, mcs_size]
            sched_predictions, no_sched_mask, len_bytes = predict_scheduling(
                sched_runner, 
                sched_history_sequence, 
                segment_num,
                mcs_index,
                exclude_link_quality,
                exp_config
            )
            num_sched_samples = sched_predictions.shape[0]
            logger.info(f"sched predictions shape: {sched_predictions.shape}")

            # calc num_rbs for the segment predictions
            # len_bytes dims: [num_sched_samples, sched_history_size, mcs_size]
            # num_rbs_size: [num_sched_samples, sched_history_size, mcs_size]
            num_rbs = calc_num_rbs(mcs_index, len_bytes)
            logger.info(f"num_rbs shape: {num_rbs.shape}")

            # predict the retx events
            if not exclude_link_quality:
                # predict link quality event
                # input dims:
                # retx_history_sequence dims: [retx_history_size, mcs_size, retx_sequence_length]
                # mcs_index dims: [mcs_size]
                # num_rbs dims: [num_sched_samples, sched_history_size, mcs_size]
                predictions_retx, predictions_rfailed = predict_retx(
                    retx_runner, retx_history_sequence,  mcs_index, num_rbs, exp_config
                )
                # output dims: [num_retx_samples, num_sched_samples, retx_history_size, mcs_size]
                num_retx_samples = predictions_retx.shape[0]
                logger.info(f"retx predictions shape: {predictions_retx.shape}")

                # now set the mretx and rfailed for the segment predictions and project the segment predictions
                # sched_predictions dims: [num_sched_samples, sched_history_size, mcs_size]
                # predictions_retx dims: [num_retx_samples, num_sched_samples, retx_history_size, mcs_size]
                # num_rbs dims: [num_sched_samples, sched_history_size, mcs_size]
                num_rfailed, num_mretx, len_bytes, segment_predictions, no_sched_mask = project_segment_predictions(
                    sched_predictions, predictions_retx, predictions_rfailed, num_rbs, no_sched_mask
                )
                # output dims of segment_predictions: [num_retx_samples*num_sched_samples, retx_history_size, mcs_size]
                # output dims of no_sched_mask: [num_retx_samples*num_sched_samples, retx_history_size, mcs_size]

            else:
                # set num_rbs for the segment predictions
                for idx in range(num_sched_samples):
                    for idy in range(sched_history_size):
                        for idz in range(mcs_size):
                            sched_predictions[idx,idy,idz]['num_rbs'] = num_rbs[idx,idy,idz]
                # size becomes [ segment_pred_size = num_sched_samples, sched_history_size, mcs_size]
                segment_predictions = sched_predictions
    
            # segment predictions have dim: [segment_pred_size, sched_history_size, mcs_size]
            logger.info(f"segment predictions shape: {segment_predictions.shape}")

            # report stats
            # sum all no_sched_mask values by firts converting them to integers
            total_num_projections = no_sched_mask.shape[0]*no_sched_mask.shape[1]*no_sched_mask.shape[2]
            no_sched_mask_sum = np.sum(no_sched_mask.astype(int))
            logger.info(f"Segment prediction status report: \n Total projections num: {total_num_projections}, completed packets:{no_sched_mask_sum}, num_rfailed:{num_rfailed}, num_mretx:{num_mretx} \n ratios - num_rfailed:{num_rfailed/total_num_projections}, num_mretx:{num_mretx/total_num_projections}, completed packets:{no_sched_mask_sum/total_num_projections} \n average bytes scheduled:{len_bytes/total_num_projections}")

            # append the segment predictions to the history sequences
            upd_sched_history, upd_retx_history, upd_mcs_history = append_segment_predictions_to_history(
                segment_predictions, sched_history_sequence, retx_history_sequence, mcs_history_sequence, no_sched_mask,
                filter_successful_attempts_for_mcs
            )                    
            # upd_sched_history dim: [segment_pred_size*sched_history_size, mcs_size, sched_sequence_length]
            # upd_retx_history dim: [segment_pred_size*retx_history_size, mcs_size, sched_sequence_length]
            # upd_mcs_history dim: [segment_pred_size*mcs_history_size, mcs_size, sched_sequence_length] 

            # check if this packet is done
            if no_sched_mask_sum/total_num_projections > 0.9:
                logger.info("No more segments to schedule, heading to the next packet")

                # prepare the results
                final_predictions.append(
                    upd_sched_history[:,:,-(prediction_counter+1):]
                )
                if len(final_predictions) == num_packets:
                    return final_predictions
                prediction_counter = 0

                # predict arrival as it is not dependent on link quality or scheduling
                # input dims: arrival_history_sequence dims: [arrival_history_size, arrival_sequence_length]
                # output dims: [num_arrival_samples, arrival_history_size]
                arrival_predictions = predict_arrival(
                    arrival_history_sequence, arrival_runner, exp_config
                )

                # output predictions_arrival has the size [num_arrival_samples, arrival_history_size]
                # append the arrival predictions to the arrival history sequences
                upd_arrival_history = append_arrival_predictions_to_arrival_history(
                    arrival_predictions, arrival_history_sequence
                )
                # output dims: [upd_arrival_history_size = num_arrival_samples*arrival_history_size, arrival_sequence_length]

                # inputs:
                # upd_mcs_history dims: [upd_mcs_history_size, mcs_size, history_sequence_length]
                # output dims:
                # predictions dims: [num_mcs_samples, upd_mcs_history_size, mcs_size]
                mcs_predictions, next_mcs_eval_ts = predict_mcs(
                    mcs_runner, upd_mcs_history, mcs_eval_interval_ms
                )

                # inputs:
                # mcs_predictions: [num_mcs_samples, mcs_history_size, mcs_size]
                # upd_mcs_history: [upd_mcs_history_size, mcs_size, mcs_sequence_length]
                # next_mcs_eval_ts: [mcs_history_size, mcs_size]
                # outputs:
                # upd2_sched_history: [upd_sched_history_size, upd_mcs_size = num_mcs_samples*mcs_size, mcs_sequence_length]
                # upd2_mcs_history: [upd_mcs_history_size, upd_mcs_size = num_mcs_samples*mcs_size, mcs_sequence_length]
                # upd2_retx_history: [upd_retx_history_size, upd_mcs_size = num_mcs_samples*mcs_size, mcs_sequence_length]
                upd2_sched_history, upd2_retx_history, upd2_mcs_history, upd_mcs_index = append_mcs_predictions_to_history(
                    mcs_predictions, upd_mcs_history, upd_sched_history, upd_retx_history, next_mcs_eval_ts, exp_config
                )

                # input history has the dim: [upd_sched_history_size, upd_mcs_size, sched_sequence_length]
                # output dims upd3_sched_history: [upd_arrival_history_size*upd_sched_history_size, upd_mcs_size, sched_sequence_length]
                upd3_sched_history, upd3_retx_history, upd3_mcs_history = append_arrival_predictions_to_history(
                    upd_arrival_history, upd2_sched_history, upd2_retx_history, upd2_mcs_history, exp_config
                )

                # update the history sequences
                mcs_history_sequence = upd3_mcs_history
                retx_history_sequence = upd3_retx_history
                sched_history_sequence = upd3_sched_history
                mcs_index = upd_mcs_index

                # go to the next packet
                break
            else:
                # update the history sequences
                mcs_history_sequence = upd_mcs_history
                retx_history_sequence = upd_retx_history
                sched_history_sequence = upd_sched_history
                prediction_counter += 1 # one more scheduling prediction

                # go to the next segment

    return final_predictions
