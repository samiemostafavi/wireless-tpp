import copy
import numpy as np
from wireless_tpp.utils import logger

NUM_RBS_PADDING = 106
NUM_SYMBOLS_PADDING = 14
MRETX_PADDING = 4
RFAILED_PADDING = 2

def predict_scheduling(sched_runner, sched_history_sequence, segment_num, mcs_index, exclude_link_quality, exp_config) -> np.ndarray:
    """
    Extended to batch dimension:
      sched_history_sequence: [batch_size, history_size, mcs_size, sched_sequence_length]
      mcs_index: [batch_size, mcs_size]
    Returns:
      predictions: [num_samples, batch_size, history_size, mcs_size]
      no_scheduling_mask: [num_samples, batch_size, history_size, mcs_size]
      len_bytes: [num_samples, batch_size, history_size, mcs_size]
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    batch_size = sched_history_sequence.shape[0]
    history_size = sched_history_sequence.shape[1]
    mcs_size = sched_history_sequence.shape[2]
    history_sequence_length = sched_history_sequence.shape[3]

    # We will create arrays of shape [batch_size, history_size, mcs_size, history_sequence_length+1]
    # then flatten to [batch_size*history_size*mcs_size, history_sequence_length+1] before feeding to the runner.
    sched_history = np.empty((batch_size, history_size, mcs_size, history_sequence_length+1), dtype=object)
    no_scheduling_mask = np.empty((batch_size, history_size, mcs_size), dtype=object)

    # Append a dummy event to the end of each sequence to act as the label
    for b in range(batch_size):
        for idx in range(history_size):
            for idy in range(mcs_size):
                no_scheduling_mask[b, idx, idy] = segment_is_not_needed(sched_history_sequence[b, idx, idy, :])
                # Construct the extended scheduling history
                new_seq = np.append(
                    copy.deepcopy(sched_history_sequence[b, idx, idy, :]),
                    {
                        'idx_event': 0, 'type_event': -1, 'slot': 0, 'len': 0, 'mcs_index': 0,
                        'mretx': 0, 'rfailed': 0, 'num_rbs': 0, 'num_symbols': 0, 'time_since_start': 0,
                        'time_since_last_event': 0, 'timestamp': 0, 'num_rbs': 0, 'packet_id': 0,
                        'segment': -1, 'depart_timestamp': 0
                    }
                )
                for pos in range(history_sequence_length+1):
                    new_seq[pos]['idx_event'] = pos
                sched_history[b, idx, idy, :] = new_seq

    # Flatten: [batch_size, history_size, mcs_size, L+1] -> [batch_size*history_size*mcs_size, L+1]
    sched_history_reshaped = sched_history.reshape(-1, history_sequence_length+1)

    # Run the prediction
    result = sched_runner.run(
        batch_size=sched_history_reshaped.shape[0],
        source_data=sched_history_reshaped,
        data_specs={
            "num_event_types": 4,  # FIXME
            "pad_token_id": 4,    # FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )

    # result['pred'] is typically a list/tuple of [prediction_for_dtime, prediction_for_len, ...]
    p_dtime = []
    p_len = []
    for batch_out in result['pred']:
        p_dtime.append(batch_out[0])
        p_len.append(batch_out[1])

    # Each p_dtime[i] is shape: [n_samples, N], same for p_len[i]
    cp_dtime = np.concatenate(p_dtime, axis=1)  # shape: [num_samples_dtime, N]
    cp_len = np.concatenate(p_len, axis=1)      # shape: [num_samples_len, N]
    # Here N = batch_size*history_size*mcs_size

    num_samples = cp_dtime.shape[0]
    # We only pick the first token's output (similar to the original code)
    cp_dtime = cp_dtime[:, :, 0]  # now [num_samples, N]
    cp_len = cp_len[:, :, 0]      # now [num_samples, N]

    # Un-flatten back to [num_samples, batch_size, history_size, mcs_size]
    cp_dtime = cp_dtime.reshape(num_samples, batch_size, history_size, mcs_size)
    cp_len = cp_len.reshape(num_samples, batch_size, history_size, mcs_size)

    # We need the last history event in each sequence to compute times
    # last_history_events has shape [batch_size, history_size, mcs_size]
    last_history_events = sched_history_sequence[:, :, :, -1]

    # Prepare final output arrays
    len_bytes = np.empty((num_samples, batch_size, history_size, mcs_size), dtype=object)
    predictions = np.empty((num_samples, batch_size, history_size, mcs_size), dtype=object)
    for b in range(batch_size):
        for idx in range(history_size):
            for idy in range(mcs_size):
                prev_sched_event = last_history_events[b, idx, idy]
                for idz in range(num_samples):
                    pred_sched_dtime = cp_dtime[idz, b, idx, idy]
                    pred_sched_len = int(cp_len[idz, b, idx, idy])

                    pred_segment_time_since_start = (prev_sched_event['time_since_start'] + pred_sched_dtime) % (
                        max_num_frames * num_slots_per_frame * slots_duration_ms
                    )
                    pred_segment_slot = (
                        prev_sched_event['slot'] + pred_sched_dtime / slots_duration_ms
                    ) % (num_slots_per_frame)
                    pred_timestamp = (
                        np.float64(prev_sched_event['timestamp'])
                        + np.float64(pred_sched_dtime / 1000.0)
                    )

                    predictions[idz, b, idx, idy] = {
                        'idx_event': -1,
                        'type_event': segment_num + 1,
                        'slot': pred_segment_slot,
                        'len': pred_sched_len,
                        'mcs_index': mcs_index[b, idy],
                        'mretx': 0 if exclude_link_quality else -1,
                        'rfailed': 0 if exclude_link_quality else -1,
                        'num_rbs': -1,
                        'num_symbols': 3,  # FIXME
                        'time_since_start': pred_segment_time_since_start,
                        'time_since_last_event': pred_sched_dtime,
                        'timestamp': pred_timestamp
                    }
                    len_bytes[idz, b, idx, idy] = pred_sched_len

    # no_scheduling_mask was [batch_size, history_size, mcs_size], we want to repeat it
    # to shape [num_samples, batch_size, history_size, mcs_size]
    no_scheduling_mask = np.repeat(no_scheduling_mask[np.newaxis, ...], num_samples, axis=0)

    return predictions, no_scheduling_mask, len_bytes


def predict_retx(retx_runner, retx_history_sequence, mcs_index: np.ndarray, num_rbs: np.ndarray, exp_config) -> np.ndarray:
    """
    Extended to batch dimension:
      retx_history_sequence: [batch_size, retx_history_size, mcs_size, retx_sequence_length]
      mcs_index: [batch_size, mcs_size]
      num_rbs: [num_sched_samples, batch_size, retx_history_size, mcs_size]
    Returns:
      predictions_retx: [num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size]
      predictions_rfailed: same shape
    """

    # Some shape checks
    batch_size = retx_history_sequence.shape[0]
    retx_history_size = retx_history_sequence.shape[1]
    mcs_size = retx_history_sequence.shape[2]
    history_sequence_length = retx_history_sequence.shape[3]

    # mcs_index: [batch_size, mcs_size]
    # num_rbs: [batch_size, num_sched_samples, retx_history_size, mcs_size]
    assert mcs_index.shape[0] == batch_size
    assert mcs_index.shape[1] == mcs_size
    assert num_rbs.shape[1] == batch_size
    assert num_rbs.shape[2] == retx_history_size
    assert num_rbs.shape[3] == mcs_size

    num_sched_samples = num_rbs.shape[0]

    # Build the conditional retx history: shape [batch_size, num_sched_samples, retx_history_size, mcs_size, history_sequence_length+1]
    cond_retx_history = np.empty(
        (batch_size, num_sched_samples, retx_history_size, mcs_size, history_sequence_length+1),
        dtype=object
    )

    for b in range(batch_size):
        for idx in range(retx_history_size):
            for idy in range(num_sched_samples):
                for idz in range(mcs_size):
                    new_seq = np.append(
                        copy.deepcopy(retx_history_sequence[b, idx, idz, :]),
                        {
                            'idx_event': 0,
                            'type_event': 0,  # block attempt
                            'timestamp': 0,
                            'time_since_start': 0,
                            'time_since_last_event': 0,
                            'rfailed': 0,
                            'mretx': 0,
                            'mcs_index': mcs_index[b, idz],
                            'num_rbs': num_rbs[idy, b, idx, idz],
                        }
                    )
                    for pos in range(history_sequence_length+1):
                        new_seq[pos]['idx_event'] = pos
                    cond_retx_history[b, idy, idx, idz, :] = new_seq

    # Flatten:
    # [batch_size, num_sched_samples, retx_history_size, mcs_size, L+1]
    # -> [batch_size*num_sched_samples*retx_history_size*mcs_size, L+1]
    cond_retx_history_reshaped = cond_retx_history.reshape(-1, history_sequence_length+1)

    # Run
    result = retx_runner.run(
        batch_size=cond_retx_history_reshaped.shape[0],
        source_data=cond_retx_history_reshaped,
        data_specs={
            "num_event_types": 2,  # FIXME
            "pad_token_id": 2,    # FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )

    p_retx = []
    p_rfailed = []
    for batch_out in result['pred']:
        p_retx.append(batch_out[0])
        p_rfailed.append(batch_out[1])

    cp_retx = np.concatenate(p_retx, axis=1)       # shape: [num_retx_samples, N]
    cp_rfailed = np.concatenate(p_rfailed, axis=1) # shape: [num_retx_samples, N]
    # N = batch_size * num_sched_samples * retx_history_size * mcs_size

    cp_retx = cp_retx[:, :, 0]    # [num_retx_samples, N]
    cp_rfailed = cp_rfailed[:, :, 0]

    num_retx_samples = cp_retx.shape[0]

    # Un-flatten to [num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size]
    cp_retx = cp_retx.reshape(num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size)
    cp_rfailed = cp_rfailed.reshape(num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size)

    return cp_retx, cp_rfailed


def predict_mcs(mcs_runner, mcs_history_sequence, mcs_eval_interval_ms) -> np.ndarray:
    """
    Extended to batch dimension:
      mcs_history_sequence: [batch_size, upd_mcs_history_size, mcs_size, history_sequence_length]
    Returns:
      predictions_mcs: [num_mcs_samples, batch_size, upd_mcs_history_size, mcs_size]
      next_mcs_eval_ts: [batch_size, upd_mcs_history_size, mcs_size]
    """

    batch_size = mcs_history_sequence.shape[0]
    upd_mcs_history_size = mcs_history_sequence.shape[1]
    mcs_size = mcs_history_sequence.shape[2]
    history_sequence_length = mcs_history_sequence.shape[3]

    # We find next_mcs_eval_ts for each (b, y, z)
    next_mcs_eval_ts = np.empty((batch_size, upd_mcs_history_size, mcs_size), dtype=object)
    mcs_history = np.empty((batch_size, upd_mcs_history_size, mcs_size, history_sequence_length+1), dtype=object)

    for b in range(batch_size):
        for idy in range(upd_mcs_history_size):
            for idz in range(mcs_size):
                next_mcs_eval_ts[b, idy, idz] = find_next_mcs_eval_ts(
                    mcs_history_sequence[b, idy, idz], mcs_eval_interval_ms
                )
                new_seq = np.append(
                    copy.deepcopy(mcs_history_sequence[b, idy, idz, :]),
                    {
                        'idx_event': 0,
                        'type_event': 1,  # MCS event
                        'timestamp': 0,
                        'time_since_start': 0,
                        'time_since_last_event': 0,
                        'mcs_index': 0,
                        'rfailed': RFAILED_PADDING,
                        'mretx': MRETX_PADDING,
                        'num_rbs': NUM_RBS_PADDING
                    }
                )
                for pos in range(history_sequence_length+1):
                    new_seq[pos]['idx_event'] = pos
                mcs_history[b, idy, idz, :] = new_seq

    # Flatten:
    # [batch_size, upd_mcs_history_size, mcs_size, L+1] -> [N, L+1]
    mcs_history_reshaped = mcs_history.reshape(-1, history_sequence_length+1)

    # Run
    result = mcs_runner.run(
        batch_size=mcs_history_reshaped.shape[0],
        source_data=mcs_history_reshaped,
        data_specs={
            "num_event_types": 2,  # FIXME
            "pad_token_id": 2,    # FIXME
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )

    p_mcs = []
    for batch_out in result['pred']:
        p_mcs.append(batch_out[0])

    cp_mcs = np.concatenate(p_mcs, axis=1)  # shape: [num_mcs_samples, N]
    cp_mcs = cp_mcs[:, :, 0]                # [num_mcs_samples, N]
    num_mcs_samples = cp_mcs.shape[0]

    # Un-flatten to [num_mcs_samples, batch_size, upd_mcs_history_size, mcs_size]
    cp_mcs = cp_mcs.reshape(num_mcs_samples, batch_size, upd_mcs_history_size, mcs_size)

    return cp_mcs, next_mcs_eval_ts


def predict_arrival(arrival_history_sequence, arrival_runner, exp_config) -> np.ndarray:
    """
    Extended to batch dimension:
      arrival_history_sequence: [batch_size, history_size, sequence_length]
    Returns:
      predictions: [num_samples, batch_size, history_size]
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']
    total_prbs_num = exp_config['total_prbs_num']
    symbols_per_slot = exp_config['symbols_per_slot']
    scheduling_map_num_integers = exp_config['scheduling_map_num_integers']
    max_num_frames = exp_config['max_num_frames']
    scheduling_time_ahead_ms = exp_config['scheduling_time_ahead_ms']
    max_harq_attempts = exp_config['max_harq_attempts']

    batch_size = arrival_history_sequence.shape[0]
    history_size = arrival_history_sequence.shape[1]
    history_sequence_length = arrival_history_sequence.shape[2]

    # Build shape [batch_size, history_size, L+1]
    arrival_history = np.empty((batch_size, history_size, history_sequence_length+1), dtype=object)
    for b in range(batch_size):
        for idx in range(history_size):
            new_seq = np.append(
                copy.deepcopy(arrival_history_sequence[b, idx, :]),
                {
                    'idx_event': history_sequence_length,
                    'type_event': 0,
                    'timestamp': 0,
                    'time_since_start': 0,
                    'time_since_last_event': 0,
                }
            )
            for pos in range(history_sequence_length+1):
                new_seq[pos]['idx_event'] = pos
            arrival_history[b, idx, :] = new_seq

    logger.info(f"packet arrival event prediction with sequence of shape: {arrival_history.shape}")

    # Flatten: [batch_size, history_size, L+1] -> [batch_size*history_size, L+1]
    arrival_history_reshaped = arrival_history.reshape(-1, history_sequence_length+1)

    # Run
    result = arrival_runner.run(
        batch_size=arrival_history_reshaped.shape[0],
        source_data=arrival_history_reshaped,
        data_specs={
            "num_event_types": 10000,  # a very large number
            "pad_token_id": 10000,     # a very large number
            "padding_strategy": 'do_not_pad'
        },
        return_predictions=True
    )

    p_dtime = []
    p_event_type = []
    for batch_out in result['pred']:
        p_dtime.append(batch_out[0])
        p_event_type.append(batch_out[1])

    cp_dtime = np.concatenate(p_dtime, axis=1)         # [num_samples, N]
    cp_event_type = np.concatenate(p_event_type, axis=1)# [num_samples, N]
    # N = batch_size*history_size

    cp_dtime = cp_dtime[:, :, 0]        # [num_samples, N]
    cp_event_type = cp_event_type[:, :, 0]

    num_samples = cp_dtime.shape[0]

    # Un-flatten back to [num_samples, batch_size, history_size]
    cp_dtime = cp_dtime.reshape(num_samples, batch_size, history_size)
    cp_event_type = cp_event_type.reshape(num_samples, batch_size, history_size)

    # Prepare predictions
    # last_history_events: [batch_size, history_size] (the last event of each sequence)
    last_history_events = arrival_history_sequence[:, :, -1]
    predictions = np.empty((num_samples, batch_size, history_size), dtype=object)
    for b in range(batch_size):
        for idx in range(history_size):
            prev_arrival_event = last_history_events[b, idx]
            for idy in range(num_samples):
                pred_arrival_dtime = cp_dtime[idy, b, idx]
                pred_arrival_len = int(cp_event_type[idy, b, idx])

                pred_time_since_start = (
                    prev_arrival_event['time_since_start'] + pred_arrival_dtime
                ) % (max_num_frames * num_slots_per_frame * slots_duration_ms)
                pred_timestap = prev_arrival_event['timestamp'] + pred_arrival_dtime / 1000.0
                predictions[idy, b, idx] = {
                    'idx_event': history_sequence_length,
                    'type_event': pred_arrival_len,
                    'time_since_start': pred_time_since_start,
                    'time_since_last_event': pred_arrival_dtime,
                    'timestamp': pred_timestap
                }
    return predictions


def calc_num_rbs(mcs_index: np.ndarray, len_bytes: np.ndarray):
    """
    Extended to batch dimension:
      mcs_index: [batch_size, mcs_size]
      len_bytes: [num_sched_samples, batch_size, history_size, mcs_size]
    Returns:
      num_rbs: [num_sched_samples, batch_size, history_size, mcs_size]
    """

    # Typically you might not need batch_size on mcs_index if your logic differs,
    # but here's how you'd handle it if your code needs it:
    num_sched_samples = len_bytes.shape[0]
    batch_size = len_bytes.shape[1]
    history_size = len_bytes.shape[2]
    mcs_size = len_bytes.shape[3]

    num_rbs = np.zeros((num_sched_samples, batch_size, history_size, mcs_size), dtype=int)

    for s in range(num_sched_samples):
        for b in range(batch_size):
            for h in range(history_size):
                for m in range(mcs_size):
                    if len_bytes[s, b, h, m] <= 20:
                        num_rbs[s, b, h, m] = 5
                    else:
                        num_rbs[s, b, h, m] = 26

    return num_rbs


def find_next_mcs_eval_ts(mcs_history_sequence, mcs_eval_interval_ms):
    """
    Single-sequence helper. 
    mcs_history_sequence: shape [mcs_sequence_length]
    """
    for idx in range(mcs_history_sequence.shape[0] - 1, -1, -1):
        if int(mcs_history_sequence[idx]['type_event']) == 1:
            return mcs_history_sequence[idx]['timestamp'] + (mcs_eval_interval_ms / 1000.0)
    logger.error("mcs_eval_is_not_needed: no MCS decision event found in the history sequence")
    return None


def segment_is_not_needed(sched_history_sequence):
    """
    Single-sequence helper. 
    sched_history_sequence: shape [sched_sequence_length]
    """
    departed_bytes = 0
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


def project_segment_predictions(
    sched_predictions, predictions_retx, predictions_rfailed, num_rbs, no_sched_mask
):
    """
    Extended to batch dimension.  Be mindful of shapes:
      sched_predictions: [num_sched_samples, batch_size, sched_history_size, mcs_size]
      predictions_retx: [num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size]
      predictions_rfailed: same shape as predictions_retx
      num_rbs: [num_sched_samples, batch_size, sched_history_size, mcs_size]
      no_sched_mask: [num_sched_samples, batch_size, sched_history_size, mcs_size]

    Returns:
      num_rfailed, num_mretx, len_bytes,
      segment_predictions: [num_retx_samples*num_sched_samples, batch_size, retx_history_size, mcs_size]
      no_sched_mask: [num_retx_samples*num_sched_samples, batch_size, retx_history_size, mcs_size]
    """

    num_retx_samples = predictions_retx.shape[0]
    batch_size = predictions_retx.shape[1]
    num_sched_samples = sched_predictions.shape[0]
    retx_history_size = predictions_retx.shape[3]
    mcs_size = sched_predictions.shape[3]

    num_rfailed_acc, num_mretx_acc, len_bytes_acc = 0, 0, 0

    # We'll build segment_predictions in shape 
    segment_predictions = np.empty(
        (num_retx_samples, num_sched_samples, batch_size, retx_history_size, mcs_size),
        dtype=object
    )

    for i_retx in range(num_retx_samples):
        for i_sched in range(num_sched_samples):
            for b in range(batch_size):
                for h in range(retx_history_size):
                    for m in range(mcs_size):
                        # copy the scheduling event fields
                        seg_copy = copy.deepcopy(sched_predictions[i_sched, b, h, m])
                        seg_copy['mretx'] = predictions_retx[i_retx, b, i_sched, h, m]
                        seg_copy['rfailed'] = predictions_rfailed[i_retx, b, i_sched, h, m]
                        seg_copy['num_rbs'] = num_rbs[i_sched, b, h, m]

                        num_rfailed_acc += int(seg_copy['rfailed'])
                        num_mretx_acc += int(seg_copy['mretx'] > 0)
                        len_bytes_acc += int(seg_copy['len'])

                        segment_predictions[i_retx, i_sched, b, h, m] = seg_copy

    # Expand no_sched_mask from [num_sched_samples, batch_size, sched_history_size, mcs_size]
    # to [num_retx_samples, num_sched_samples, batch_size, retx_history_size, mcs_size].
    # However, note that retx_history_size == sched_history_size if you constructed data that way.
    no_sched_mask_expanded = np.repeat(
        no_sched_mask[np.newaxis, ...],
        num_retx_samples,
        axis=0
    )  # [num_retx_samples, num_sched_samples, batch_size, sched_history_size, mcs_size]

    # Flatten dimension 0 & 1 so that shape becomes
    # [num_retx_samples * num_sched_samples, batch_size, retx_history_size, mcs_size]
    segment_predictions = segment_predictions.reshape(
        num_retx_samples * num_sched_samples,
        batch_size,
        retx_history_size,
        mcs_size
    )
    no_sched_mask_expanded = no_sched_mask_expanded.reshape(
        num_retx_samples * num_sched_samples,
        batch_size,
        retx_history_size,
        mcs_size
    )

    return (
        num_rfailed_acc,
        num_mretx_acc,
        len_bytes_acc,
        segment_predictions,
        no_sched_mask_expanded
    )


def append_arrival_predictions_to_arrival_history(arrival_predictions, arrival_history_sequence):
    """
    Extended to batch dimension:
      arrival_predictions: [num_arrival_samples, batch_size, arrival_history_size]
      arrival_history_sequence: [batch_size, arrival_history_size, arrival_sequence_length]

    Returns:
      upd_arrival_history: [num_arrival_samples*arrival_history_size, batch_size, arrival_sequence_length]
    """

    num_arrival_samples = arrival_predictions.shape[0]
    batch_size = arrival_predictions.shape[1]
    arrival_history_size = arrival_predictions.shape[2]
    arrival_sequence_length = arrival_history_sequence.shape[2]

    # We'll build shape [batch_size, num_arrival_samples, arrival_history_size, arrival_sequence_length]
    upd_arrival_history = np.empty(
        (batch_size, num_arrival_samples, arrival_history_size, arrival_sequence_length),
        dtype=object
    )

    for i_samp in range(num_arrival_samples):
        for b in range(batch_size):
            for i_hist in range(arrival_history_size):
                # shift out the oldest event and append the new event
                new_seq = np.append(
                    copy.deepcopy(arrival_history_sequence[b, i_hist, 1:]),
                    copy.deepcopy(arrival_predictions[i_samp, b, i_hist])
                )
                for pos in range(arrival_sequence_length):
                    new_seq[pos]['idx_event'] = pos
                upd_arrival_history[b, i_samp, i_hist, :] = new_seq

    # Finally flatten the first two dims -> [batch_size, num_arrival_samples * arrival_history_size, arrival_sequence_length]
    upd_arrival_history = upd_arrival_history.reshape(batch_size, -1, arrival_sequence_length)

    logger.info(f"updated arrival history shape: {upd_arrival_history.shape}")
    return upd_arrival_history


def append_mcs_predictions_to_history(
    mcs_predictions, upd_mcs_history, upd_sched_history, upd_retx_history, next_mcs_eval_ts, exp_config
):
    """
    Extended to batch dimension:
      mcs_predictions: [num_mcs_samples, batch_size, upd_mcs_history_size, mcs_size]
      upd_mcs_history: [batch_size, upd_mcs_history_size, mcs_size, mcs_sequence_length]
      upd_sched_history: [batch_size, upd_sched_history_size, mcs_size, sched_sequence_length]
      upd_retx_history: [batch_size, upd_retx_history_size, mcs_size, retx_sequence_length]
      next_mcs_eval_ts: [batch_size, upd_mcs_history_size, mcs_size]

    Returns:
      upd2_sched_history: [batch_size, upd_mcs_history_size, num_mcs_samples*mcs_size, sched_sequence_length]
      upd2_retx_history, upd2_mcs_history, upd_mcs_index
    """

    max_num_frames = exp_config['max_num_frames']
    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    num_mcs_samples = mcs_predictions.shape[0]
    batch_size = mcs_predictions.shape[1]
    upd_mcs_history_size = mcs_predictions.shape[2]
    mcs_size = mcs_predictions.shape[3]

    mcs_sequence_length = upd_mcs_history.shape[3]
    sched_sequence_length = upd_sched_history.shape[3]
    retx_sequence_length = upd_retx_history.shape[3]

    # Build shape [batch_size, upd_mcs_history_size, num_mcs_samples, mcs_size, mcs_sequence_length]
    upd2_mcs_history = np.empty(
        (batch_size, upd_mcs_history_size, num_mcs_samples, mcs_size, mcs_sequence_length), dtype=object
    )
    upd_mcs_index = np.empty((batch_size, num_mcs_samples, mcs_size), dtype=int)

    for b in range(batch_size):
        for i_samp in range(num_mcs_samples):
            for i_hist in range(upd_mcs_history_size):
                for m in range(mcs_size):
                    prev_mcs_event = upd_mcs_history[b, i_hist, m, -1]
                    pred_timestamp = next_mcs_eval_ts[b, i_hist, m]
                    pred_dtime_ms = (pred_timestamp - prev_mcs_event['timestamp']) * 1000.0

                    pred_time_since_start = (
                        prev_mcs_event['time_since_start'] + pred_dtime_ms
                    ) % (max_num_frames * num_slots_per_frame * slots_duration_ms)

                    new_seq = np.append(
                        copy.deepcopy(upd_mcs_history[b, i_hist, m, 1:]),
                        {
                            'idx_event': mcs_sequence_length,
                            'type_event': 1,  # MCS event
                            'timestamp': pred_timestamp,
                            'time_since_start': pred_time_since_start,
                            'time_since_last_event': pred_dtime_ms,
                            'mcs_index': int(mcs_predictions[i_samp, b, i_hist, m]),
                            'rfailed': RFAILED_PADDING,
                            'mretx': MRETX_PADDING,
                            'num_rbs': NUM_RBS_PADDING
                        }
                    )
                    for pos in range(mcs_sequence_length):
                        new_seq[pos]['idx_event'] = pos
                    upd2_mcs_history[b, i_hist, i_samp, m, :] = new_seq
                    upd_mcs_index[b, i_samp, m] = int(mcs_predictions[i_samp, b, i_hist, m])

    # Now we want to repeat sched_history, retx_history for each MCS sample
    # sched_history shape: [batch_size, sched_history_size, mcs_size, sched_sequence_length]
    upd2_sched_history = np.repeat(
        upd_sched_history[:, :, np.newaxis, ...],
        num_mcs_samples,
        axis=2
    )  # shape [batch_size, sched_history_size, num_mcs_samples, mcs_size, sched_sequence_length]
    upd2_retx_history = np.repeat(
        upd_retx_history[:, :, np.newaxis, ...],
        num_mcs_samples,
        axis=2
    )  # shape [batch_size, retx_history_size, num_mcs_samples, mcs_size, retx_sequence_length]

    # Reshape them to keep a single dimension for (num_mcs_samples * mcs_size)
    # so final shape: [batch_size, sched_history_size, num_mcs_samples*mcs_size, sched_sequence_length]
    # but let's carefully do it:
    bsz, shsz, nms, mm, slen = upd2_sched_history.shape
    upd2_sched_history = upd2_sched_history.reshape(
        bsz,
        shsz,
        nms * mm,
        slen
    )
    bsz, rhsz, nms, mm, rlen = upd2_retx_history.shape
    upd2_retx_history = upd2_retx_history.reshape(
        bsz,
        rhsz,
        nms * mm,
        rlen
    )

    # Finally do the same for mcs_history
    bsz, mcs_hist_size, nms, mm, mlen = upd2_mcs_history.shape
    upd2_mcs_history = upd2_mcs_history.reshape(
        bsz,
        mcs_hist_size,
        nms * mm,
        mlen
    )

    upd_mcs_index = upd_mcs_index.reshape(batch_size, num_mcs_samples * mcs_size)

    logger.info(f"updated scheduling history shape: {upd2_sched_history.shape}")
    logger.info(f"updated retx history shape: {upd2_retx_history.shape}")
    logger.info(f"updated mcs history shape: {upd2_mcs_history.shape}")
    logger.info(f"updated mcs index shape: {upd_mcs_index.shape}")

    return upd2_sched_history, upd2_retx_history, upd2_mcs_history, upd_mcs_index


def append_arrival_predictions_to_history(
    upd_arrival_history,
    upd2_sched_history,
    upd2_retx_history,
    upd2_mcs_history,
    upd_mcs_index,
    exp_config
):
    """
    Extended to batch dimension.

    Shapes:
      upd_arrival_history:  [batch_size, upd_arrival_history_size, arrival_sequence_length]
      upd2_sched_history:   [batch_size, some_dim, mcs_size, sched_sequence_length]
      upd2_retx_history:    [batch_size, some_dim, mcs_size, retx_sequence_length]
      upd2_mcs_history:     [batch_size, some_dim, mcs_size, mcs_sequence_length]
      upd_mcs_index:        [batch_size, mcs_size]  (or [batch_size, 1])

    We want to expand the scheduling history to account for each new arrival event.
    Then, we repeat the retx- and mcs- histories as needed to match the new dimension.
    Finally, we flatten (upd_arrival_history_size*some_dim) into one dimension.

    Returns:
      upd3_sched_history:   [batch_size, upd_arrival_history_size*some_dim, mcs_size, sched_sequence_length]
      upd3_retx_history:    [batch_size, upd_arrival_history_size*some_dim, mcs_size, retx_sequence_length]
      upd3_mcs_history:     [batch_size, upd_arrival_history_size*some_dim, mcs_size, mcs_sequence_length]
    """

    slots_duration_ms = exp_config['slots_duration_ms']
    num_slots_per_frame = exp_config['slots_per_frame']

    # Unpack shapes for readability
    # upd_arrival_history -> (B, A, LA)
    B, A, LA = upd_arrival_history.shape
    
    # upd2_sched_history -> (B, S, M, LS)
    # retx -> (B, S, M, LR), mcs -> (B, S, M, LM)
    _, S, M, LS = upd2_sched_history.shape
    _, _, _, LR = upd2_retx_history.shape
    _, _, _, LM = upd2_mcs_history.shape

    # We will create an expanded scheduling history of shape [B, A, S, M, LS]
    # Then flatten A*S -> [A*S].
    upd3_sched_history_expanded = np.empty(
        (B, A, S, M, LS), dtype=object
    )

    # 1) Expand scheduling for each new arrival
    for b in range(B):
        for a_idx in range(A):
            for s_idx in range(S):
                for m_idx in range(M):
                    # Grab the newly predicted arrival event from the last position in upd_arrival_history
                    new_arrival_event = upd_arrival_history[b, a_idx, -1]
                    
                    # Grab the last scheduling event from upd2_sched_history
                    prev_sched_event = upd2_sched_history[b, s_idx, m_idx, -1]

                    # Compute time difference, etc.
                    pred_sched_dtime = (new_arrival_event['timestamp'] - prev_sched_event['timestamp']) * 1000.0
                    pred_segment_slot = (prev_sched_event['slot'] + (pred_sched_dtime / slots_duration_ms)) % num_slots_per_frame
                    
                    # Shift out the oldest event in scheduling and append new arrival event
                    new_sched_seq = np.append(
                        copy.deepcopy(upd2_sched_history[b, s_idx, m_idx, 1:]),
                        {
                            'idx_event': -1,  # will fix after insertion
                            'type_event': 0,  # arrival event
                            'slot': pred_segment_slot,
                            'len': int(new_arrival_event['type_event']),  # arrival "type_event" often used as size
                            'mcs_index': int(upd_mcs_index[b, m_idx]) if upd_mcs_index.ndim > 1 else 0,
                            'mretx': MRETX_PADDING,      # MRETX_PADDING
                            'rfailed': RFAILED_PADDING,    # RFAILED_PADDING
                            'num_rbs': NUM_RBS_PADDING,  # NUM_RBS_PADDING
                            'num_symbols': NUM_SYMBOLS_PADDING, # NUM_SYMBOLS_PADDING
                            'time_since_start': new_arrival_event['time_since_start'],
                            'time_since_last_event': pred_sched_dtime,
                            'timestamp': new_arrival_event['timestamp']
                        }
                    )
                    # Fix idx_event to be [0..LS-1]
                    for pos in range(LS):
                        new_sched_seq[pos]['idx_event'] = pos

                    upd3_sched_history_expanded[b, a_idx, s_idx, m_idx, :] = new_sched_seq

    # 2) Now we repeat the retx- and mcs- histories so they match dimension A in the 2nd axis
    #    Then flatten (A*S) in that axis.
    # shape before repeat: [B, S, M, LR] -> we want [B, A, S, M, LR]
    upd3_retx_history_expanded = np.repeat(
        upd2_retx_history[:, np.newaxis, ...],  # insert new axis for A
        A,
        axis=1
    )  # shape now: [B, A, S, M, LR]

    upd3_mcs_history_expanded = np.repeat(
        upd2_mcs_history[:, np.newaxis, ...], 
        A,
        axis=1
    )  # shape now: [B, A, S, M, LM]

    # 3) Flatten the second axis (A*S) so final shape is [B, A*S, M, LS], etc.
    upd3_sched_history = upd3_sched_history_expanded.reshape(B, A * S, M, LS)
    upd3_retx_history = upd3_retx_history_expanded.reshape(B, A * S, M, LR)
    upd3_mcs_history = upd3_mcs_history_expanded.reshape(B, A * S, M, LM)

    logger.info(f"updated scheduling history shape: {upd3_sched_history.shape}")
    logger.info(f"updated retx history shape: {upd3_retx_history.shape}")
    logger.info(f"updated mcs history shape: {upd3_mcs_history.shape}")

    return upd3_sched_history, upd3_retx_history, upd3_mcs_history


def append_segment_predictions_to_history(
    segment_predictions,
    sched_history_sequence,
    retx_history_sequence,
    mcs_history_sequence,
    no_sched_mask,
    filter_successful_attempts_for_mcs
):
    """
    Extended to batch dimension:
      segment_predictions: [segment_pred_size, batch_size, retx_history_size, mcs_size]
      sched_history_sequence: [sched_history_size, mcs_size, sched_sequence_length] (original code)
         BUT now likely [batch_size, sched_history_size, mcs_size, sched_sequence_length]
      no_sched_mask: [segment_pred_size, batch_size, sched_history_size, mcs_size] in your original flow
      etc.
    Adjust carefully to keep consistent with your revised shapes.
    """

    # For brevity, here's how you'd do it if you keep your new shapes consistent
    # The main difference is that each dimension has an extra 'batch_size' now.

    segment_pred_size = segment_predictions.shape[0]
    batch_size = segment_predictions.shape[1]
    retx_history_size = segment_predictions.shape[2]
    mcs_history_size = mcs_history_sequence.shape[1]
    mcs_size = segment_predictions.shape[3]

    # If your existing histories have shape [batch_size, sched_history_size, mcs_size, sched_sequence_length], do:
    sched_history_size = sched_history_sequence.shape[1]
    sched_sequence_length = sched_history_sequence.shape[3]
    retx_sequence_length = retx_history_sequence.shape[3]
    mcs_sequence_length = mcs_history_sequence.shape[3]

    upd_sched_history = np.empty(
        (batch_size, segment_pred_size, sched_history_size, mcs_size, sched_sequence_length),
        dtype=object
    )
    upd_retx_history = np.empty(
        (batch_size, segment_pred_size, retx_history_size, mcs_size, retx_sequence_length),
        dtype=object
    )
    upd_mcs_history = np.empty(
        (batch_size, segment_pred_size, mcs_history_size, mcs_size, mcs_sequence_length),
        dtype=object
    )

    for b in range(batch_size):
        for idx in range(sched_history_size):
            for i_pred in range(segment_pred_size):
                for m in range(mcs_size):
                    if no_sched_mask[i_pred, b, idx, m]:
                        # Keep the latest event as is
                        upd_sched_history[b, i_pred, idx, m, :] = copy.deepcopy(sched_history_sequence[b, idx, m, :])
                        upd_retx_history[b, i_pred, idx, m, :] = copy.deepcopy(retx_history_sequence[b, idx, m, :])
                        upd_mcs_history[b, i_pred, idx, m, :] = copy.deepcopy(mcs_history_sequence[b, idx, m, :])
                    else:
                        # shift + append
                        new_sched = np.append(
                            copy.deepcopy(sched_history_sequence[b, idx, m, 1:]),
                            copy.deepcopy(segment_predictions[i_pred, b, idx, m])
                        )
                        for pos in range(sched_sequence_length):
                            new_sched[pos]['idx_event'] = pos
                        upd_sched_history[b, i_pred, idx, m, :] = new_sched

                        block_event = copy.deepcopy(segment_predictions[i_pred, b, idx, m])
                        block_event['type_event'] = 0
                        block_event['time_since_last_event'] = (
                            block_event['timestamp'] - retx_history_sequence[b, idx, m, -1]['timestamp']
                        ) * 1000.0
                        del block_event['len'], block_event['slot'], block_event['num_symbols']
                        new_retx = np.append(
                            copy.deepcopy(retx_history_sequence[b, idx, m, 1:]),
                            block_event
                        )
                        for pos in range(retx_sequence_length):
                            new_retx[pos]['idx_event'] = pos
                        upd_retx_history[b, i_pred, idx, m, :] = new_retx

                        if filter_successful_attempts_for_mcs and not (
                            block_event['rfailed'] > 0 or block_event['mretx'] > 0
                        ):
                            # just copy old
                            upd_mcs_history[b, i_pred, idx, m, :] = copy.deepcopy(mcs_history_sequence[b, idx, m, :])
                        else:
                            block_event_for_mcs = copy.deepcopy(block_event)
                            block_event_for_mcs['time_since_last_event'] = (
                                block_event_for_mcs['timestamp'] - mcs_history_sequence[b, idx, m, -1]['timestamp']
                            ) * 1000.0
                            new_mcs = np.append(
                                copy.deepcopy(mcs_history_sequence[b, idx, m, 1:]),
                                block_event_for_mcs
                            )
                            for pos in range(mcs_sequence_length):
                                new_mcs[pos]['idx_event'] = pos
                            upd_mcs_history[b, i_pred, idx, m, :] = new_mcs

    # Now flatten out the first two dims if needed:
    upd_sched_history = upd_sched_history.reshape(batch_size, -1, mcs_size, sched_sequence_length)
    upd_retx_history = upd_retx_history.reshape(batch_size, -1, mcs_size, retx_sequence_length)
    upd_mcs_history = upd_mcs_history.reshape(batch_size, -1, mcs_size, mcs_sequence_length)

    logger.info(f"updated sched history shape: {upd_sched_history.shape}")
    logger.info(f"updated retx history shape: {upd_retx_history.shape}")
    logger.info(f"updated mcs history shape: {upd_mcs_history.shape}")

    return upd_sched_history, upd_retx_history, upd_mcs_history


def reduce_history_dimension(
    sched_history_sequence,
    retx_history_sequence,
    mcs_history_sequence,
    history_dimension_limit,
    mcs_dimension_limit,
    mcs_index
):
    """
    Extended to batch dimension - now be aware that:
      sched_history_sequence: [batch_size, sH, mcs_size, sched_seq_length]
      ...
    If you truly want to reduce dimension across the entire batch, you might do so for each batch entry independently.
    Here is a simplistic approach: do nothing or do random picks per batch index. Adjust as needed.
    """

    # For a minimal example, we assume you want to reduce across the "sH" dimension (the 2nd dimension)
    # and possibly across the "mcs_size" dimension (the 3rd dimension).
    # You can do so per-batch or globally. Here's a simple global approach:

    batch_size = sched_history_sequence.shape[0]
    sH = sched_history_sequence.shape[1]
    current_mcs_size = sched_history_sequence.shape[2]

    if sH > history_dimension_limit:
        selected_indices = np.random.choice(sH, history_dimension_limit, replace=False)
        sched_history_sequence = sched_history_sequence[:, selected_indices, ...]
        retx_history_sequence = retx_history_sequence[:, selected_indices, ...]
        mcs_history_sequence = mcs_history_sequence[:, selected_indices, ...]

    if current_mcs_size > mcs_dimension_limit:
        selected_indices = np.random.choice(current_mcs_size, mcs_dimension_limit, replace=False)
        sched_history_sequence = sched_history_sequence[:, :, selected_indices, :]
        retx_history_sequence = retx_history_sequence[:, :, selected_indices, :]
        mcs_history_sequence = mcs_history_sequence[:, :, selected_indices, :]
        mcs_index = mcs_index[:, selected_indices]

    logger.info(f"reduced dim sched history sequences shape: {sched_history_sequence.shape}")
    logger.info(f"reduced dim retx history sequences shape: {retx_history_sequence.shape}")
    logger.info(f"reduced dim mcs history sequences shape: {mcs_history_sequence.shape}")
    logger.info(f"reduced dim mcs index shape: {mcs_index.shape}")

    return sched_history_sequence, retx_history_sequence, mcs_history_sequence, mcs_index



def next_packet_and_mcs_prediction(
    arrival_runner,
    mcs_runner,
    arrival_history_sequence,
    upd_sched_history,
    upd_retx_history,
    upd_mcs_history,
    mcs_eval_interval_ms,
    exp_config,
):
    """
    Extended to batch dimension.  The new shapes are:
      arrival_history_sequence: [batch_size, arrival_history_size, arrival_sequence_length]
      upd_sched_history: [batch_size, S, mcs_size, sched_sequence_length]
      upd_retx_history, upd_mcs_history: similarly
    Returns:
      upd3_sched_history, upd3_retx_history, upd3_mcs_history, upd_mcs_index
    """

    # 1) Predict arrival
    arrival_predictions = predict_arrival(
        arrival_history_sequence, arrival_runner, exp_config
    )
    # arrival_predictions: [num_arrival_samples, batch_size, arrival_history_size]

    # 2) Append arrival predictions to the arrival history
    upd_arrival_history = append_arrival_predictions_to_arrival_history(
        arrival_predictions, arrival_history_sequence
    )
    # shape: [num_arrival_samples*batch_size, arrival_history_size, arrival_sequence_length]

    # 3) Predict MCS
    mcs_predictions, next_mcs_eval_ts = predict_mcs(
        mcs_runner, upd_mcs_history, mcs_eval_interval_ms
    )
    # mcs_predictions: [num_mcs_samples, batch_size, mcs_history_size, mcs_size]

    # 4) Append MCS predictions to history
    upd2_sched_history, upd2_retx_history, upd2_mcs_history, upd_mcs_index = append_mcs_predictions_to_history(
        mcs_predictions, upd_mcs_history, upd_sched_history, upd_retx_history, next_mcs_eval_ts, exp_config
    )

    # 5) Append arrival predictions to the new history sequences
    upd3_sched_history, upd3_retx_history, upd3_mcs_history = append_arrival_predictions_to_history(
        upd_arrival_history,
        upd2_sched_history,
        upd2_retx_history,
        upd2_mcs_history,
        upd_mcs_index,
        exp_config
    )

    return upd3_sched_history, upd3_retx_history, upd3_mcs_history, upd_mcs_index


def sample_based_e2e_prediction_batch(
    data,
    arrival_runner,
    mcs_runner,
    retx_runner,
    sched_runner,
    exp_config,
    num_future_packet_predictions,
    mcs_eval_interval_ms=100,
    filter_successful_attempts_for_mcs=True,
    mcs_dimension_limit=20,
    history_dimension_limit=100,
    exclude_retx_predictions=True,
    max_num_segments=5
):
    """
    Example end-to-end pipeline with batch dimension = data_batch_size = 1 in your original code,
    but now you can pass multiple data entries in 'data'. Adjust how you handle data to ensure
    shapes are consistent: [batch_size, ...].
    """

    final_predictions = []

    # Suppose `data` is already shaped [batch_size] or similar. For demonstration, assume batch_size=1 for each data item.
    # You could unify multiple data items into arrays for arrival, scheduling, etc.
    # Here's a simple example of processing them in a loop. 
    # If you truly want a big batch, you'd need to stack data['arrival'] from all items, etc.

    # For demonstration, assume batch_size=1 to keep it close to your original.  
    # If you have bigger batch, adapt the shapes accordingly.

    batch_size = len(data)  # or however you measure your data batch
    logger.info(f"Running E2E prediction on batch_size={batch_size}")

    # Build arrays of shape [batch_size, ...]
    # This requires that each data[i]['arrival'] etc. are of the same shape so we can stack.
    # For demonstration, do:
    arrival_history_sequence = []
    sched_history_sequence = []
    mcs_history_sequence = []
    retx_history_sequence = []
    mcs_index_list = []

    for i_b in range(batch_size):
        arrival_history_sequence.append([data[i_b]['arrival']])    # shape [history_size, arrival_sequence_length]
        sched_history_sequence.append([[data[i_b]['scheduling']]])   # shape [segment_history_size, mcs_size, sched_sequence_length]
        mcs_history_sequence.append([[data[i_b]['mcs']]])            # shape [segment_history_size, mcs_size, mcs_sequence_length]
        retx_history_sequence.append([[data[i_b]['retx']]])          # shape [segment_history_size, mcs_size, retx_sequence_length]
        packet0_arrival = data[i_b]['scheduling'][-1]
        mcs_index_list.append([packet0_arrival['mcs_index']])    # shape [mcs_size=1]

    # Convert to numpy with an extra batch dim 
    arrival_history_sequence = np.array(arrival_history_sequence, dtype=object)  # shape [batch_size, history_size, arrival_sequence_length]
    sched_history_sequence = np.array(sched_history_sequence, dtype=object)      # shape [batch_size, segment_history_size, mcs_size, sched_sequence_length]
    mcs_history_sequence = np.array(mcs_history_sequence, dtype=object)          # shape [batch_size, segment_history_size, mcs_size, mcs_sequence_length]
    retx_history_sequence = np.array(retx_history_sequence, dtype=object)        # shape [batch_size, segment_history_size, mcs_size, retx_sequence_length]
    mcs_index = np.array(mcs_index_list, dtype=object)                           # shape [batch_size, 1]
    
    # run some checks
    assert mcs_history_sequence.shape[0] == batch_size
    assert sched_history_sequence.shape[0] == batch_size
    assert retx_history_sequence.shape[0] == batch_size
    assert arrival_history_sequence.shape[0] == batch_size
    assert mcs_index.shape[0] == batch_size
    assert sched_history_sequence.shape[3] > 0
    assert retx_history_sequence.shape[3] > 0
    assert mcs_history_sequence.shape[3] > 0

    for packet_num in range(num_future_packet_predictions):
        for segment_num in range(max_num_segments):
            # Possibly reduce dimension
            sched_history_sequence, retx_history_sequence, mcs_history_sequence, mcs_index = reduce_history_dimension(
                sched_history_sequence,
                retx_history_sequence,
                mcs_history_sequence,
                history_dimension_limit,
                mcs_dimension_limit,
                mcs_index
            )

            bsz = sched_history_sequence.shape[0]
            shsz = sched_history_sequence.shape[1]
            mcsize = sched_history_sequence.shape[2]

            logger.info(
                f"Predicting packet {packet_num}, segment {segment_num}, shapes (sched={sched_history_sequence.shape}, retx={retx_history_sequence.shape}, mcs={mcs_history_sequence.shape})"
            )

            # 1) predict scheduling
            sched_predictions, no_sched_mask, len_bytes = predict_scheduling(
                sched_runner,
                sched_history_sequence,
                segment_num,
                mcs_index,  # shape [batch_size, mcs_size]
                exclude_retx_predictions,
                exp_config
            )
            # sched_predictions: [num_sched_samples, batch_size, shsz, mcsize]
            num_sched_samples = sched_predictions.shape[0]

            # 2) calc num_rbs
            num_rbs = calc_num_rbs(mcs_index, len_bytes)
            logger.info(f"num_rbs shape: {num_rbs.shape}")

            if not exclude_retx_predictions:
                # 3) predict retx
                predictions_retx, predictions_rfailed = predict_retx(
                    retx_runner,
                    retx_history_sequence,
                    mcs_index,
                    num_rbs,
                    exp_config
                )
                # predictions_retx: [num_retx_samples, batch_size, num_sched_samples, retx_history_size, mcs_size]

                # 4) project segment predictions
                num_rfailed_acc, num_mretx_acc, len_bytes_acc, segment_predictions, no_sched_mask_expanded = \
                    project_segment_predictions(
                        sched_predictions, predictions_retx, predictions_rfailed, num_rbs, no_sched_mask
                    )
                # segment_predictions shape: [num_retx_samples*num_sched_samples, batch_size, retx_history_size, mcs_size]
                # no_sched_mask_expanded: same shape

            else:
                # Just set num_rbs
                for i_sch in range(num_sched_samples):
                    for b in range(bsz):
                        for i_hist in range(shsz):
                            for m in range(mcsize):
                                sched_predictions[i_sch, b, i_hist, m]['num_rbs'] = num_rbs[i_sch, b, i_hist, m]

                segment_predictions = sched_predictions
                # Reshape to match the "expanded" shape if you want to unify logic
                segment_predictions = segment_predictions.reshape(num_sched_samples, bsz, shsz, mcsize)
                no_sched_mask_expanded = no_sched_mask  # keep shapes consistent
                num_rfailed_acc, num_mretx_acc, len_bytes_acc = 0, 0, 0

            seg_pred_shape = segment_predictions.shape
            logger.info(f"segment predictions shape: {seg_pred_shape}")

            # Some stats
            total_num_projections = no_sched_mask_expanded.size
            no_sched_mask_sum = np.sum(no_sched_mask_expanded.astype(int))
            logger.info(
                f"Segment {segment_num}: total={total_num_projections}, completed={no_sched_mask_sum}, "
                f"rfailed={num_rfailed_acc}, mretx={num_mretx_acc}, avg_bytes={len_bytes_acc/max(1, total_num_projections)}"
            )

            # 5) append segment predictions
            upd_sched_history, upd_retx_history, upd_mcs_history = append_segment_predictions_to_history(
                segment_predictions,
                sched_history_sequence,
                retx_history_sequence,
                mcs_history_sequence,
                no_sched_mask_expanded,
                filter_successful_attempts_for_mcs
            )

            # Check if packet is done
            if no_sched_mask_sum / total_num_projections > 0.9:
                logger.info("No more segments to schedule.")
                break
            else:
                # keep the updated sequences
                sched_history_sequence = upd_sched_history
                retx_history_sequence = upd_retx_history
                mcs_history_sequence = upd_mcs_history


        # prepare the results
        result_history = copy.deepcopy(upd_sched_history).reshape(
            upd_sched_history.shape[0], upd_sched_history.shape[1]*upd_sched_history.shape[2], upd_sched_history.shape[3]
        )
        final_predictions.append(
            result_history[:,:,-(segment_num+1):]
        )
        if len(final_predictions) >= num_future_packet_predictions:
            break

        # 6) Next packet + next MCS index
        upd3_sched_history, upd3_retx_history, upd3_mcs_history, upd_mcs_index = next_packet_and_mcs_prediction(
            arrival_runner,
            mcs_runner,
            arrival_history_sequence,
            sched_history_sequence,
            retx_history_sequence,
            mcs_history_sequence,
            mcs_eval_interval_ms,
            exp_config
        )

        # update for next iteration
        sched_history_sequence = upd3_sched_history
        retx_history_sequence = upd3_retx_history
        mcs_history_sequence = upd3_mcs_history
        mcs_index = upd_mcs_index

    return final_predictions
