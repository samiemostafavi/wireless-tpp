import math
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import py_assert, is_tf_available, logger

class TPPDatasetScheduling(Dataset):
    def __init__(self, data: Dict):
        self.data_dict = data

        # we incorporate the following additional keys in the source_data
        # 'len' 'mcs_index' 'mac_retx' 'rlc_failed' 'num_rbs'
        self.slot_seqs = self.data_dict['slot_seqs']
        self.len_seqs = self.data_dict['len_seqs']
        self.mcs_seqs = self.data_dict['mcs_seqs']
        self.mac_retx_seqs = self.data_dict['mac_retx_seqs']
        self.rlc_failed_seqs = self.data_dict['rlc_failed_seqs']
        self.num_rbs_seqs = self.data_dict['num_rbs_seqs']

        # default keys
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']

    def __len__(self):
        """

        Returns: length of the dataset

        """
        py_assert(
            len(self.slot_seqs) == len(self.time_seqs) and
            len(self.len_seqs) == len(self.time_seqs) and
            len(self.mcs_seqs) == len(self.time_seqs) and
            len(self.mac_retx_seqs) == len(self.time_seqs) and
            len(self.rlc_failed_seqs) == len(self.time_seqs) and 
            len(self.num_rbs_seqs) == len(self.time_seqs) and 
            len(self.time_seqs) == len(self.type_seqs) and 
            len(self.time_delta_seqs) == len(self.type_seqs),
            ValueError,
            f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
            f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}"
        )
        return len(self.time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs and type_seqs element

        """
        return dict(
            {
                'slot_seqs': self.slot_seqs[idx],
                'len_seqs': self.len_seqs[idx],
                'mcs_seqs': self.mcs_seqs[idx],
                'mac_retx_seqs': self.mac_retx_seqs[idx],
                'rlc_failed_seqs': self.rlc_failed_seqs[idx],
                'num_rbs_seqs': self.num_rbs_seqs[idx],
                'time_seqs': self.time_seqs[idx], 
                'time_delta_seqs': self.time_delta_seqs[idx],
                'type_seqs': self.type_seqs[idx]
            }
        )

    def to_tf_dataset(self, data_collator: TPPDataCollator, **kwargs):
        logger.error("Tensorflow is not available.")
        return None

    def get_stats(self, **kwargs):
        
        inp_type = kwargs.get('inp_type', 'time_delta_seqs')
        packet_or_segment = kwargs.get('packet_or_segment', False) # True: packet, False: segment

        if inp_type == 'time_delta_seqs': 
            val_seqs = self.time_delta_seqs
        elif inp_type == 'slot_seqs': 
            val_seqs = self.slot_seqs
        elif inp_type == 'len_seqs': 
            val_seqs = self.len_seqs
        elif inp_type == 'mcs_seqs':
            val_seqs = self.mcs_seqs
        elif inp_type == 'mac_retx_seqs':
            val_seqs = self.mac_retx_seqs
        elif inp_type == 'rlc_failed_seqs':
            val_seqs = self.rlc_failed_seqs
        elif inp_type == 'num_rbs_seqs':
            val_seqs = self.num_rbs_seqs
        else:
            raise ValueError(f"Invalid input type: {inp_type}")

        # as we won't predict packet arrival times, we will exclude them from statistics
        num_event_types = kwargs.get('num_event_types', np.inf) # for example 6: 5 segments, 1 packet arrival
    

        # then if (mark <= num_event_types_segment_only) we will consider it as segment event type
        # if (mark > num_event_types_segment_only) we will consider it as packet arrival event type
        x_bar, s_2_x, xp_bar, s_2_xp, n = 0., 0., 0, 0., 0
        min_val, max_val = np.inf, -np.inf
        min_mark, max_mark = np.inf, -np.inf
        for inp_vals, inp_marks in zip(val_seqs, self.type_seqs):
            filtered_vals = []
            filtered_marks = []
            for val, mark in zip(inp_vals, inp_marks):
                if packet_or_segment:
                    # look for packet events
                    if mark == 0:
                        filtered_vals.append(val)
                        filtered_marks.append(mark)
                else:
                    # look for segment events
                    if mark > 0:
                        filtered_vals.append(val)
                        filtered_marks.append(mark)
            vals = np.array(filtered_vals)
            marks = np.array(filtered_marks)
            min_val = min(min_val, vals.min())
            max_val = max(max_val, vals.max())
            min_mark = min(min_mark, marks.min())
            max_mark = max(max_mark, marks.max())
            yp_bar = marks.mean()
            s_2_yp = marks.var()
            y_bar = vals.mean()
            s_2_y = vals.var()
            m = vals.shape[0]
            n += m
            # Formulat taken from https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            s_2_x = (((n - 1) * s_2_x + (m - 1) * s_2_y) / (n + m - 1)) + (
                        (n * m * ((x_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1)))
            x_bar = (n * x_bar + m * y_bar) / (n + m)

            s_2_xp = (((n - 1) * s_2_xp + (m - 1) * s_2_yp) / (n + m - 1)) + (
                        (n * m * ((xp_bar - yp_bar) ** 2)) / ((n + m) * (n + m - 1)))
            xp_bar = (n * xp_bar + m * yp_bar) / (n + m)

        logger.info(f"{'Packet' if packet_or_segment else 'Segment'} events {inp_type} mean and variance:  {x_bar}, {(s_2_x ** 0.5)}")
        logger.info(f'min {inp_type}: {min_val}')
        logger.info(f'max {inp_type}: {max_val}')

        logger.info(f"{'Packet' if packet_or_segment else 'Segment'} events event types mean and variance: {xp_bar}, {(s_2_xp ** 0.5)}")
        logger.info(f'min_mark: {min_mark}')
        logger.info(f'max_mark: {max_mark}')

        return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), min_val, max_val, min_mark, max_mark

def get_data_loader(dataset: TPPDatasetScheduling, backend: str, tokenizer: EventTokenizer, **kwargs):
    use_torch = backend == 'torch'

    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy

    if use_torch:
        data_collator = TPPDataCollator(tokenizer=tokenizer,
                                        return_tensors='pt',
                                        max_length=tokenizer.model_max_length,
                                        padding=padding,
                                        truncation=truncation)

        return DataLoader(dataset,
                          collate_fn=data_collator,
                          **kwargs)
    else:
        # we pass to placeholders
        data_collator = TPPDataCollator(tokenizer=tokenizer,
                                        return_tensors='np',
                                        max_length=tokenizer.model_max_length,
                                        padding=padding,
                                        truncation=truncation)

        return dataset.to_tf_dataset(data_collator, **kwargs)
