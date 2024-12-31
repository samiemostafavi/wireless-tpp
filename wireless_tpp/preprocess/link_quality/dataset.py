from typing import Dict

import numpy as np
from wireless_tpp.preprocess import BaseTPPDataset

from wireless_tpp.preprocess.data_collator import TPPDataCollator
from wireless_tpp.utils import py_assert, logger

class TPPDatasetLinkQuality(BaseTPPDataset):
    def __init__(self, data: Dict):
        self.data_dict = data
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']
        self.mcs_seqs = self.data_dict.get('mcs_seqs')
        self.num_rbs_seqs = self.data_dict.get('num_rbs_seqs')
        self.mretx_seqs = self.data_dict.get('mretx_seqs')
        self.rfailed_seqs = self.data_dict.get('rfailed_seqs')

    def __len__(self):
        """

        Returns: length of the dataset

        """
        py_assert(len(self.mcs_seqs) == len(self.time_seqs) and len(self.time_seqs) == len(self.type_seqs) and len(self.time_delta_seqs) == len(self.type_seqs),
                ValueError,
                f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
                f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}")


        return len(self.time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs and type_seqs element

        """
        return dict({ 
            'time_seqs': self.time_seqs[idx], 
            'time_delta_seqs': self.time_delta_seqs[idx],
            'type_seqs': self.type_seqs[idx],
            'mcs_seqs': self.mcs_seqs[idx],
            'num_rbs_seqs': self.num_rbs_seqs[idx],
            'mretx_seqs': self.mretx_seqs[idx],
            'rfailed_seqs': self.rfailed_seqs[idx]
        })


    def to_tf_dataset(self, data_collator: TPPDataCollator, **kwargs):
        logger.error("Tensorflow is not available.")
        return None

    def get_stats(self, **kwargs):
        
        inp_type = kwargs.get('inp_type', 'time_delta_seqs')
        mcs_or_segment = kwargs.get('mcs_or_segment', False) # True: packet, False: segment

        if inp_type == 'time_delta_seqs': 
            val_seqs = self.time_delta_seqs
        elif inp_type == 'mcs_seqs':
            val_seqs = self.mcs_seqs
        elif inp_type == 'mretx_seqs':
            val_seqs = self.mretx_seqs
        elif inp_type == 'rfailed_seqs':
            val_seqs = self.rfailed_seqs
        elif inp_type == 'num_rbs_seqs':
            val_seqs = self.num_rbs_seqs
        else:
            raise ValueError(f"Invalid input type: {inp_type}")

        # then if (mark <= num_event_types_segment_only) we will consider it as segment event type
        # if (mark > num_event_types_segment_only) we will consider it as packet arrival event type
        x_bar, s_2_x, xp_bar, s_2_xp, n = 0., 0., 0, 0., 0
        min_val, max_val = np.inf, -np.inf
        min_mark, max_mark = np.inf, -np.inf
        for inp_vals, inp_marks in zip(val_seqs, self.type_seqs):
            filtered_vals = []
            filtered_marks = []
            for val, mark in zip(inp_vals, inp_marks):
                if mcs_or_segment:
                    # look for mcs events
                    if mark == 1:
                        filtered_vals.append(val)
                        filtered_marks.append(mark)
                else:
                    # look for segment events
                    if mark == 0:
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

        logger.info(f"{'MCS' if mcs_or_segment else 'Segment'} events {inp_type} mean and variance:  {x_bar}, {(s_2_x ** 0.5)}")
        logger.info(f'min {inp_type}: {min_val}')
        logger.info(f'max {inp_type}: {max_val}')

        logger.info(f"{'MCS' if mcs_or_segment else 'Segment'} events event types mean and variance: {xp_bar}, {(s_2_xp ** 0.5)}")
        logger.info(f'min_mark: {min_mark}')
        logger.info(f'max_mark: {max_mark}')

        return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), min_val, max_val, min_mark, max_mark
