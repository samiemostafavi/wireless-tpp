from typing import Dict

import numpy as np
from wireless_tpp.preprocess import BaseTPPDataset

from wireless_tpp.preprocess.data_collator import TPPDataCollator
from wireless_tpp.utils import py_assert, logger


class TPPDatasetPacketArrival(BaseTPPDataset):
    def __init__(self, data: Dict):
        self.data_dict = data
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']

    def __len__(self):
        """

        Returns: length of the dataset

        """

        py_assert(len(self.time_seqs) == len(self.type_seqs) and len(self.time_delta_seqs) == len(self.type_seqs),
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
        return dict({'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                     'type_seqs': self.type_seqs[idx]})

    def to_tf_dataset(self, data_collator: TPPDataCollator, **kwargs):
        logger.error("Tensorflow is not available.")
        return None

    def get_dt_stats(self):
        x_bar, s_2_x, xp_bar, s_2_xp, n = 0., 0., 0, 0., 0
        min_dt, max_dt = np.inf, -np.inf
        min_mark, max_mark = np.inf, -np.inf

        for dts, marks in zip(self.time_delta_seqs, self.type_seqs):
            dts = np.array(dts[1:-1 if marks[-1] == -1 else None])
            marks = np.array(marks)
            min_dt = min(min_dt, dts.min())
            max_dt = max(max_dt, dts.max())
            min_mark = min(min_mark, marks.min())
            max_mark = max(max_mark, marks.max())
            yp_bar = marks.mean()
            s_2_yp = marks.var()
            y_bar = dts.mean()
            s_2_y = dts.var()
            m = dts.shape[0]
            n += m
            # Formulat taken from https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
            s_2_x = (((n - 1) * s_2_x + (m - 1) * s_2_y) / (n + m - 1)) + (
                        (n * m * ((x_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1)))
            x_bar = (n * x_bar + m * y_bar) / (n + m)

            s_2_xp = (((n - 1) * s_2_xp + (m - 1) * s_2_yp) / (n + m - 1)) + (
                        (n * m * ((xp_bar - yp_bar) ** 2)) / ((n + m) * (n + m - 1)))
            xp_bar = (n * xp_bar + m * yp_bar) / (n + m)

        logger.info(f"delta times mean and variance:  {x_bar}, {(s_2_x ** 0.5)}")
        logger.info(f'min_dt: {min_dt}')
        logger.info(f'max_dt: {max_dt}')

        logger.info(f"Event types mean and variance: {xp_bar}, {(s_2_xp ** 0.5)}")
        logger.info(f'min_mark: {min_mark}')
        logger.info(f'max_mark: {max_mark}')

        return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), min_dt, max_dt, min_mark, max_mark


