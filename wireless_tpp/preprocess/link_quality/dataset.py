from typing import Dict

import numpy as np
from wireless_tpp.preprocess import BaseTPPDataset

from wireless_tpp.preprocess.data_collator import TPPDataCollator
from wireless_tpp.utils import py_assert, logger

class TPPDatasetLinkQuality(BaseTPPDataset):
    def __init__(self, data: Dict):
        self.data_dict = data
        self.mcs_seqs = self.data_dict.get('mcs_seqs', None)
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']
        self.mcs_time_seqs = self.data_dict.get('mcs_time_seqs', None)
        self.mcs_time_delta_seqs = self.data_dict.get('mcs_time_delta_seqs', None)
        self.mcs_type_seqs = self.data_dict.get('mcs_type_seqs', None)

    def __len__(self):
        """

        Returns: length of the dataset

        """
        if self.mcs_seqs:
            py_assert(len(self.mcs_seqs) == len(self.time_seqs) and len(self.time_seqs) == len(self.type_seqs) and len(self.time_delta_seqs) == len(self.type_seqs),
                    ValueError,
                    f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
                    f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}")
        else:
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
        if self.mcs_seqs:
            return dict({'mcs_seqs': self.mcs_seqs[idx], 'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                        'type_seqs': self.type_seqs[idx]})
        else:
            return dict({'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                        'type_seqs': self.type_seqs[idx]})

    def to_tf_dataset(self, data_collator: TPPDataCollator, **kwargs):
        logger.error("Tensorflow is not available.")
        return None

    def get_dt_stats(self, **kwargs):

        # get includes_mcs and mcs_events from kwargs
        includes_mcs = kwargs.get('includes_mcs', False)
        mcs_events = kwargs.get('mcs_events', False)
        num_event_types_no_mcs = kwargs.get('num_event_types_no_mcs', np.inf)
        
        # includes_mcs
        x_bar_mcs = None
        s_2_x_mcs = None
        min_mcs = None
        max_mcs = None
        if includes_mcs:
            if not self.mcs_seqs:
                logger.error("No MCS events found in the dataset although includes_mcs is set.")
            else:
                x_bar, s_2_x, n = 0., 0., 0
                min_mcs, max_mcs = np.inf, -np.inf
                for mcss in zip(self.mcs_seqs):
                    mcss = np.array(mcss)
                    min_mcs = min(min_mcs, mcss.min())
                    max_mcs = max(max_mcs, mcss.max())
                    y_bar = mcss.mean()
                    s_2_y = mcss.var()
                    m = mcss.shape[0]
                    n += m
                    s_2_x = (((n - 1) * s_2_x + (m - 1) * s_2_y) / (n + m - 1)) + (
                            (n * m * ((x_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1)))
                    x_bar = (n * x_bar + m * y_bar) / (n + m)

                logger.info(f"mcs mean and variance:  {x_bar}, {(s_2_x ** 0.5)}")
                x_bar_mcs = x_bar
                s_2_x_mcs = s_2_x
                logger.info(f'min_mcs: {min_mcs}')
                logger.info(f'max_mcs: {max_mcs}')

        # real retransmission events
        x_bar, s_2_x, xp_bar, s_2_xp, n = 0., 0., 0, 0., 0
        min_dt, max_dt = np.inf, -np.inf
        min_mark, max_mark = np.inf, -np.inf
        for inp_dts, inp_marks in zip(self.time_delta_seqs, self.type_seqs):
            # filter out the mcs events
            if mcs_events:
                filtered_dts = []
                filtered_marks = []
                for dt, mark in zip(inp_dts, inp_marks):
                    if mark < num_event_types_no_mcs:
                        filtered_dts.append(dt)
                        filtered_marks.append(mark)
                dts = np.array(filtered_dts)
                marks = np.array(filtered_marks)
            else:
                dts = np.array(inp_dts)
                marks = np.array(inp_marks)
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

        # mcs events
        m_bar, s_2_m, mp_bar, s_2_mp, n = 0., 0., 0, 0., 0
        m_min_dt, m_max_dt = np.inf, -np.inf
        m_min_mark, m_max_mark = np.inf, -np.inf
        if mcs_events:
            if self.mcs_time_delta_seqs == None or self.mcs_type_seqs == None:
                logger.error("No MCS events found in the dataset although mcs_events is set.")
            else:
                for inp_dts, inp_marks in zip(self.mcs_time_delta_seqs, self.mcs_type_seqs):
                    if len(inp_dts) == 0:
                        continue
                    dts = np.array(inp_dts)
                    marks = np.array(inp_marks)
                    m_min_dt = min(m_min_dt, dts.min())
                    m_max_dt = max(m_max_dt, dts.max())
                    m_min_mark = min(m_min_mark, marks.min())
                    m_max_mark = max(m_max_mark, marks.max())
                    yp_bar = marks.mean()
                    s_2_yp = marks.var()
                    y_bar = dts.mean()
                    s_2_y = dts.var()
                    m = dts.shape[0]
                    n += m
                    # Formulat taken from https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
                    s_2_m = (((n - 1) * s_2_m + (m - 1) * s_2_y) / (n + m - 1)) + (
                                (n * m * ((m_bar - y_bar) ** 2)) / ((n + m) * (n + m - 1)))
                    m_bar = (n * m_bar + m * y_bar) / (n + m)

                    s_2_mp = (((n - 1) * s_2_mp + (m - 1) * s_2_yp) / (n + m - 1)) + (
                                (n * m * ((mp_bar - yp_bar) ** 2)) / ((n + m) * (n + m - 1)))
                    mp_bar = (n * mp_bar + m * yp_bar) / (n + m)

                logger.info(f"MCS delta times mean and variance:  {m_bar}, {(s_2_m ** 0.5)}")
                logger.info(f'MCS min_dt: {m_min_dt}')
                logger.info(f'MCS max_dt: {m_max_dt}')

                logger.info(f"MCS event types mean and variance: {mp_bar}, {(s_2_mp ** 0.5)}")
                logger.info(f'MCS min_mark: {m_min_mark}')
                logger.info(f'MCS max_mark: {m_max_mark}')

        if includes_mcs:
            return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), x_bar_mcs, (s_2_x_mcs ** 0.5), min_dt, max_dt, min_mark, max_mark, min_mcs, max_mcs
        elif mcs_events:
            return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), min_dt, max_dt, min_mark, max_mark, m_bar, (s_2_m ** 0.5), mp_bar, (s_2_mp ** 0.5), m_min_dt, m_max_dt, m_min_mark, m_max_mark
        else:
            return x_bar, (s_2_x ** 0.5), xp_bar, (s_2_xp ** 0.5), min_dt, max_dt, min_mark, max_mark
