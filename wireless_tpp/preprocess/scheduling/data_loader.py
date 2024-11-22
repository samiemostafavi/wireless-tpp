from wireless_tpp.preprocess import BaseTPPDataLoader
from wireless_tpp.utils import load_pickle, py_assert

class TPPDataLoaderScheduling(BaseTPPDataLoader):
    def __init__(self, data_config, backend, **kwargs):
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, e.g., tensorflow or torch.
        """
        self.data_config = data_config
        self.num_event_types = data_config.data_specs.num_event_types
        self.backend = backend
        self.kwargs = kwargs

    def build_input_from_pkl(self, source_dir: str, split: str):
        data = load_pickle(source_dir)

        if self.num_event_types is not None:
            py_assert(data["dim_process"] == self.num_event_types,
                    ValueError,
                    "inconsistent dim_process in different splits?")

        source_data = data[split]

        # we incorporate the following additional keys in the source_data
        # 'len' 'mcs_index' 'mac_retx' 'rlc_failed' 'num_rbs'
        slot_seqs = [[x["slot"] for x in seq] for seq in source_data]
        len_seqs = [[x["len"] for x in seq] for seq in source_data]
        mcs_seqs = [[x["mcs_index"] for x in seq] for seq in source_data]
        mac_retx_seqs = [[x["mac_retx"] for x in seq] for seq in source_data]
        rlc_failed_seqs = [[x["rlc_failed"] for x in seq] for seq in source_data]
        num_rbs_seqs = [[x["num_rbs"] for x in seq] for seq in source_data]

        # default keys
        time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]

        input_dict = dict(
            {
                'slot_seqs': slot_seqs,
                'len_seqs': len_seqs,
                'mcs_seqs': mcs_seqs,
                'mac_retx_seqs': mac_retx_seqs,
                'rlc_failed_seqs': rlc_failed_seqs,
                'num_rbs_seqs': num_rbs_seqs,
                'time_seqs': time_seqs, 
                'time_delta_seqs': time_delta_seqs, 
                'type_seqs': type_seqs
            }
        )

        return input_dict

    def build_input_from_json(self, source_dir: str, split: str):
        from datasets import load_dataset
        split_ = 'validation' if split == 'dev' else split
        # load locally
        if source_dir.split('.')[-1] == 'json':
            data = load_dataset('json', data_files={split_: source_dir}, split=split_)
        elif source_dir.startswith('easytpp'):
            data = load_dataset(source_dir, split=split_)
        else:
            raise NotImplementedError

        py_assert(data['dim_process'][0] == self.num_event_types,
                  ValueError,
                  "inconsistent dim_process in different splits?")

        # we incorporate the following additional keys in the source_data
        # 'len' 'mcs_index' 'mac_retx' 'rlc_failed' 'num_rbs'
        slot_seqs = data['slot']
        len_seqs = data['len']
        mcs_seqs = data['mcs_index']
        mac_retx_seqs = data['mac_retx']
        rlc_failed_seqs = data['rlc_failed']
        num_rbs_seqs = data['num_rbs']

        # default keys
        time_seqs = data['time_since_start']
        type_seqs = data['type_event']
        time_delta_seqs = data['time_since_last_event']

        input_dict = dict(
            {
                'slot_seqs': slot_seqs,
                'len_seqs': len_seqs,
                'mcs_seqs': mcs_seqs,
                'mac_retx_seqs': mac_retx_seqs,
                'rlc_failed_seqs': rlc_failed_seqs,
                'num_rbs_seqs': num_rbs_seqs,
                'time_seqs': time_seqs, 
                'time_delta_seqs': time_delta_seqs, 
                'type_seqs': type_seqs
            }
        )
        return input_dict
