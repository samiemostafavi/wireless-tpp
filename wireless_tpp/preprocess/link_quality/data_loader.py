from wireless_tpp.preprocess import BaseTPPDataLoader
from wireless_tpp.utils import load_pickle, py_assert
from wireless_tpp.preprocess import get_data_loader
from wireless_tpp.preprocess import EventTokenizer
from wireless_tpp.config_factory import DataSpecConfig

from .dataset import TPPDatasetLinkQuality

class TPPDataLoaderLinkQuality(BaseTPPDataLoader):
    def __init__(self, data_config, backend, **kwargs):
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, e.g., tensorflow or torch.
        """
        if data_config is not None:
            self.data_config = data_config
            self.num_event_types = data_config.data_specs.num_event_types
        else:
            self.data_config = None
            self.num_event_types = None
        self.backend = backend
        self.source_data = kwargs.get('source_data', None)
        self.source_data_specs = kwargs.get('source_data_specs', None)
        self.kwargs = kwargs

    def build_input_from_pkl(self, source_dir: str, split: str):
        data = load_pickle(source_dir)

        if self.num_event_types is not None:
            py_assert(data["dim_process"] == self.num_event_types,
                    ValueError,
                    "inconsistent dim_process in different splits?")

        source_data = data[split]
        time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]
        type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        mcs_seqs = [[x["mcs_index"] for x in seq] for seq in source_data]
        num_rbs_seqs = [[x["num_rbs"] for x in seq] for seq in source_data]
        mretx_seqs = [[x["mretx"] for x in seq] for seq in source_data]
        rfailed_seqs = [[x["rfailed"] for x in seq] for seq in source_data]

        input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs, 'mcs_seqs': mcs_seqs, 'num_rbs_seqs': num_rbs_seqs, 'mretx_seqs': mretx_seqs, 'rfailed_seqs' : rfailed_seqs})

        return input_dict

    def build_input_from_json(self, source_dir: str, split: str):
        raise NotImplementedError

    def get_loader(self, split='train', **kwargs):
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            EasyTPP.DataLoader: the data loader for tpp data.
        """
        if self.source_data is not None:
            time_seqs = [[x["time_since_start"] for x in seq] for seq in self.source_data]
            time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in self.source_data]
            type_seqs = [[x["type_event"] for x in seq] for seq in self.source_data]
            mcs_seqs = [[x["mcs_index"] for x in seq] for seq in self.source_data]
            num_rbs_seqs = [[x["num_rbs"] for x in seq] for seq in self.source_data]
            mretx_seqs = [[x["mretx"] for x in seq] for seq in self.source_data]
            rfailed_seqs = [[x["rfailed"] for x in seq] for seq in self.source_data]

            data = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs, 'mcs_seqs': mcs_seqs, 'num_rbs_seqs': num_rbs_seqs, 'mretx_seqs': mretx_seqs, 'rfailed_seqs' : rfailed_seqs})
        else:
            data_dir = self.data_config.get_data_dir(split)
            data_source_type = data_dir.split('.')[-1]

            if data_source_type == 'pkl':
                data = self.build_input_from_pkl(data_dir, split)
            else:
                data = self.build_input_from_json(data_dir, split)

        dataset = TPPDatasetLinkQuality(data)
        if self.data_config is not None:
            tokenizer = EventTokenizer(self.data_config.data_specs)
        else:
            tokenizer = EventTokenizer(DataSpecConfig(**self.source_data_specs))
        loader = get_data_loader(dataset,
                                 self.backend,
                                 tokenizer,
                                 batch_size=self.kwargs['batch_size'],
                                 shuffle=self.kwargs['shuffle'],
                                 **kwargs)

        return loader