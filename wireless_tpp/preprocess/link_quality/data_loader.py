from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import load_pickle, py_assert


class TPPDataLoaderLinkQuality:
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
        if self.data_config.data_specs.includes_mcs:
            mcs_seqs = [[x["mcs_index"] for x in seq] for seq in source_data]
        time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]
        if self.data_config.data_specs.mcs_events:
            nenm = self.data_config.data_specs.num_event_types_no_mcs
            mcs_time_seqs = [[x["time_since_start"] for x in seq if x["type_event"] >= nenm] for seq in source_data]
            mcs_type_seqs = [[x["type_event"] for x in seq if x["type_event"] >= nenm] for seq in source_data]
            mcs_time_delta_seqs = [[x["time_since_last_event"] for x in seq if x["type_event"] >= nenm] for seq in source_data]


        if self.data_config.data_specs.includes_mcs:
            input_dict = dict({'mcs_seqs': mcs_seqs, 'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs})
        elif self.data_config.data_specs.mcs_events:
            input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs, 'mcs_time_seqs': mcs_time_seqs, 'mcs_time_delta_seqs': mcs_time_delta_seqs, 'mcs_type_seqs': mcs_type_seqs})
        else:
            input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs})
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

        if self.data_config.data_specs.includes_mcs:
            mcs_seqs = data['mcs_index']
        time_seqs = data['time_since_start']
        type_seqs = data['type_event']
        time_delta_seqs = data['time_since_last_event']
        if self.data_config.data_specs.mcs_events:
            print("mcs_events is not supported for json data")
            return dict({})

        if self.data_config.data_specs.includes_mcs:
            input_dict = dict({'mcs_seqs': mcs_seqs, 'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs})
        elif self.data_config.data_specs.mcs_events:
            input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs, 'mcs_time_seqs': mcs_time_seqs, 'mcs_time_delta_seqs': mcs_time_delta_seqs, 'mcs_type_seqs': mcs_type_seqs})
        else:
            input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs})
        return input_dict

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
        data_dir = self.data_config.get_data_dir(split)
        data_source_type = data_dir.split('.')[-1]

        if data_source_type == 'pkl':
            data = self.build_input_from_pkl(data_dir, split)
        else:
            data = self.build_input_from_json(data_dir, split)

        dataset = TPPDataset(data)
        tokenizer = EventTokenizer(self.data_config.data_specs)
        loader = get_data_loader(dataset,
                                 self.backend,
                                 tokenizer,
                                 batch_size=self.kwargs['batch_size'],
                                 shuffle=self.kwargs['shuffle'],
                                 **kwargs)

        return loader

    def train_loader(self, **kwargs):
        """Return the train loader

        Returns:
            EasyTPP.DataLoader: data loader for train set.
        """
        return self.get_loader('train', **kwargs)

    def valid_loader(self, **kwargs):
        """Return the valid loader

        Returns:
            EasyTPP.DataLoader: data loader for valid set.
        """
        return self.get_loader('dev', **kwargs)

    def test_loader(self, **kwargs):
        """Return the test loader

        Returns:
            EasyTPP.DataLoader: data loader for test set.
        """
        return self.get_loader('test', **kwargs)

