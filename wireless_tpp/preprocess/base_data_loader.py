from abc import abstractmethod
from wireless_tpp.preprocess import get_data_loader
from wireless_tpp.preprocess import BaseTPPDataset
from wireless_tpp.preprocess.event_tokenizer import EventTokenizer


class BaseTPPDataLoader:
    @abstractmethod
    def build_input_from_pkl(self, source_dir: str, split: str):
        pass

    @abstractmethod
    def build_input_from_json(self, source_dir: str, split: str):
        pass

    @abstractmethod
    def get_loader(self, split='train', **kwargs):
        pass

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

        dataset = BaseTPPDataset(data)
        tokenizer = EventTokenizer(self.data_config.data_specs)
        loader = get_data_loader(dataset,
                                 self.backend,
                                 tokenizer,
                                 batch_size=self.kwargs['batch_size'],
                                 shuffle=self.kwargs['shuffle'],
                                 **kwargs)

        return loader