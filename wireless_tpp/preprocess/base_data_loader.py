from abc import abstractmethod

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