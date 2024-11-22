from abc import abstractmethod
from torch.utils.data import Dataset, DataLoader

from wireless_tpp.preprocess.data_collator import TPPDataCollator
from wireless_tpp.preprocess.event_tokenizer import EventTokenizer

class BaseTPPDataset(Dataset):
    @abstractmethod
    def get_dt_stats(self, **kwargs):
        pass

def get_data_loader(dataset: BaseTPPDataset, backend: str, tokenizer: EventTokenizer, **kwargs):
    use_torch = backend == 'torch'

    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy

    data_collator = TPPDataCollator(tokenizer=tokenizer,
                                    return_tensors='pt',
                                    max_length=tokenizer.model_max_length,
                                    padding=padding,
                                    truncation=truncation)

    return DataLoader(dataset,
                        collate_fn=data_collator,
                        **kwargs)