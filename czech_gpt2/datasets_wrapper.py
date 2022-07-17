import datasets
import itertools

from pathlib import Path

class DatasetsWrapper:
    '''Class encapsulating the "datasets" package classes adding additional features.'''    
    def __init__(self, *args, dataset = None, batch_size = 1000, stream_iter = False, **kwargs):
        '''
        Constructs a new instance of DatasetsWrapper class.

        :param dataset: Name of the dataset, defaults to None
        :param batch_size: Batch size for batch iteration of the dataset, defaults to 1000
        :param stream_iter: Flag indicating wheter to use the streaming iterator or the batch iterator, defaults to False
        '''
        self.batch_size = batch_size
        self.stream_iter = stream_iter
        self.control_chars_filter = ControlCharsFilter()

        if dataset is not None: 
            self._dataset = dataset
        else:
            self._dataset = datasets.load_dataset(*args, **kwargs)

    def __iter__(self):
        if isinstance(self._dataset, datasets.IterableDataset) or self.stream_iter:
            return self._stream_iter()
        elif isinstance(self._dataset, datasets.Dataset):
            return self._batch_iter()
        else:
            return self._dataset.__iter__()

    def __getitem__(self, key):
        item = self._dataset.__getitem__(key)
        if isinstance(item, (datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset)):
            return self._copy_self(item)

        return item

    def __del__(self):
        if getattr(self._dataset, "__del__", None) is not None:
            self._dataset.__del__()

    def __enter__(self):
        return self._dataset.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self):
        return self._dataset.__len__()

    def _stream_iter(self):
        for text in self._dataset:
            yield text["text"]

    def _batch_iter(self):
        for start in range(0, len(self._dataset), self.batch_size):
            yield self._dataset[start:start + self.batch_size]["text"]

    def __getattr__(self, __name: str):
        attr = getattr(self._dataset, __name)
        if callable(attr):
            def __delegate_inner(*args, **kwargs):  
                result = attr(*args, **kwargs)
                if isinstance(result, (datasets.DatasetDict, datasets.Dataset, datasets.IterableDatasetDict, datasets.IterableDataset)):
                    return self._copy_self(result)
                else:
                    return result

            setattr(self, __name, __delegate_inner)
            return __delegate_inner

        return attr

    def _copy_self(self, dataset):
        return DatasetsWrapper(
            dataset=dataset,
            batch_size=self.batch_size
        )

    def save_to_file(self, path, text_delim="", control_chars_behaviour="remove"):
        '''
        Saves the whole dataset into the given path. Possibly filtering out texts with control characters in them.

        :param path: Path to save the dataset
        :param text_delim: Text delimiter to separate the texts, defaults to ""
        :param control_chars_behaviour: How should treat the control chars. One of [None, "filter", "remove"], defaults to "remove"
        '''
        datasets_to_save = []
        if isinstance(self._dataset, dict):
            datasets_to_save = datasets_to_save.extend(zip(self._dataset.keys(), map(self._copy_self, self._dataset.values())))
        else:
            datasets_to_save.append((self._dataset.split, self))

        path = Path(path) if isinstance(path, str) else path
        text_delim = text_delim if text_delim == "" or text_delim is None else f"{text_delim}\n"
        for k, d in datasets_to_save:
            with open(f"{path/k}.txt", "wt") as f:
                for text in d:
                    if control_chars_behaviour == "filter":
                        text = self.control_chars_filter.filter(text)
                    elif control_chars_behaviour == "remove" and self.control_chars_filter.contains_control_chars(text):
                        continue

                    f.write(f"{text}\n{text_delim}")

class ControlCharsFilter:
    '''Helper class dealing with the control characters.'''
    _CONTROL_CHARS = set(map(chr, itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))))
    _CONTROL_CHARS_EXCEPTIONS = {"\n", "\r", "\t"}
    _CONTROL_CHARS_MAP = {ord(char):None for char in _CONTROL_CHARS - _CONTROL_CHARS_EXCEPTIONS}

    def contains_control_chars(self, text):
        """
        Checks whether text contains a control character or not.

        :param text: Text to be checked
        :return: True if text contain a control character
        """        
        text_len = len(text)
        return text_len != len(self.filter(text))

    def filter(self, text):
        '''
        Filters out the control character from the text.

        :param text: Text to be filtered
        :return: Filtered text
        '''
        return text.translate(self._CONTROL_CHARS_MAP)