import datasets
import sys
import itertools

from pathlib import Path

class DatasetsWrapper:
    def __init__(self, *args, dataset = None, batch_size = 1000, min_text_len = None, max_text_len = None, stream_iter = False, **kwargs):
        self.batch_size = batch_size
        self.min_text_len = min_text_len if min_text_len is not None else 0
        self.max_text_len = max_text_len if max_text_len is not None else sys.maxsize
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
            if self._is_usable(text["text"]):
                yield text["text"]

    def _batch_iter(self):
        for start in range(0, len(self._dataset), self.batch_size):
            yield list(filter(self._is_usable, self._dataset[start:start + self.batch_size]["text"]))

    def _is_usable(self, text):
        txt_len = len(text)
        return txt_len >= self.min_text_len and txt_len <= self.max_text_len

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
            batch_size=self.batch_size, 
            min_text_len=self.min_text_len, 
            max_text_len=self.max_text_len
        )

    def _filter_out_control_chars(self, text):
        return self.control_chars_filter.filter(text)

    def save_to_file(self, path, text_delim="", filter_control_chars=True):
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
                    if filter_control_chars:
                        text = self._filter_out_control_chars(text)

                    f.write(f"{text}\n{text_delim}")

class ControlCharsFilter:
    _CONTROL_CHARS = set(map(chr, itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))))
    _CONTROL_CHARS_EXCEPTIONS = {"\n", "\r", "\t"}
    _CONTROL_CHARS_MAP = {ord(char):None for char in _CONTROL_CHARS - _CONTROL_CHARS_EXCEPTIONS}

    def filter(self, text):
        return text.translate(self._CONTROL_CHARS_MAP)