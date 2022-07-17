#!/usr/bin/python3

import re
import os
import pickle

from string import digits
from extractors.extractor import Extractor
from preprocessors.preprocessor import Preprocessor
from pathlib import Path
from utils import *

class TrueCasingPreprocessor(Preprocessor):
    '''Coverts all words written only with capital letters into form with only one capital letter.'''
    _DO_SKIP = {
        'CHEST', 'PORTABLE', 'CONCLUSION', 'TYPE', 'FOR', 'SINGLE', 'REPORT', 'COMMENT', 'CC', 'TWO', 'URGENT',
        'WET', 'BRINGS', 'MOTHER', 'BRACES', 'PEOPLE', 'ROTHIA', 'READ', 'PEOPLE', 'HIDING'
    }
    _DO_NOT_SKIP = {'PA', 'AP', 'CXR', 'RN'}
    _SEP = "<##--separator--##>"

    def __init__(self, extractor: Extractor = None, directory: str = None, dataset: str = None, skip_regex: str = None) -> None:
        '''
        Constructs new AnonymousSequencePreprocessor class.

        :param extractor: Specific dataset exractor
        :param directory:
        :param dataset:
        :param skip_regex:
        '''
        super().__init__()
        
        self._skip_regex = skip_regex
        if extractor and directory and dataset:
            self._get_word_frequencies(extractor, directory, dataset)

    def preprocess(self, text: str) -> str:
        # split text
        line_splits = text.split('\n')

        # split each line by whitespaces and deletes empty elements(representing multiple whitespaces)
        line_splits = map(self._convert_line, line_splits)

        # return processed text
        return '\n'.join(line_splits)

    def _convert_line(self, line: str) -> str:
        '''Splits line into words and each word written in uppercase only convert to title notation.'''
        
        text_len = len(line)
        start = text_len - len(line.lstrip())
        end = text_len - len(line.rstrip())

        words = re.split(r'(\W+)', line[start:-end or None])
        words = map(self._get_cased_word, words)

        return line[:start] + "".join(words) + (line[-end:] if end > 0 else "")

    def _get_cased_word(self, word: str) -> str:       
        return "".join(map(self._get_cased_part, re.split(r"(_)", word)))

    def _get_cased_part(self, part: str) -> str:
        if not part.isupper() or len(part) == 1:
            return part

        if not self._word_frequencies:
            return part.title() if len(part) > 3 else part # maybe do it lower instead of title
        
        lower = part.lower()
        return self._word_frequencies[lower] if self._should_find_in_dict(lower) else lower

    def _should_find_in_dict(self, part: str) -> bool:
        return part in self._word_frequencies and (len(part) <= 6 or self._is_formula(part))

    def _is_formula(self, part: str) -> bool:
        stripped = part.strip(digits)
        return not stripped.isalpha() and stripped.isalnum()

    def _get_word_frequencies(self, extractor: Extractor, directory: str, dataset: str) -> dict[str, str]:
        base = Path(__file__).parent

        # check if for given dataset wordFrequencies exists
        if os.path.exists(base/'{}.pkl'.format(dataset)):
            with open(base/'{}.pkl'.format(dataset), 'rb') as f:
                self._word_frequencies = pickle.load(f)

            return

        print(f"Creating new true casing dictionaries for '{dataset}' dataset.", flush=True)
        # if they dont exist for given dataset, calculate them
        all_frequencies = dict()
        with ProgressBar(get_total_dir_size(directory)) as progress_bar:
            for subdir, _, filenames in os.walk(directory):
                for filename in filenames:
                    text = extractor.extract_report(os.path.join(subdir, filename))
                    text_parts = re.sub(self._skip_regex, TrueCasingPreprocessor._SEP, text).split(TrueCasingPreprocessor._SEP) if self._skip_regex is not None else [text]
                    for text_part in text_parts:
                        self._process_text_part(text_part, all_frequencies)

                    progress_bar.update()

        # for each get the element with max occurences
        self._word_frequencies = dict()
        for key, value in all_frequencies.items():
            self._word_frequencies[key] = max(value, key=value.get)

        # save for further usage
        with open(base/'{}.pkl'.format(dataset), 'wb') as f:
            pickle.dump(self._word_frequencies, f)

        # save for debugging
        with open(base/'all_{}.pkl'.format(dataset), 'wb') as f:
            pickle.dump(all_frequencies, f)

        print(f"Successfully created true casing dictionaries for '{dataset}' dataset.", flush=True)

    def _process_text_part(self, text_part: str, all_frequencies: dict[str, dict[str, int]]) -> None:
        for line in re.split(r"((?: *\r? *[\n]){1,})", text_part):
            if not line or line.isspace():
                continue
            
            for sent in re.split(r"([.?!;]+)", line):
                for part in filter(None, re.split(r"([^:]+:)", sent)):
                    # skip those headings which are all in upper
                    is_heading_to_skip = ":" in part and part.isupper()

                    for word in re.split(r'(\W+)', part):
                        # keep only parts which cannot be skipped from headings written in uppercase only
                        if is_heading_to_skip and word not in TrueCasingPreprocessor._DO_NOT_SKIP:
                            continue

                        # skip words which should be skipped in order to get correct stats about them
                        if word in TrueCasingPreprocessor._DO_SKIP:
                            continue

                        # update stats
                        lower = word.lower()                       
                        if lower not in all_frequencies:
                            all_frequencies[lower] = dict() 

                        if word not in all_frequencies[lower]:
                            all_frequencies[lower][word] = 0

                        all_frequencies[lower][word] += 1