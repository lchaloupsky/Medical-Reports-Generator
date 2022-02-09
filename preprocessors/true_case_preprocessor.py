#!/usr/bin/python3

import re
import os
import pickle

from extractors.extractor import Extractor
from preprocessors.preprocessor import Preprocessor

class TrueCasingPreprocessor(Preprocessor):
    def __init__(self, extractor: Extractor = None, directory: str = None, dataset: str = None) -> None:
        super().__init__()
        if extractor and directory and dataset:
            self._get_word_frequencies(extractor, directory, dataset)

    def preprocess(self, text: str) -> str:
        '''Coverts all words written only with capital letters into form with only one capital letter.'''

        # split text
        line_splits = text.split('\n')

        # split each line by whitespaces and deletes empty elements(representing multiple whitespaces)
        line_splits = map(self._convert_line, line_splits)

        # return processed text
        return '\n'.join(line_splits)

    def _convert_line(self, line: str):
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
            return part.title() if len(part) > 3 else part
        
        return self._word_frequencies[part.lower()] if len(part) <= 4 else part.lower()

    def _get_word_frequencies(self, extractor: Extractor, directory: str, dataset: str):
        # check if for given dataset wordFrequencies exists
        if os.path.exists('{}.pkl'.format(dataset)):
            with open('{}.pkl'.format(dataset), 'rb') as f:
                self._word_frequencies = pickle.load(f)

            return

        # if they dont exist for given dataset, calculate them
        all_frequencies = dict()
        for subdir, _, filenames in os.walk(directory):
            for filename in filenames:
                for line in extractor.extract_report(os.path.join(subdir, filename)).split():
                    for word in re.split(r'(\W+)', line):
                        lower = word.lower()
                        
                        if lower not in all_frequencies:
                            all_frequencies[lower] = dict() 

                        if word not in all_frequencies[lower]:
                            all_frequencies[lower][word] = 0

                        all_frequencies[lower][word] += 1

        # for each get the element with max occurences
        self._word_frequencies = dict()
        for key, value in all_frequencies.items():
            self._word_frequencies[key] = max(value, key=value.get)

        # save for furhter usage
        with open('{}.pkl'.format(dataset), 'wb') as f:
            pickle.dump(self._word_frequencies, f)
