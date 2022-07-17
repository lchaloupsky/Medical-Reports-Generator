#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class LowercasePreprocessor(Preprocessor):
    '''Converts text into lowercase.'''
    def preprocess(self, text: str) -> str:
        return text.lower()