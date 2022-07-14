#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class LowercasePreprocessor(Preprocessor):
    def preprocess(self, text: str) -> str:
        '''Simply converts text into lowercase.'''
        return text.lower()