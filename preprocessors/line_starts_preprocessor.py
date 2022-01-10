#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class LineStartsPreprocessor(Preprocessor):
    def preprocess(self, text: str):
        '''Takes away all whitespaces at the start/end of a line which are common for all lines.'''

        # split text
        line_splits = text.split('\n')

        # find minimum number of white spaces of all lines on both sides
        min_whitespaces_left = min(map(lambda x: len(x) - len(x.lstrip()), line_splits))
        min_whitespaces_right = min(map(lambda x: len(x) - len(x.rstrip()), line_splits))

        # delete all common whitespaces and return processed text
        return '\n'.join(map(lambda x: x[min_whitespaces_left : -min_whitespaces_right or len(x)], line_splits))
