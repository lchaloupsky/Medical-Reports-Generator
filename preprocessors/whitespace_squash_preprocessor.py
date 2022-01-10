#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class WhitespacesSquashPreprocessor(Preprocessor):
    def preprocess(self, text: str):
        '''Squashes all multiple occurences of whitespace inside(not on the sides) text into one.'''

        # split text
        line_splits = text.split('\n')

        # split each line by whitespaces and deletes empty elements(representing multiple whitespaces)
        line_splits = map(self._split_and_keep_sides, line_splits)

        # return processed text
        return '\n'.join(line_splits)

    def _split_and_keep_sides(self, text: str):
        '''Squashes whitespaces inside text with keeping trailing spaces.'''
        
        text_len = len(text)
        start = text_len - len(text.lstrip())
        end = text_len - len(text.rstrip())

        return text[:start] + " ".join(text.split()) + (text[-end:] if end > 0 else "")
