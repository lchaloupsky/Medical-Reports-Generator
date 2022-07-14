#!/usr/bin/python3

import re

from preprocessors.preprocessor import Preprocessor

class AnonymousSequencePreprocessor(Preprocessor):
    DEFAULT_SEQ: str = "_"

    def __init__(self, seq: str = DEFAULT_SEQ) -> None:
        '''The "seq" attribute represents a common start of every anonymous sequence.'''
        super().__init__()
        self._seq = seq if seq is not None else self.DEFAULT_SEQ


    def preprocess(self, text: str) -> str:
        '''Identifies all sequences anonymizing sensitive data and turns them into separate words.'''
        seq_end = self._seq[-1]
        split = re.split(f"({self._seq[:-1]}{seq_end}+)", text)
        curr = ""
        for s in filter(None, split):
            if s == len(s) * seq_end:
                curr += s if not curr or not curr[-1].isalnum() else f" {s}"
            else:
                curr += s if not curr or not s[0].isalnum()     else f" {s}"

        return curr