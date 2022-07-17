#!/usr/bin/python3

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    '''Abstract class for classes preprocessing text.'''
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        '''
        Preprocesses text for translation.

        :param text: Text to be preprocessed
        :return Preprocessed text
        '''
        raise NotImplementedError()