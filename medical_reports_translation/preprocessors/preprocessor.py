#!/usr/bin/python3

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        '''Preprocesses text for translation.'''
        raise NotImplementedError()