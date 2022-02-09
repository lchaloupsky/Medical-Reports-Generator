#!/usr/bin/python3

from abc import ABC, abstractmethod

class Extractor(ABC):
    def __init__(self, base_dir="."):
        self.base_dir = "{0}/".format(base_dir)

    @abstractmethod
    def extract_report(self, file_name: str) -> str:
        '''Extracts text from report.'''
        raise NotImplementedError()
