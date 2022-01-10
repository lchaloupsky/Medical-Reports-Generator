#!/usr/bin/python3

from abc import ABC, abstractmethod

class Extractor(ABC):
    def __init__(self, baseDir="."):
        self.baseDir = "{0}/".format(baseDir)

    @abstractmethod
    def extractReport(self, fileName: str) -> str:
        '''Extracts text from report.'''
        raise NotImplementedError()
