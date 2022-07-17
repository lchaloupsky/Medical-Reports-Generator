#!/usr/bin/python3

from abc import ABC, abstractmethod

class Extractor(ABC):
    ''' 
    Abstract class for the dataset file text extraction.
    '''
    def __init__(self, base_dir: str = "."):
        '''
        Constructs new Extractor class.
        :param base_dir: Base directory for the files
        '''
        self.base_dir = "{0}/".format(base_dir)

    @abstractmethod
    def extract_report(self, file_path: str) -> str:
        '''
        Extracts text from the report file.

        :param file_path: The file name or path to the file
        :return Extracted report text
        '''
        raise NotImplementedError()
