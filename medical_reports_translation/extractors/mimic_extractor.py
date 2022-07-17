#!/usr/bin/python3

import os

from extractors.extractor import Extractor

class MimicReportExtractor(Extractor):
    '''Class for extracting text from MIMIC-CXR reports.'''
    def __init__(self, base_dir: str = "."):
        super().__init__(base_dir)

    def extract_report(self, file_path: str) -> str:
        with open(os.path.normpath(os.path.join(self.base_dir, file_path))) as file:
            return "".join(file.readlines())
