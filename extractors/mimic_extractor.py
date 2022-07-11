#!/usr/bin/python3

import os

from extractors.extractor import Extractor

class MimicReportExtractor(Extractor):
    def __init__(self, base_dir="."):
        super().__init__(base_dir)

    def extract_report(self, file_name: str) -> str:
        with open(os.path.normpath(os.path.join(self.base_dir, file_name))) as file:
            return "".join(file.readlines())
