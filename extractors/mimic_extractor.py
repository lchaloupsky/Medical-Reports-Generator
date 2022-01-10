#!/usr/bin/python3

from extractors.extractor import Extractor

class MimicReportExtractor(Extractor):
    def __init__(self, baseDir="."):
        self.baseDir = "{0}/".format(baseDir)

    def extractReport(self, fileName: str) -> str:
        with open(fileName) as file:
            return "".join(file.readlines())
