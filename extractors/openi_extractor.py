#!/usr/bin/python3

import os

from xml.dom import minidom
from extractors.extractor import Extractor

class OpenIReportExtractor(Extractor):
    def __init__(self, baseDir="."):
        self.baseDir = "{0}/".format(baseDir)

    def extractReport(self, fileName: str) -> str:
        report_xml = minidom.parse(os.path.join(self.baseDir, fileName))
        abstract_texts_xml = report_xml \
            .getElementsByTagName("MedlineCitation")[0] \
            .getElementsByTagName("Article")[0] \
            .getElementsByTagName("Abstract")[0] \
            .getElementsByTagName("AbstractText")

        report = ""
        for element in abstract_texts_xml:
            report += "{}: {}\n\n".format(str(element.attributes["Label"].value).title(), element.firstChild.data if element.firstChild else "")

        return report[:-2]