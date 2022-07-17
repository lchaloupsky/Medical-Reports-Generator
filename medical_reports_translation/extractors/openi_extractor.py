#!/usr/bin/python3

import os

from xml.dom import minidom
from extractors.extractor import Extractor

class OpenIReportExtractor(Extractor):
    '''Class for extracting text from OpenI reports.'''
    def __init__(self, base_dir: str = "."):
        super().__init__(base_dir)

    def extract_report(self, file_path: str) -> str:
        report_xml = minidom.parse(os.path.normpath(os.path.join(self.base_dir, file_path)))
        abstract_texts_xml = report_xml \
            .getElementsByTagName("MedlineCitation")[0] \
            .getElementsByTagName("Article")[0] \
            .getElementsByTagName("Abstract")[0] \
            .getElementsByTagName("AbstractText")

        report = ""
        for element in abstract_texts_xml:
            report += "{}: {}\n\n".format(str(element.attributes["Label"].value).title(), element.firstChild.data if element.firstChild else "")

        return report[:-2]
