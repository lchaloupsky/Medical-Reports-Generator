import os

from extractors import *

def test_mimic():
    extractor = MimicReportExtractor(os.path.dirname(__file__))
    output_txt = "Test: Testing text."
    assert extractor.extract_report("test.txt") == output_txt