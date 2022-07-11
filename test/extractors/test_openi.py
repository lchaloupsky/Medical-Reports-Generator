import os

from extractors import *

def test_openi():
    extractor = OpenIReportExtractor(os.path.dirname(__file__))
    output_txt = "Test: Testing text."
    assert extractor.extract_report("test.xml") == output_txt