from preprocessors import *

def test_line_starts():
    prep = LineStartsPreprocessor()
    input_txt = \
    "     FINAL REPORT     \n" + \
    "   Dummy report text.   \n" + \
    "   Another dummy report text.   "
    output_txt = \
    "  FINAL REPORT  \n" + \
    "Dummy report text.\n" + \
    "Another dummy report text."
    assert prep.preprocess(input_txt) == output_txt

    input_txt = \
    "     FINAL REPORT     \n" + \
    "Dummy report text.\n" + \
    "Another dummy report text."
    output_txt = \
    "     FINAL REPORT     \n" + \
    "Dummy report text.\n" + \
    "Another dummy report text."
    assert prep.preprocess(input_txt) == output_txt