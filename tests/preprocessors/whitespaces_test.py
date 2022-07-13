from preprocessors import *

def test_whitespace():
    prep = WhitespacesSquashPreprocessor()
    input_txt = " Lorem ipsum dolor     sit amet, consectetuer \t\t\t adipiscing elit.   Aenean placerat.  "
    output_txt = " Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean placerat.  "
    assert prep.preprocess(input_txt) == output_txt