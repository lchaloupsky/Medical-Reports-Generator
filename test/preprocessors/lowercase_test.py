from preprocessors import *

def test_lowercase():
    prep = LowercasePreprocessor()
    input_txt = "1. RANDOMLY cased TExT."
    assert prep.preprocess(input_txt) == input_txt.lower()