from preprocessors import *

def test_units():
    prep = UnitsPreprocessor()
    input_txt = "random.  10cm, 20pps, 3VD, 4x12, 3HISTORY, 3N0M0. Random"
    output_txt = "random.  10 cm, 20 pps, 3VD, 4x12, 3 HISTORY, 3N0M0. Random"
    assert prep.preprocess(input_txt) == output_txt