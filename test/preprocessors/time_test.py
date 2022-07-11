from preprocessors import *

def test_time():
    prep = TimePreprocessor()
    input_txt = "945AM ds 945 AM da \n \n 10 10pm 5:06 A.M. 5:06AM \n \n dsad 4pmsad as 9AM I am happy"
    output_txt = "9:45 AM ds 9:45 AM da \n \n 10:10 pm 5:06 A.M. 5:06 AM \n \n dsad 4 pmsad as 9 AM I am happy"
    assert prep.preprocess(input_txt) == output_txt 