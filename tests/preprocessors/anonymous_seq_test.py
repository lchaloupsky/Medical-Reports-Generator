from preprocessors import *

def test_anonymous_seq_prep():
    prep = AnonymousSequencePreprocessor("XX")

    input_txt  = "XXXX. XXXX-years-old XXXXWhat? XXhello XXXX, XXXX! XXXX/ XXX Xerox XXXX's XXXX: XXXX- HelloXXXX XXXX_. HelloXX. Why am I here? XXXX WordXXWord XXXX/XXXX/weakness/lp XXXX-A-XXXX XXXX-XXXX 4XXXX x-XXXX. x-XXXX ComparisXXXX/11 11//15/XXXX XXXX<XXXX>technologist XXXX..XXXX"
    output_txt = "XXXX. XXXX-years-old XXXX What? XX hello XXXX, XXXX! XXXX/ XXX Xerox XXXX's XXXX: XXXX- Hello XXXX XXXX_. Hello XX. Why am I here? XXXX Word XX Word XXXX/XXXX/weakness/lp XXXX-A-XXXX XXXX-XXXX 4 XXXX x-XXXX. x-XXXX Comparis XXXX/11 11//15/XXXX XXXX<XXXX>technologist XXXX..XXXX"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "COMPARISON: Comparison XXXX, XXXX. Well-expanded and clear lungs."
    output_txt = "COMPARISON: Comparison XXXX, XXXX. Well-expanded and clear lungs."
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "INDICATION: 786.2, 53XXXX with XXXX x 6 months"
    output_txt = "INDICATION: 786.2, 53 XXXX with XXXX x 6 months"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "contour. No XXXX acute abnormalities."
    output_txt = "contour. No XXXX acute abnormalities."
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "XXXX-year-old male, pain ComparisXXXXXXXX\n"
    output_txt = "XXXX-year-old male, pain Comparis XXXXXXXX\n"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "COMPARISON: Chest radiograph XXXX/XXXX, right foot radiograph XXXX/XXXX.\n\n"
    output_txt = "COMPARISON: Chest radiograph XXXX/XXXX, right foot radiograph XXXX/XXXX.\n\n"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "There are XXXX-filled loops. The bowel XXXX pattern."
    output_txt = "There are XXXX-filled loops. The bowel XXXX pattern."
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "XXXX-year-old male, XXXX.\n\nFINDINGS: Something XX-XX-XX-XX-XX/XX.XXX:XXX;"
    output_txt = "XXXX-year-old male, XXXX.\n\nFINDINGS: Something XX-XX-XX-XX-XX/XX.XXX:XXX;"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "COMPARISON: Chest x-XXXX XXXX, XXXX\nSomething different"
    output_txt = "COMPARISON: Chest x-XXXX XXXX, XXXX\nSomething different"
    assert prep.preprocess(input_txt) == output_txt

    input_txt = "IMPRESSION: XXXXyo AAM with XXXX and history of asthma with XXXX for six weeks, immunosuppre"
    output_txt = "IMPRESSION: XXXX yo AAM with XXXX and history of asthma with XXXX for six weeks, immunosuppre"
    assert prep.preprocess(input_txt) == output_txt