from preprocessors import *

def test_capitalize_starts():
    prep = CapitalizeStartsPreprocessor()
    input_txt = "                                 Final report\n\
    Examination:  Chest (portable ap)\n\
    \n\
    Findings: \n\
    \n\
    Nullam rhoncus aliquam metus.\n\
    \n\
    In convallis. Ut enim ad minima veniam, quis nostrum exercitationem \n\
    ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? \n\
    Donec quis nibh at felis congue commodo. Nullam at arcu a est sollicitudin euismod. \n\
    Aliquam id dolor. Integer lacinia.\n\
    \n\
    Impression: \n\
    \n\
    Ut tempus purus at lorem."
    assert prep.preprocess(input_txt.lower()) == input_txt

    input_txt = "\
    Addendum:  Aliquam erat volutpat. Nullam sapien sem, ornare ac.\n\
    \n\
    Reason for examination:  Nullam rhoncus aliquam metus.\n\
    \n\
    Aenean vel massa quis mauris vehicula lacinia.\n\
    \n\
    Pellentesque pretium lectus id turpis. Fusce tellus odio, dapibus id fermentum quis, \n\
    suscipit id erat. Pellentesque habitant morbi tristique senectus et \n\
    netus et malesuada fames ac turpis egestas. Etiam egestas wisi a erat. \n\
    Nullam at arcu a est sollicitudin euismod."
    assert prep.preprocess(input_txt.lower()) == input_txt