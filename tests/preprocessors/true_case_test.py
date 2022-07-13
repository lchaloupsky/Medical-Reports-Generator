from preprocessors import *
from extractors import *

def test_true_case():
    prep = TrueCasingPreprocessor(OpenIReportExtractor(), "test", "openi", None)
    input_txt = "Unkown abbr. PLT, OO, AA, REQ, MIP, COMPARE PA examination, MRI examination, CT examination, VERYLONGMISTAKE, positive TB, NonStaDarD CasE. Nam libero tempore, cum soluta nobis est."
    output_txt = "Unkown abbr. plt, oo, aa, req, mip, compare PA examination, MRI examination, CT examination, verylongmistake, positive TB, NonStaDarD CasE. Nam libero tempore, cum soluta nobis est."
    assert prep.preprocess(input_txt) == output_txt