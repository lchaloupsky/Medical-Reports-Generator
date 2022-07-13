from preprocessors import *

def test_semicolon_paragraph():
    prep = SemicolonParagraphPreprocessor()
    input_text = "\
    Indication: lorem ipsum\n\
    \n\
    Findings:\n\
    \n\
    Sed ac dolor sit amet purus malesuada congue. Aenean id metus id velit \n\
    ullamcorper pulvinar.\n\
    \n\
    Impression:\n\
    \n\
    Nam libero tempore, cum soluta nobis est"
    output_text = "\
    Indication: lorem ipsum\n\
    \n\
    Findings:    Sed ac dolor sit amet purus malesuada congue. Aenean id metus id velit \n\
    ullamcorper pulvinar.\n\
    \n\
    Impression:    Nam libero tempore, cum soluta nobis est"
    assert prep.preprocess(input_text) == output_text

    input_text = "\
    FINDINGS: \n\
    \n\
    Nam libero tempore, cum soluta nobis est.\n\
    \n\
    Aliquam erat volutpat. Nullam sapien sem, ornare ac, \n\
    nonummy non, lobortis a enim.\n\
    \n\
    IMPRESSION: \n\
    \n\
    Nam libero tempore, cum soluta nobis est."
    output_text = "\
    FINDINGS:     Nam libero tempore, cum soluta nobis est.\n\
    \n\
    Aliquam erat volutpat. Nullam sapien sem, ornare ac, \n\
    nonummy non, lobortis a enim.\n\
    \n\
    IMPRESSION:     Nam libero tempore, cum soluta nobis est."
    assert prep.preprocess(input_text) == output_text

    input_text = "Comparison: None.\n\n\
    Indication: LOREM IPSUM\n\n\
    Findings: Nam libero tempore, cum soluta nobis est.\n\n\
    Impression: "
    assert prep.preprocess(input_text) == input_text