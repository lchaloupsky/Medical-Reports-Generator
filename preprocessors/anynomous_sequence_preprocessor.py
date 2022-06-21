#!/usr/bin/python3

import re

from preprocessors.preprocessor import Preprocessor

class AnonymousSequencePreprocessor(Preprocessor):
    DEFAULT_SEQ: str = "_"

    def __init__(self, seq: str = DEFAULT_SEQ) -> None:
        '''The "seq" attribute represents a common start of every anonymous sequence.'''
        super().__init__()
        self._seq = seq if seq is not None else self.DEFAULT_SEQ


    def preprocess(self, text: str) -> str:
        '''Identifies all sequences anonymizing sensitive data and turns them into separate words.'''
        seq_end = self._seq[-1]
        split = re.split(f"({self._seq[:-1]}{seq_end}+)", text)
        curr = ""
        for s in filter(None, split):
            if s == len(s) * seq_end:
                curr += s if not curr or not curr[-1].isalnum() else f" {s}"
            else:
                curr += s if not curr or not s[0].isalnum()     else f" {s}"

        return curr

'''
prep = AnonymousSequencePreprocessor("XX")
input  = "XXXX. XXXX-years-old XXXXWhat? XXhello XXXX, XXXX! XXXX/ XXX Xerox XXXX's XXXX: XXXX- HelloXXXX XXXX_. HelloXX. Why am I here? XXXX WordXXWord XXXX/XXXX/weakness/lp XXXX-A-XXXX XXXX-XXXX 4XXXX x-XXXX. x-XXXX ComparisXXXX/11 11//15/XXXX XXXX<XXXX>technologist XXXX..XXXX"
output = "XXXX. XXXX-years-old XXXX What? XX hello XXXX, XXXX! XXXX/ XXX Xerox XXXX's XXXX: XXXX- Hello XXXX XXXX_. Hello XX. Why am I here? XXXX Word XX Word XXXX/XXXX/weakness/lp XXXX-A-XXXX XXXX-XXXX 4 XXXX x-XXXX. x-XXXX Comparis XXXX/11 11//15/XXXX XXXX<XXXX>technologist XXXX..XXXX"
preprocessed = prep.preprocess(input)
print(preprocessed)
print(output)
print(preprocessed == output)
print(prep.preprocess("COMPARISON: Comparison XXXX, XXXX. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified."))
print(prep.preprocess("INDICATION: 786.2, 53XXXX with XXXX x 6 months"))
print(prep.preprocess("contour. No XXXX acute abnormalities since the previous chest radiograph."))
print(prep.preprocess("XXXX-year-old male, pain ComparisXXXXXXXX\n"))
print(prep.preprocess("COMPARISON: Chest radiograph XXXX/XXXX, right foot radiograph XXXX/XXXX.\n\n"))
print(prep.preprocess("There are very low lung volumes with associated central bronchovascular crowding. There is elevation of the left hemidiaphragm. There are XXXX-filled loops of mildly dilated colon in the left upper quadrant. The bowel XXXX pattern is not well evaluated secondary to incomplete imaging of the abdomen. There is no pneumothorax or definite pleural effusion. The streaky opacities in the lung bases may represent atelectasis. No definite infectious infiltrate is seen. There is scoliosis and exaggeration of the thoracic kyphosis."))
print(prep.preprocess("XXXX-year-old male, XXXX.\n\nFINDINGS: Something XX-XX-XX-XX-XX/XX.XXX:XXX;"))
print(prep.preprocess("COMPARISON: Chest x-XXXX XXXX, XXXX\nSomething different"))
print(prep.preprocess("IMPRESSION: XXXXyo AAM with XXXX and history of asthma with XXXX for six weeks, immunosuppre"))
'''