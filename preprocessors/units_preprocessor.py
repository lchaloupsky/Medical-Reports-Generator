#!/usr/bin/python3

import re

from preprocessors.preprocessor import Preprocessor

class UnitsPreprocessor(Preprocessor):
    _NUMBER_EXCEPTIONS = {u + str(i) for u in {"nm", "μm", "mm", "cm", "dm", "m", "km"} for i in range(2,4)}
    _UNITS_EXCEPTIONS = {"vd", "des"}

    def preprocess(self, text: str) -> str:
        '''Finds units to numbers which are glued together in one word and splits them.\n
        This also includes non-existing units - mistakes in text, where the number and the word should be separated.'''

        # split text and map each line
        return "\n".join(map(self._map_line, text.split('\n')))

    def _map_line(self, line: str) -> str:
        # split to words and map each word
        return "".join(map(self._map_word, re.split(r"(\W+)", line)))

    def _map_word(self, word: str) -> str:
        # if word is empty or is made only of whitespaces or it is not starting with digit return word
        if not word or word.isspace() or not word[0].isdigit():
            return word

        print(word)
        num_part, text_part = self._split_word(word)
        # the text part contains also some numbers - it is not any unit, but some medical thing e.g. 3N0M0
        if not text_part.isalpha() and text_part.lower() not in UnitsPreprocessor._NUMBER_EXCEPTIONS:
            return word

        # units long at least 4 chars are rather mistakes or full name of the unit
        if len(text_part) >= 4 or text_part.lower() not in UnitsPreprocessor._UNITS_EXCEPTIONS:
            return self._get_splitted_form(num_part, text_part)

        return word

    def _split_word(self, word: str) -> tuple[str, str]:
        i = 0
        while i < len(word) and word[i].isdigit(): 
            i += 1

        return word[:i], word[i:]

    def _get_splitted_form(self, num_part: str, text_part: str) -> str:
        return f"{num_part} {text_part}"
        

text2 = "                                 FINAL REPORT\n\
 EXAMINATION:  CHEST (PORTABLE AP)\n\
 \n\
 INDICATION:  ___F with cough  // acute process?\n\
 \n\
 COMPARISON:  Chest radiograph ___\n\
 \n\
 FINDINGS: \n\
 \n\
 Single frontal view of the chest provided.\n\
 \n\
 10cm, There is no focal consolidation, effusion, or pneumothorax. The\n\
 cardiomediastinal silhouette is normal.  10cm, 20pps, 3VD, 4x12, 3HISTORY, 3N0M0. Again seen are multiple clips\n\
 projecting over the left breast and remote left-sided rib fractures.  No free\n\
 air below the right hemidiaphragm is seen. 13Pro\n\
 \n\
 IMPRESSION: \n\
 \n\
 No acute intrathoracic process."

#p = UnitsPreprocessor()
#print(p.preprocess(text2))