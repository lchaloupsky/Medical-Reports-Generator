#!/usr/bin/python3

import re

from preprocessors.preprocessor import Preprocessor

class CapitalizeStartsPreprocessor(Preprocessor):
    def preprocess(self, text: str) -> str:
        '''Capitalize each sentence start and the headings.'''
        return "".join(map(self._map_paragraph, re.split(r"((?: *\r? *[\n]){2,})", text)))

    def _map_paragraph(self, par: str) -> str:
        if not par or par.isspace():
            return par

        # first split by ": " - the titles
        semicolon_split = list(filter(None, re.split(r"(^[^:]+:)", par)))

        # there is no semicolon in paragraph
        if len(semicolon_split) == 1:
            return self._map_paragraph_lines(semicolon_split[0])
                   
        # capitalize the rest of the sentences
        return self._map_title_lines(semicolon_split[0]) + self._map_paragraph_lines("".join(semicolon_split[1:]))

    def _map_title_lines(self, title: str) -> str:
        return "\n".join(map(self._map_paragraph_lines, title.split("\n")))

    def _map_paragraph_lines(self, par: str) -> str:
        return "".join(map(self._capitalize, re.split(r"([.?!;]+)", par)))

    def _capitalize(self, text: str) -> str:
        start = len(text) - len(text.lstrip())
        return text[0:start] + text[start:start+1].upper() + text[start + 1:]

text = "                                 FINAL REPORT\n\
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
 There is no focal consolidation, effusion, or pneumothorax. The\n\
 cardiomediastinal silhouette is normal.  Again seen are multiple clips\n\
 projecting over the left breast and remote left-sided rib fractures.  No free\n\
 air below the right hemidiaphragm is seen.\n\
 \n\
 IMPRESSION: \n\
 \n\
 No acute intrathoracic process."

#p = CapitalizeStartsPreprocessor()
#print(p.preprocess(text.lower()))

text2 = "                                 FINAL ADDENDUM\n\
 ADDENDUM:  Findings were discussed with Dr. ___ ___ the phone by radiology\n\
 resident, Dr. ___ ___ at 1:45 p.m. on ___.\n\
 \n\
 ______________________________________________________________________________\n\
                                 FINAL REPORT\n\
 REASON FOR EXAMINATION:  Evaluation of the patient with pneumothorax.\n\
 \n\
 Portable AP radiograph of the chest was compared to prior study obtained on\n\
 ___.\n\
 \n\
 There is currently presence of moderate pneumothorax and no evidence of\n\
 effusion in the upper portion of the pleura demonstrated.  The extensive\n\
 consolidation over the lungs is redemonstrated.  Overall, no substantial\n\
 change otherwise since the prior study seen."

#p = CapitalizeStartsPreprocessor()
#print(p.preprocess(text2.lower()))