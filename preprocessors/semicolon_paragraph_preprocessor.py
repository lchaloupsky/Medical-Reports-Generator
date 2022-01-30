#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class SemicolonParagraphPreprocessor(Preprocessor):
    SEMICOLON = ":"

    def preprocess(self, text: str):
        '''Puts paragraps starting after a semicolon on the same line.'''

        # split text
        line_splits = text.split('\n')

        line_iter, final_lines = iter(line_splits), []
        for line in line_iter:
            next_lines = [line]
            line_split = line.split()
            
            # if there is some headline with no text starting on the same line:
            if line_split and line_split[-1][-1] == self.SEMICOLON:            
                while True:
                    next_line = next(line_iter)
                    next_lines.append(next_line)
                    
                    # skip empty lines
                    if len(next_line.split()) == 0:
                        continue

                    # if we encountered another headline(possibly with text)
                    if self.SEMICOLON in next_line:
                        break

                    # there is a line which is not a headline line
                    next_lines = [line + next_line]
                    break

            final_lines += next_lines

        # return processed text
        return '\n'.join(final_lines)

s = SemicolonParagraphPreprocessor()
text = "                                 Final Report\n\
 Examination:  Chest (PA And Lat)\n\
\n\
 Indication:  ___ year old woman with ?pleural effusion  // ?pleural effusion\n\
\n\
 Technique:  Chest PA and lateral\n\
\n\
 Comparison:  ___\n\
\n\
 Findings:\n\
\n\
 Cardiac size cannot be evaluated.  Large left pleural effusion is new.  Small\n\
 right effusion is new.  The upper lungs are clear.  Right lower lobe opacities\n\
 are better seen in prior CT.  There is no pneumothorax.  There are mild\n\
 degenerative changes in the thoracic spine\n\
\n\
 Impression:\n\
\n\
 Large left pleural effusion"

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
 There is no focal consolidation, effusion, or pneumothorax. The\n\
 cardiomediastinal silhouette is normal.  Again seen are multiple clips\n\
 projecting over the left breast and remote left-sided rib fractures.  No free\n\
 air below the right hemidiaphragm is seen.\n\
 \n\
 IMPRESSION: \n\
 \n\
 No acute intrathoracic process."

#print(s.preprocess(text2))