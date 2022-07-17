#!/usr/bin/python3

from preprocessors.preprocessor import Preprocessor

class SemicolonParagraphPreprocessor(Preprocessor):
    '''Puts paragraps starting after a semicolon on the same line.'''
    _SEMICOLON = ":"

    def preprocess(self, text: str) -> str:
        # split text
        line_splits = text.split('\n')

        line_iter, final_lines = iter(line_splits), []
        for line in line_iter:
            next_lines = [line]
            line_split = line.split()
            
            # if there is some headline with no text starting on the same line:
            if line_split and line_split[-1][-1] == self._SEMICOLON:            
                while True:
                    # There is an empty heading as the final line of the text
                    if (next_line := next(line_iter, None)) is None:
                        break

                    next_lines.append(next_line)
                    
                    # skip empty lines
                    if len(next_line.split()) == 0:
                        continue

                    # if we encountered another headline(possibly with text)
                    if self._SEMICOLON in next_line:
                        break

                    # there is a line which is not a headline line
                    next_lines = [line + next_line]
                    break

            final_lines += next_lines

        # return processed text
        return '\n'.join(final_lines)