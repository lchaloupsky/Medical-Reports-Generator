#!/usr/bin/python3

import re

from preprocessors.preprocessor import Preprocessor

class TimePreprocessor(Preprocessor):
    def preprocess(self, text: str) -> str:
        '''Finds all incorrectly written english form times(not dates) and corrects them.
        For example: 945AM -> 9:45 AM
        '''
        for found in reversed(list(re.finditer(r'(a|A|p|P)\.{,1}(m|M)\.{,1}', text))):
            # skip those where the found part is not preceded with a number
            if not self._is_time(text, found.start()): 
                continue
            
            digits_start, digits_end = self._get_digits_range(text, found.start(), found.end())
            text = text[:digits_start] + self._get_correct_time(text, found.start(), found.end(), digits_start, digits_end) + text[found.end():]

        # delete all common whitespaces and return processed text
        return text

    def _get_correct_time(self, text: str, start: int, end: int, digits_start: int, digits_end: int):
        corrected = text[digits_start : digits_end + 1]
        # incorrect time format
        if ":" not in text[digits_start : digits_end + 1]:
            corrected = "".join(corrected.split())

            # some different meaning - time can have at most 4 digits 
            if len(corrected) > 4: 
                return text[digits_start : digits_end + 1] + text[start : end]

            corrected = f"{corrected[:-2]}:{corrected[-2:]}" if len(corrected) > 2 else corrected

        # return final corrected time text
        return f"{corrected} {text[start:end]}"

    def _get_digits_range(self, text: str, start: int, end: int):
        digits_start, digits_end = start, None
        for i in range(start - 1, -1, -1):
            if str.isspace(text[i]) or text[i] == ":": 
                continue
            elif str.isdigit(text[i]): 
                digits_start = i
                digits_end = i if digits_end == None else digits_end
            else: 
                break

        return digits_start, digits_end

    def _is_time(self, text: str, start: int) -> bool:
        for i in range(start - 1, -1, -1):
            if str.isspace(text[i]): 
                continue

            if str.isdigit(text[i]): 
                return True
            else: 
                return False 

        return False

# TODO: 12pmplease/3pmasses examples? - should it be splitted or not?

#t = TimePreprocessor()
#print(t.preprocess("945AM ds 945 AM da \n \n 10 10pm 5:06 A.M. 5:06AM \n \n dsad 4pmsad as 9AM I am happy"))
