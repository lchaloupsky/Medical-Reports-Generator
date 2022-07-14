#!/usr/bin/python3

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--original_csv_dir', default=".", type=str, help="Original csv files directory.")
parser.add_argument('--translations_dir', default="../translations_openi_cubbit_cased/data/ecgen-radiology", type=str, help="Directory with the translated files.")
parser.add_argument('--output_dir', default=".", type=str, help="Final path to save the modified csv files.")

_FINDINGS = ["Nálezy:", "Zjištění:"]
_IMPRESSIONS = ["Imprese:", "Dojem:", "Dojmy:", "Nález:", "Údaje:", "Impression:", "Úvaha:", "Náraz:", "Úbytek:", "Úroveň:"]
_CSVS = ["testing_set", "training_set", "all_data"]
_DATA_FOLDER = "../translations_openi_cubbit_cased/data/ecgen-radiology"

'''
Notes: 2191.txt was manually fixed
'''

def _find_paragraph(data: str, headlines: list[str]):
    for f in headlines:
        # find headline
        start = data.find(f)
        if start == -1: continue

        # find end of the paragraph and extract the text
        text = data[start + len(f):]
        end = text.find("\n\n")
        text = text[:end if end != -1 else len(text)]

        return text.strip()

    return None

def modify_csv_to_translations(args: argparse.Namespace) -> None:
    # TODO: use arguments instead of hard coded constants

    for csv_name in _CSVS:
        print(f"Transforming {csv_name}.csv")

        csv = pd.read_csv(f"{csv_name}.csv")
        for i, image in enumerate(csv["Image Index"]):
            # get image/report name
            image = image.split("_")[0].replace("CXR", "")

            # load data
            with open(f"{_DATA_FOLDER}/{image}.txt", "r", encoding="utf-8") as f:
                data = "".join(f.readlines())

            findings = _find_paragraph(data, _FINDINGS)
            impression = _find_paragraph(data, _IMPRESSIONS)

            # Not found any corresponding czech data - this is for debug only
            if findings == None and impression == None:
                print(f"DEBUG: Cannot find any data for: {image}")

            csv.loc[i, "Findings"] = f"startseq {findings} endseq"
            csv.loc[i, "Impression"] = f"startseq {impression} endseq"
            csv.loc[i, "Caption"] = f"startseq {chr(34) if impression and findings else ''}{impression}{chr(10) if impression and findings else ''}{findings}{chr(34) if impression and findings else ''} endseq" # chr(10) == \n, chr(34) == '"'

        csv.to_csv(f"{csv_name}_cz.csv", encoding="utf-8", sep=",", index=False, line_terminator="\n")

    print("Transformation done.")

if __name__ == "__main__":
    modify_csv_to_translations(parser.parse_args([] if "__file__" not in globals() else None))