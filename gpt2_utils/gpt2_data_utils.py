import os
import csv
import argparse

from typing import List, Tuple

parser = argparse.ArgumentParser()
# arguments used during generation of text
parser.add_argument('--folder', default=".", type=str, help="Folder containing all data, that should be aggregated.", required=True)
parser.add_argument('--text_delim', default="<###|---text_delimiter---|###>", type=str, help="Text delimiter sequence indcating the end of the text within a file.")
parser.add_argument('--regenerate', default=False, type=bool, help="Flag indicating whether the aggregates should be regenerated even if already exist or not.")
parser.add_argument('--line_split', default=None, type=int, help="Number of lines after which a new entry should be created.")
parser.add_argument('--extensions', default=[], type=str, nargs="*", help="Additional file extensions that should be taken into account.")

def create_aggregates_from_texts(folder: str, text_delim: str, regenerate: bool = False, line_split: int = None, extensions: List[str] = []) -> Tuple([str, str]):
    '''
    Goes through specified 'folder' and reads every *txt* file which it can find anywhere in the 'folder' and generates an aggregation of all found files into one csv and one txt files. 
    Each file will be splitted with 'text_delim' into paragraphs. 
    Each paragraph will be then saved as one entry in the final csv file.
    If 'regenerate' is true, then the aggregated files are generated from the beginning regardless wheter they exist or not.

    Parameters:
        folder: str - the folder where to begin the search
        text_delim: str - the text delimiter to separate independent texts within one file
        regenerate: bool - the flag indicating if the final files should be generated even if they already exists.
        line_split: int - the number of lines after which the text should split.
        extensions: list[str] - other extensions that should be taken into account

    Returns:
        (str, str) - paths to aggregated *csv* and *txt* files
    '''
    if folder is None:
        raise ValueError("You have to specify a folder!")

    if not os.path.isdir(folder):
        raise ValueError("Specified folder does not exist!")

    if line_split is not None and line_split < 1:
        raise ValueError("Incorrect line_split value! The value has to be at least 1!")

    csv_file_name = f"{os.path.basename(folder.strip('/'))}_agg.csv"
    txt_file_name = f"{os.path.basename(folder.strip('/'))}_agg.txt"

    csv_file_path = os.path.join(folder, csv_file_name)
    txt_file_path = os.path.join(folder, txt_file_name)

    if not regenerate and (os.path.isfile(csv_file_path) or os.path.isfile(txt_file_path)):
        print("Final csv or txt file already exists. Returning.")
        return

    extensions = [".txt"] + extensions
    with open(csv_file_path, mode="wt", newline='', encoding="utf-8") as csv_file, \
         open(txt_file_path, mode="wt", newline='', encoding="utf-8") as txt_file:
        writer = csv.writer(csv_file)

        # csv header
        writer.writerow(["text"])

        # csv/txt body
        for dir, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename == csv_file_name or filename == txt_file_name:
                    continue

                if not any(filename.endswith(extension) for extension in extensions):
                    continue

                text = ""
                with open(os.path.join(dir, filename), encoding="utf-8") as file:
                    for i, line in enumerate(file, start=1):
                        if line.strip() == text_delim or (line_split is not None and i % line_split == 0):
                            writer.writerow([text.strip()])
                            txt_file.write(text.strip())

                            csv_file.flush()
                            txt_file.flush()

                            text = ""
                            continue
                        
                        text += line
                
                if text:
                    writer.writerow([text.strip()])
                    txt_file.write(text.strip())


    return (csv_file_path, txt_file_path)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    create_aggregates_from_texts(args.folder, args.text_delim, args.regenare, args.line_split, args.extensions)