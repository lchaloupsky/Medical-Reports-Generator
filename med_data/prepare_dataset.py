#!/usr/bin/python3

import argparse
import re

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='none', choices=['none', 'fix', 'remove'], type=str, help="The way incorrect dashes should be treated.")
parser.add_argument('--data', type=str, required=True, help="Dataset location path.")
parser.add_argument('--dest', type=str, required=True, help="Destination directory location path.")

_UNK_CHARS = {"�", "", "", "", "", "", "", "", "", "", "", "Ï", "ã", "â", "À", "Û", "Ô", "ø", "ì", "ù", "è", "‰", "˘", "@$#,/", "Ù", "Ë", "Ð", "¢", "Æ", "Å", "ª", "£"}
_REGEXES = [
    re.compile(r"\u00ad *"),            # hidden char
    re.compile(r" +(,)"),               # correct comma formatting
    re.compile(r"([\w]-) -([\w])"),     # fix incorrect formatting
    re.compile(r"([(]) +"),             # parenthesis formatting
    re.compile(r" +([)])"),             # parenthesis formatting
    re.compile(r"\?{4,}"),              # more than 3 "?"
    re.compile(r"\.{4,}"),              # more than 3 "."
    re.compile(r"([\w]/) "),            # fix incorrect slash
    re.compile(r"\[[0-9, -]+\]"),       # remove citations
    re.compile(r"([^0-9]) (:)")         # fix colon formatting
]

_REGEXES_TO_DEL = [
    re.compile(r"\w\?\w"),              # another type of text with not recognized chars
    re.compile(r"^[\W]+$")              # lines without normal text
]

def process_file(file_path: Path, dest_path: Path):
    dest_file = dest_path/file_path.parent/f"{file_path.stem}_fixed{file_path.suffix}"
    dest_file.parent.mkdir(exist_ok=True, parents=True)

    with dest_file.open("w", encoding="utf-8") as dest_f:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                # filter out lines containing invalid chars
                if len(_UNK_CHARS.intersection(line)) > 0:
                    continue

                should_skip = False
                for reg in _REGEXES_TO_DEL:
                    if reg.search(line) is not None:
                        should_skip = True
                        break

                if should_skip:
                    continue

                # replace different types of dashes and spaces
                line = line \
                    .replace("‑", "-") \
                    .replace("–", "-") \
                    .replace("−", "-") \
                    .replace(u"\u00a0", " ") \
                    .replace(u"\u200a", " ") \
                    .replace("", " ") \
                    .replace("“", "\"") \
                    .replace("„", "\"") \
                    .replace("”", "\"") \
                    .replace("‟", "\"") \
                    .replace("‘", "\'") \
                    .replace("‚", "\'") \
                    .replace("’", "\'") \
                    .replace("‛", "\'") \
                    .replace("`", "\'") \
                    .replace(r"''", "\'") \
                    .replace("…", "...") \
                    .replace(" ?", "?") \
                    .replace(" .", ".") \
                    .replace(" !", "!") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace("", "") \
                    .replace(u"\u0083", " ") \
                    .replace("-• ", "") \
                    .replace("﻿", "") \
                    .replace("®", "") \
                    .replace("©", "") \
                    .replace("›", ">") \
                    .replace("‹", "<") \
                    .replace("•", "-") \
                    .replace("↑", "|") \
                    .replace("↓", "|") \
                    .replace("ı", ",") \
                    .replace("¸", ",") \
                    .replace("■■", "")

                line = " ".join(line.split())

                # apply all regexes
                for reg in _REGEXES:
                    line = "".join(reg.split(line))

                # filter too short lines
                if len(line) <= 8:
                    continue

                # write modified text
                dest_f.write(line + "\n")
                dest_f.flush()

def _get_regex_dash(args: argparse.Namespace):
    # regex to fix/remove incorrectly formated dashes
    if args.action == "none":
        return
        
    _REGEXES.append(re.compile(r"(\w+-) ") if args.action == "fix" else re.compile(r"(\w+)- "))

def main(args: argparse.Namespace):
    data_path, dest_path = Path(args.data), Path(args.dest)
    if not data_path.exists():
        print(f"ERROR: {data_path.absolute} is not a valid path. Please specify a valid path.")
        return

    _get_regex_dash(args)
    dest_path.mkdir(parents=True, exist_ok=True)
    if data_path.is_file():
        process_file(data_path, dest_path)
        return

    for file in data_path.iterdir():
        if file.name.endswith(".py") or file.name.startswith("README"):
            continue

        process_file(file, dest_path)

if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))