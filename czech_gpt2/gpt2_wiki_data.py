#!/usr/bin/python3
from fastai.basics import *

import re
import pandas as pd
import argparse
import sys
import csv

# script arguments
parser = argparse.ArgumentParser(description="Creates wiki files suitable for the GPT-2 training process.")
parser.add_argument('--lang', default='cs', type=str, help="The wiki language to be processed.", required=True)

def download_url(url, path):
    '''
    Dowloads wiki dump.

    :param url: Wikidump url
    :param path: Path where to save the dump
    '''
    resp = requests.get(url, stream=True)
    with path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def get_wiki(path, lang):
    '''
    Gets the wiki for specific language and preprocesses it.

    :param path: Path where the processing should take its place
    :param lang: Wikidump language
    '''
    name = f"{lang}wiki"
    if (path/name).exists():
        print(f"{path/name} already exists, not downloading.")
        return

    xml_fn = f"{lang}wiki-latest-pages-articles.xml"
    zip_fn = f"{xml_fn}.bz2"

    if not (path/xml_fn).exists():
        print(f"Downloading {zip_fn}.")
        download_url(f"https://dumps.wikimedia.org/{name}/latest/{zip_fn}", path/zip_fn)
        print(f"Unzipping {zip_fn}.")
        bunzip(path/zip_fn)

    # Change working directory to `path`
    prev_cwd = Path.cwd()
    os.chdir(path)
    
    # Get wikiextractor
    if not (path/"wikiextractor").exists(): 
        os.system("git clone https://github.com/attardi/wikiextractor.git")

    # Extraction
    print(f"Extracting {xml_fn}.")
    os.system(f"python{'' if sys.platform.startswith('win') else '3'} -m wikiextractor.wikiextractor.WikiExtractor --processes 4 --no-templates -b 100G -q {xml_fn}")
    shutil.move(str(path/"text/AA/wiki_00"), str(path/name))
    shutil.rmtree(path/"text")
    
    # Return working directory to previous
    os.chdir(prev_cwd)

def split_wiki(path, lang):
    '''
    Splits wiki into individual files.

    :param path: Path where the processing should take its place
    :param lang: Wiki language
    :return: Path to wiki files
    '''
    dest = path/"temp"
    if dest.exists():
        print(f"{dest} already exists, not splitting.")
        return dest

    f = None
    name = f"{lang}wiki"
    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    with (path/name).open(encoding="utf8") as lines:
        for l in lines:
            if l.startswith('<doc id="'):
                title = title_re.findall(l)[0].replace('/','_') \
                    .replace('?', '').replace('\\', '_').replace(':', '').replace('*', '') \
                    .replace('|', '').replace('\"', '').replace('<', '').replace('>', '')

                if len(title) > 150: continue
                if f: f.close()
                f = (dest/f'{title}.txt').open('w', encoding="utf8")
            else: 
                f.write(l)
        f.close()
        
    return dest
        
def get_one_clean_txt_file(dest,lang):
    '''
    Creates single clean .txt file comprising of all wiki texts.

    :param dest: Wiki files location
    :param lang: Wiki lanugage
    '''
    fname = f"{lang}wiki_agg.txt"
    # regex to delete </doc>
    doc_re = re.compile(rf"([\w\W]*)<\/doc>")
    
    all_texts = ""
    for l in dest.ls():
        # open file and get content without first line which is the title
        with l.open('r+', encoding="utf-8") as f:
            f.readline()
            text = f.read()
            if len(text) < 1800: 
                continue

            # get content without </doc> and delete empty line and whitespaces at the head and tail
            text = doc_re.findall(text)[0].strip()

            # concatenate text
            all_texts += f"{text}\n"
  
    # Save final txt
    with open(dest.parent/fname, "w", encoding="utf-8") as txt_f: 
        txt_f.write(all_texts)
    print(f"All texts from '{lang}' wikipedia saved in the file {dest.parent/fname}.\n")

def get_one_clean_csv_file(dest,lang):
    '''
    Creates single clean .csv file comprising of all wiki texts.

    :param dest: Wiki files location
    :param lang: Wiki lanugage
    '''
    fname = f"{lang}wiki_agg.csv"
    # regex to delete </doc>
    doc_re = re.compile(rf"([\w\W]*)<\/doc>")
    
    all_texts = list()
    for l in dest.ls():
        # open file and get content without first line which is the title
        with l.open("r+", encoding="utf-8") as f:
            f.readline()
            text = f.read()
            if len(text) < 1800: 
                continue

            # get content without </doc> and delete empty line and whitespaces at the head and tail
            text = doc_re.findall(text)[0].strip()

            # append text
            all_texts.append(text)
  
    # Create the pandas DataFrame 
    df = pd.DataFrame(all_texts, columns = ["text"])
    
    # Save final csv
    df.to_csv(dest.parent/fname, index=False, encoding="utf-8")  
    print(f"All texts from '{lang}' wikipedia saved in the file {dest.parent/fname}.\n")

def get_clean_files(dest, lang):
    '''
    Creates clean .txt and .csv files comprising of all wiki texts.

    :param dest: Wiki files location
    :param lang: Wiki lanugage
    '''
    fname_txt, fname_csv = f"{lang}wiki_agg.txt", f"{lang}wiki_agg.csv"

    # regex to delete </doc>
    doc_re = re.compile(rf"([\w\W]*)<\/doc>")
    
    with open(dest.parent/fname_csv, mode="wt", newline='', encoding="utf-8") as csv_file, \
         open(dest.parent/fname_txt, mode="wt", newline='', encoding="utf-8") as txt_file:
        writer = csv.writer(csv_file)

        # csv header
        writer.writerow(["text"])
        all_texts_txt, all_texts_csv = "", []
        for i, l in enumerate(dest.ls(), start=1):
            if i % 100 == 0 and len(all_texts_csv) > 0:
                writer.writerows(all_texts_csv)
                txt_file.write(all_texts_txt)

                csv_file.flush()
                txt_file.flush()
                all_texts_txt, all_texts_csv = "", []

            # open file and get content without first line which is the title
            with l.open('r+', encoding="utf-8") as f:
                f.readline()
                text = f.read()
                if len(text) < 1800: 
                    continue

                # get content without </doc> and delete empty line and whitespaces at the head and tail
                text = doc_re.findall(text)[0].strip()
                all_texts_txt += f"{text}\n"
                all_texts_csv.append([text])

        writer.writerows(all_texts_csv)
        txt_file.write(all_texts_txt)  
  
    print(f"All texts from '{lang}' wikipedia saved in the files {dest.parent/fname_txt}, {dest.parent/fname_csv}.\n")

def main(args):
    base_bath = Path(__file__).parent.resolve()
    get_wiki(base_bath, args.lang) 
    get_clean_files(split_wiki(base_bath, args.lang), args.lang)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))