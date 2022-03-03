#!/usr/bin/python3

import argparse
import os
import datetime
import logging
import time
import concurrent.futures as cf

from concurrent.futures import ThreadPoolExecutor
from preprocessors import *
from translators import *
from extractors import *

parser = argparse.ArgumentParser()
parser.add_argument('--translator', default='cubbit', choices=['cubbitt', 'deepl'], type=str, help="Translator used for reports translation.")
parser.add_argument('--dataset', default='mimic', choices=['mimic', 'openi'], type=str, help="Dataset intended for translation.")
parser.add_argument('--data', default=None, type=str, help="Dataset location path.")
parser.add_argument('--preprocess', default="pipeline", choices=["lowercase", "pipeline", "none"], type=str, help="Dataset preprocessing mode.")
parser.add_argument('--anonymous_seq', default=None, type=str, help="A common start of every anonymous character sequence in reports.")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/output_{}.log'.format(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")), encoding='utf-8', filemode='w', level=logging.DEBUG, datefmt='%Y-%d-%m %H:%M:%S')

MAX_REPEAT_COUNT = 5
PREPROCESSORS: list[Preprocessor] = []
DATASET_FOLDERS = {"openi": "ecgen-radiology", "mimic": "files"}

def _preprocess(text: str) -> str:
    for preprocessor in PREPROCESSORS:
        text = preprocessor.preprocess(text)
    
    return text

def _log_error(filename: str, response: Response):
    logging.error("Maximum repeat count reached. Cannot translate report: {}\n Reason code: {}\n Reason: {}".format(filename, response.status_code, response.reason))

def translate_report(translator: Translator, extractor: Extractor, filename: str, destination: str):
    # extract text from a file
    text = extractor.extract_report(filename)
    #if "1022" in filename:
    #    print(text)

    # preprocess text
    text = _preprocess(text)

    repeats = 0
    while True:
        # send request to translate
        response = translator.translate(text)
        response.encoding = 'utf-8'     #response.apparent_encoding

        # check response
        if response.status_code != 200:
            # wait between repetition
            repeats += 1
            time.sleep(repeats) # FOR DEEPL -> modify to 3 * repeats

            if repeats == MAX_REPEAT_COUNT:
                print("Cannot translate report: {}. Reason: {}".format(filename, response.reason))
                _log_error(filename, response)
                return 0

            continue

        # get translated text from response
        translation = translator.get_text(text, response)

        # save into a file
        os.makedirs(destination, exist_ok=True)
        with open("{}/{}.txt".format(destination, os.path.splitext(os.path.basename(filename))[0]), "w", encoding="utf-8") as file:
            file.write(translation)

        return 1

def _get_translator(args: argparse.Namespace) -> Translator:
    return DeepLTranslator() if args.translator == "deepl" else CubbittTranslator()

def _get_extractor(args: argparse.Namespace) -> Extractor:
    return OpenIReportExtractor() if args.dataset == "openi" else MimicReportExtractor()

def _get_dataset_location(args: argparse.Namespace) -> str:
    if args.data != None:
        return args.data

    return "./data/{}".format(DATASET_FOLDERS[args.dataset])

def _get_preprocessors(args: argparse.Namespace, extractor: Extractor) -> str:
    if args.preprocess == "lowercase":
        PREPROCESSORS.extend([
            LowercasePreprocessor(), 
            CapitalizeStartsPreprocessor()
        ])
    elif args.preprocess == "pipeline":
        PREPROCESSORS.extend([
            LineStartsPreprocessor(), 
            AnonymousSequencePreprocessor(args.anonymous_seq), 
            UnitsPreprocessor(),
            TrueCasingPreprocessor(extractor, _get_dataset_location(args), args.dataset), 
            SemicolonParagraphPreprocessor(),
            CapitalizeStartsPreprocessor(),
            TimePreprocessor(), 
            WhitespacesSquashPreprocessor()
        ])

def main(args: argparse.Namespace):
    # create dir for current translations
    destination = "./translations/translations_{}_{}_{}".format(args.dataset, args.translator, datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    os.makedirs(destination, exist_ok=True) 

    # create background jobs and translate
    futures = []
    translator = _get_translator(args)
    extractor = _get_extractor(args)
    _get_preprocessors(args, extractor)

    final_sum, final_destination = 0, None
    i = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        try:
            for subdir, _, filenames in os.walk(_get_dataset_location(args)):
                final_destination = os.path.join(destination, os.path.normpath(subdir))
                final_sum += len(filenames)
                for filename in filenames:
                    #i += 1

                    #if "960" in filename:
                    #translate_report(translator, extractor, os.path.join(subdir, filename), final_destination)
                    
                    #if i == 100:
                    #    break
                    
                    # create new job
                    futures.append(pool.submit(translate_report, translator, extractor, os.path.join(subdir, filename), final_destination))
        
            dones = 0
            for _ in cf.as_completed(futures):
                dones += 1
                print("Currently processed: {}".format(dones), end='\r')

            print("\nTranslation done.")
        except KeyboardInterrupt as e:
            # stop all running threads
            pool.shutdown(wait=False, cancel_futures=True)
            pool._threads.clear()
            cf.thread._threads_queues.clear()
            raise e

    done_sum = sum(map(lambda f: f.result(), futures))
    if done_sum != final_sum:
        print("{} report(s) cannot be translated, please check the log file for additional information.".format(final_sum - done_sum))

    logging.shutdown()

if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))
