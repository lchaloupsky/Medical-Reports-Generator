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
from utils import *

parser = argparse.ArgumentParser(description="The script serves for the purpose of translating medical report datasets.\n Note: The DeepL translator is not practically usable.")
parser.add_argument('--translator', default='cubbitt', choices=['cubbitt', 'deepl'], type=str, help="Translator used for reports translation.")
parser.add_argument('--dataset', default='openi', choices=['mimic', 'openi'], type=str, help="Dataset intended for translation.")
parser.add_argument('--data', default=None, type=str, help="Dataset location path.")
parser.add_argument('--preprocess', default="pipeline", choices=["lowercase", "pipeline", "none"], type=str, help="Dataset preprocessing mode.")
parser.add_argument('--preprocess_only', default=False, type=bool, help="Flag indicating whether the data should be only preprocessed without translation.")
parser.add_argument('--anonymous_seq', default=None, type=str, help="A common start of every anonymous character sequence in reports.")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/output_{}.log'.format(datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")), encoding='utf-8', filemode='w', level=logging.DEBUG, datefmt='%Y-%d-%m %H:%M:%S')

MAX_REPEAT_COUNT: int = 10
PREPROCESSORS: list[Preprocessor] = []
DATASET_FOLDERS: dict[str, str] = {"openi": "ecgen-radiology", "mimic": "files"}
MAX_CONCURRENT_TASK_SUBMITTED_COUNT: int = 64

def _preprocess(text: str) -> str:
    for preprocessor in PREPROCESSORS:
        text = preprocessor.preprocess(text)
    
    return text

def _log_error(filename: str, response: Response):
    logging.error("Maximum repeat count reached. Cannot translate report: {}\n Reason code: {}\n Reason: {}".format(filename, response.status_code, response.reason))

def translate_report(translator: Translator, extractor: Extractor, filename: str, destination: str, preprocess_only: bool):
    '''
    Translates medical report and saves it locally.

    :param translator: Translator to be used for the report translation.
    :param extractor: Specific dataset translator
    :param filename: Report to be translated
    :param destination: Destination where the translated text should be saved
    :param preprocess_only: Flag indicating whether the text should be preprocessed only and not translated
    :return: 1/0 if the report was translated or not
    '''

    # extract text from a file
    text = extractor.extract_report(filename)

    # preprocess text
    text = _preprocess(text)

    repeats = 0
    while not preprocess_only:
        # send request to translate
        response = translator.translate(text)
        response.encoding = 'utf-8'     #response.apparent_encoding

        # check response
        if response.status_code != 200:
            # wait between repetition
            repeats += 1
            time.sleep(repeats * 5 if type(translator) is DeepLTranslator else repeats)

            if repeats == MAX_REPEAT_COUNT:
                print("Cannot translate report: {}. Reason: {}".format(filename, response.reason))
                _log_error(filename, response)
                return 0

            continue

        # get translated text from response
        text = translator.get_text(text, response)
        break

    # save into a file
    os.makedirs(destination, exist_ok=True)
    with open("{}/{}.txt".format(destination, os.path.splitext(os.path.basename(filename))[0]), "w", encoding="utf-8") as file:
        file.write(text)

    return 1

def _get_translator(args: argparse.Namespace) -> Translator:
    return DeepLTranslator() if args.translator == "deepl" else CubbittTranslator()

def _get_extractor(args: argparse.Namespace) -> Extractor:
    return OpenIReportExtractor() if args.dataset == "openi" else MimicReportExtractor()

def _get_dataset_location(args: argparse.Namespace) -> str:
    if args.data != None:
        return args.data

    return "./data/{}".format(DATASET_FOLDERS[args.dataset])

def _get_skip_regex(args: argparse.Namespace) -> str:
    return None if args.dataset == "openi" else r" {3,}FINAL +.*\n"

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
            TrueCasingPreprocessor(extractor, _get_dataset_location(args), args.dataset, _get_skip_regex(args)), 
            SemicolonParagraphPreprocessor(),
            CapitalizeStartsPreprocessor(),
            TimePreprocessor(), 
            WhitespacesSquashPreprocessor()
        ])

def _wait_for_completion(futures: list[cf.Future], progress_bar: ProgressBar) -> int:
    def update(f: cf.Future) -> int:
        progress_bar.update()
        return f.result()

    return sum([update(f) for f in cf.as_completed(futures)])

def main(args: argparse.Namespace):
    # create dir for current translations
    destination = "./translations/translations_{}_{}_{}".format(args.dataset, args.translator, datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    os.makedirs(destination, exist_ok=True) 

    # create background jobs and translate
    futures = []
    translator = _get_translator(args)
    extractor = _get_extractor(args)
    _get_preprocessors(args, extractor)

    total, done_sum, final_destination = 0, 0, None
    print(f"Translating '{args.dataset}' dataset.")
    with ThreadPoolExecutor(max_workers=32) as pool, ProgressBar(get_total_dir_size(_get_dataset_location(args)), 1) as progress_bar:
        try:
            for subdir, _, filenames in os.walk(_get_dataset_location(args)):
                final_destination = os.path.join(destination, os.path.normpath(subdir))
                for filename in filenames:                    
                    # create new job
                    futures.append(pool.submit(translate_report, translator, extractor, os.path.join(subdir, filename), final_destination, args.preprocess_only))
                    if (total := total + 1) % MAX_CONCURRENT_TASK_SUBMITTED_COUNT == 0:
                        done_sum += _wait_for_completion(futures, progress_bar)
                        futures = []
            
            done_sum += _wait_for_completion(futures, progress_bar)                 

        except KeyboardInterrupt as e:
            # stop all running threads
            pool.shutdown(wait=False, cancel_futures=True)
            pool._threads.clear()
            cf.thread._threads_queues.clear()
            raise e

    print(f"The translation of '{args.dataset}' dataset completed.")
    if done_sum != total:
        print(f"{total - done_sum} report(s) cannot be translated, please check the log file for additional information.")

    logging.shutdown()

if __name__ == '__main__':
    main(parser.parse_args([] if "__file__" not in globals() else None))
