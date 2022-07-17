# Medical Reports Translation

## Overview
Medical Reports Translation serves for the purposes of medical datasets translation.

## Instalation
### Requirements
* Python 3.9.7 as the code was written in it.

### Dependencies
Install all required Python dependencies.
```bash
pip3 install -r requirements.txt
```
*Note*: The *requirements_full.txt* file contains all packages installed locally during the developement phase.

## Structure
The repository is structured as follows:

### extractors
* Contains `Extractor` classes. 
* These serves for the text extraction from the dataset files as not all datasets are just simple text files. You can define your own class by inheriting the `Extractor` abstract class.
* Currently supported are **OpenI**(Indiana University chest X-ray) and **MIMIC-CXR** datasets.

### preprocessors
* Contains `Preprocessor` classes. 
* These serves for the text preprocessing. Dataset texts may contain undesirable pattern or elements that can be corrected by preprocessing. You can define your own class by inheriting the `Preprocessor` abstract class.
* Each preprocessor is described inside its own class.

### translators 
* Contains `Translator` classes. 
* These serves for the text translation. You can define your own class by inheriting the `Translator` abstract class.
* Currently supported are **CUBBITT** and **DeepL**(unusable in real terms due to strict DeepL limits) translators.

### tests
* Contains `pytest` tests for **preprocessors** and **extractors** validation.

### utils
* Additional utilies and classes used in the translation process.

## Run Translation Process
The translation of the dataset can be run in the following way. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
python3 dataset_translate.py [-h] 
        [--translator {cubbitt,deepl}] 
        [--dataset {mimic,openi}] 
        [--data DATA]
        [--preprocess {lowercase,pipeline,none}] 
        [--preprocess_only PREPROCESS_ONLY]
        [--anonymous_seq ANONYMOUS_SEQ]
``` 
The overall script pipeline is currently focused on translation from English to Czech. For other target language, the files need to be updated accordingly.

### Generated files and directories
The script generates the following directories:
#### translations
* contains all translation runs identified by the translator, dataset, time etc. The translations are contained inside the corresponding run directory. All translations are **.txt** files
#### logs
* contains additional logs identified in the similar way as above.
  
## Run Tests
To run the tests:
```bash
python3 -m pytest tests
``` 