# Medical Reports Generation

## Overview
Medical Reports Generation part contains all source code and data used for the training and evaluation of the medical reports generation models.

The medical reports generation models source code is based on this repository: [GPT2-Chest-X-Ray-Report-Generation (CDGPT2)](https://github.com/omar-mohamed/GPT2-Chest-X-Ray-Report-Generation)

The full OpenI(Indiana chest X-ray) dataset can be found [here](https://openi.nlm.nih.gov/faq#collection).

## Trained Models
All trained Czech GPT-2 models are available [here](https://owncloud.cesnet.cz/index.php/s/ckBG2kx2WNVi2VP).

## Requirements
All code was tested with the following configuration:
* Python - 3.8.6 GCCcore-10.2.0
* CUDA - 11.4.1
* cuDNN - 8.2.2.26

### Instalation
Install all required Python dependencies.
```bash
pip3 install -r requirements.txt
```
*Note*: The *requirements_full.txt* file contains all packages installed on the computing cluster during the developement phase.

## czech_csv_train_data
Contains modified original `.csv` files to Czech containig reports used for the training and testing.

## eval_final
Contains all data used for the manual evaluation of the generated reports. To prepare these files run the `prepare_eval_final.py` script. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
python3 prepare_eval_final.py [-h] [--img_dir IMG_DIR] 
        [--models_outputs MODELS_OUTPUTS]
        [--reports_orig_dir REPORTS_ORIG_DIR] 
        [--reports_trans_dir REPORTS_TRANS_DIR]
        [--csv_file CSV_FILE]
```

## model_outputs
Contains predicted outputs of all trained models.

## NLG Evaluation
For the final evaluation of our models is used this [NLG repository](https://github.com/Maluuba/nlg-eval). This project provides multiple NLP metrics commonly used for this task. However, it is normally intended for evaluating English texts and we need to update it for Czech language.

The `NLG_eval/package` contains modified packege used on the computing cluster for the final evaluation. The source files and data were modified as follows.

Otherwise the package source code must be modified in the following way. For METEOR evaluation, iside the package, modifiy in the `nlgeval/pycocoevalcap/meteor/meteor.py` the `meteor_cmd` array and replace the "en" with "cz". Furthermore, download https://github.com/cmu-mtlab/meteor/blob/master/data/paraphrase-cz.gz file to the `nlgeval/pycocoevalcap/meteor/data` directory.

For the embedding-based metrics we have to replace the English GloVe vectors with the Czech word2vec ones. As these are very large and cannot be attached in this repository, it is necessary to run the following pipeline. To replace the original English GloVe vectors with the Czech word2vec vectors, run following commands inside the `$HOME/.cache/nlgeval` directory after installing the **nlgeval** package:
```bash
# using bash
cd $HOME/.cache/nlgeval
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz
gunzip cc.cs.300.bin.gz
```
```python
# using python3
import nlgeval.word2vec.generate_w2v_files as nwv
nwv.txt2bin("cc.cs.300.vec.txt")
``` 
```bash
# using bash
mv cc.cs.300.vec.bin glove.6B.300d.model.bin
mv cc.cs.300.vec.bin.vectors.npy glove.6B.300d.model.bin.vectors.npy
```

## Modify Original Prepared .csv Files
To run the training on the Czech data, it is necessary to prepare training and testing files with Czech translations. The original solution uses 3 `.csv` files. In order to modify them and replace the English reports with the corresponding Czech translations run the following script. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
modify_csv_to_translations.py [-h]
        [--original_csv_dir ORIGINAL_CSV_DIR]
        [--translations_dir TRANSLATIONS_DIR]
        [--output_dir OUTPUT_DIR]
```

## Predict Reports
In order to predict report for single X-ray image, put and run the `predict_for_img.py` script in the `GPT2-Chest-X-Ray-Report-Generation` directory. The reason for its current placement is to distinguish our code from the code of the original repository. Before running the script, follow the steps described below needed for training and testing processes.

## GPT2-Chest-X-Ray-Report-Generation
Original repository for the medical report generation task from [GPT2-Chest-X-Ray-Report-Generation (CDGPT2)](https://github.com/omar-mohamed/GPT2-Chest-X-Ray-Report-Generation). In order to run on the computing cluster and the newer version of the tensorflow library, the following files had to be modified accordingly.
* `train.py`
* `test.py`

Furthermore, in order to run the training, testing or prediction on the custom GPT-2 models, perform these steps:
1. Follow original *README* steps
2. Put your GPT-2 model directory inside the top level of the `GPT2-Chest-X-Ray-Report-Generation` directory
3. Change "distilgpt2" and "gpt2" model and tokenizer string identifiers to the name of your GPT-2 model in these files - `train.py`, `test.py`, `tokenizer wrapper.py`
4. Switch the original .csv files in the `IU-XRay` directory with the ones in the `czech_csv_train_data` directory (if not already switched - both the original and the modified Czech `.csv` files are currently stored there)
5. Change hyperparameters in `configs.py` to the desired values

After that the training, testing or predtiction process can be run by running its corresponding command:
```bash
python3 train.py
python3 test.py
python3 predict_for_img.py
```

Checkpoints are stored in the `checkpoints/CDGPT2` directory. After training finishes, the latest model is stored directly inside this folder. The best one is inside the `checkpoints/CDGPT2/best` subdirectory. 

The model present in the `checkpoints/CDGPT2` is loaded both for the testing and prediction purposes. For testing and prediction, it is important to have the appropriate GPT-2 tokenizer string identifier in the `tokenizer wrapper.py`. Otherwise, nonsense text might be predicted.

### Running On Cluster
The `run.sh` and `run_test.sh` scripts were used to run the training and testing on the computing cluster for one specific configuration. These are just examples, how the jobs can be submitted.