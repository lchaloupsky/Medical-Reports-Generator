# Czech GPT-2

## Overview
This part contains code for fine-tuning the GPT-2 language models to any specified data. For the purposes of the diploma thesis it is used to fine-tune the Czech GPT-2 language model from English GPT2 using the trasnfer learning.

## Prerequisites
All code was tested with the following configuration:
* Python - 3.8.6 GCCcore-10.2.0
* CUDA - 11.1
* cuDNN - 8.0.5 (8005)

### Instalation
Install all required Python dependencies.
```bash
pip3 install -r requirements.txt
```

## DatasetsWrapper usage
The usage of DatasetsWrapper class is identical to the usage of the original classes of the **datasets** package. The DatasetsWrapper encapsulates these classes and adds additional features:
- direct text iteration
- direct text batch iteration
- control charactes filtering
- option to save the dataset directly into one local text file

The features are very useful if one works with datasets in streaming mode.

Example:
```python
from datasets_wrapper import DatasetsWrapper

# same definition as for the datasets classes
dataset = DatasetsWrapper("oscar", name="unshuffled_deduplicated_cs", split="train", streaming=True)
dataset = dataset.filter(lambda x: len(x["text"]) >= 1200)

# direct text iteration
for text in dataset:
    # work directly on the texts
    pass

# save dataset to local text file
dataset.save_to_file("storage/data/oscar/oscar_local.txt", text_delim="<###|---text_delimiter---|###>")
```

## Preparation of data for training
The data can be prepared into required format in the following way:
```python
import gpt2_data_utils as gdu

gdu.create_aggregates_from_texts("storage/data/oscar", text_delim="<###|---text_delimiter---|###>")
```

## Training
For the fine-tuning process, run the following script. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
python3 gpt2_train.py [-h] [--type {gradual,full}] 
            [--model MODEL] [--max_len MAX_LEN] 
            [--pretrained_weights PRETRAINED_WEIGHTS] 
            [--dataset DATASET] [--data_path DATA_PATH] 
            [--batch_size BATCH_SIZE]
            [--train_data_ratio TRAIN_DATA_RATIO] 
            [--sequence_length SEQUENCE_LENGTH] [--debug DEBUG] 
            [--resume_training RESUME_TRAINING] 
            [--find_learning_rates FIND_LEARNING_RATES]
            [--find_learning_rates_epoch FIND_LEARNING_RATES_EPOCH]
            [--pretokenize_data PRETOKENIZE_DATA] 
            [--save_checkpoints SAVE_CHECKPOINTS] 
            [--learning_rates LEARNING_RATES [LEARNING_RATES ...]]
```

## Text generation
For the text generation purposes, use following script. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
python3 gpt2_generate.py [-h] [--max_len MAX_LEN] 
            [--model MODEL] 
            [--top_k TOP_K] 
            [--top_p TOP_P] 
            [--repetition_penalty REPETITION_PENALTY] 
            [--temperature TEMPERATURE] 
            [--do_sample DO_SAMPLE]
            [--num_return_sequences NUM_RETURN_SEQUENCES]
            [--num_beams NUM_BEAMS]
            [--length_penalty LENGTH_PENALTY] 
            [--bos_token_id BOS_TOKEN_ID] 
            [--eos_token_id EOS_TOKEN_ID]
            [--pad_token_id PAD_TOKEN_ID]
```