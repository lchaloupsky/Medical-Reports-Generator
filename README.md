# Medical-Reports-Generator
## Overview
This repository contains source code and text for a diploma thesis dealing with the problem of automatic text report generation to **X-rays**.

:warning: The final task is not implemented yet. There are many subtasks which must be solved. They are also included in this repository.

## Repository layout
The repository contains 3 main parts
* **gpt2_utils** - contains code for fine-tuning GPT2 language model to any data it gets, for the diploma thesis it is used to fine-tune Czech GPT2 language model from English GPT2 using trasnfer learning
* **vzor-dp** - text part of the diploma thesis, as the thesis is written in English the text is contained in the **vzor-dp/en** folder
* **rest of the repository** - the rest of the repository contains code to automatic translation of any text using REST API of 2 translators
  * [**CUBBITT**](https://lindat.mff.cuni.cz/services/translation/) - Freely available translator without restrictions on its REST API created at the Charles University.
  * [**DeepL**](https://www.deepl.com/translator) - There is a way how to translate using reverse engeneering of the REST API used on their web page, but it has limitations for circa 25 consecutive calls in an hour.

## Instalation
### Python Requirements
*Note:* Each part of the repository has different requirements, please take a look inside them for more info.

* ### **Automatic translation**
  The automatic translation requires **Python 3.9** and higher.
* ### **GPT2**
  The GPT2 part requires **Python 3.8**.
* ### **Medical report generation**
  The Medical report generation requires **Python 3.8**.

Python dependencies in each part can be installed by running: 
```bash
pip3 install -r requirements.txt
```

### GPU requirements
To run the training or testing procedures on the GPU, the GPU must support appropriate CUDA and cuDNN versions. Both **GPT2** and **Medical report generation** parts were trained on **CUDA 11** and **cuDNN 8** versions.

### Download
```git
git clone https://github.com/lchaloupsky/Medical-Reports-Generator.git
```

## Usage
*Note:* Each part has different usage, please take a look inside them for more info.

Main script for automatic translation is `dataset_trasnlate.py`. Please take a look in the script itself for supported arguments - all of them are described together with their accepted values and types.