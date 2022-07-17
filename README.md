# Medical-Reports-Generator
## Overview
**Medical Reports Generator** repository contains the source code and text for a diploma thesis dealing with the problem of automatic generation of text reports for **X-ray** images.

## Trained Models
All trained models are available [here](TODO).

## Repository layout
The repository comprises of 4 parts:
* [`czech_gpt2`](./czech_gpt2/) - contains code for fine-tuning GPT2 language model to any data it gets, for the diploma thesis it is used to fine-tune Czech GPT2 language model from English GPT2 using trasnfer learning
* [`medical_reports_generation`](./medical_reports_generation/) - contains all source code and data related to the training and evaluation of the medical reports generation models
* [`medical_reports_translation`](./medical_reports_translation/) - this repository contains code to automatic machine translation.
* [`vzor-dp`](./vzor-dp/) - text part of the diploma thesis, as the thesis is written in English the text is contained in the **vzor-dp/en** folder as **thesis.pdf**

See the individual parts for more information.

## Instalation
### Download
```git
git clone https://github.com/lchaloupsky/Medical-Reports-Generator.git
```

### Python Requirements
:warning: Each part of the repository may have different requirements, please take a look inside of them for more information.

### GPU requirements
To run the training or testing procedures on the GPU, the GPU must support appropriate CUDA and cuDNN versions. Both **GPT2** and **Medical report generation** parts were trained on **CUDA 11** and **cuDNN 8** versions. 

More details are available inside correspoding parts.