# Medical Reports Generation

## Overview
TODO

## NLG Evaluation
TODO description

To replace the original English GloVe vectors with the Czech word2vec vectors, run following commands inside the **$HOME/.cache/nlgeval** directory after installing the **nlgeval** package:
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