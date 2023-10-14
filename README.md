# A small Romanian BERT model
This is the repository with which to build a small Romanian BERT model based on the [CoRoLa corpus](https://corola.racai.ro/).  
It needs the Romanian aware WordPiece tokenizer that is avaiable [here](https://github.com/racai-ai/ro-wordpiece-tokenizer).

BERT parameters are:
* maximum sequence length `256`
* `L = 4` (4 stacked layers)
* `Η = 256` (size of the hidden layer)
* number of attention heads `8`

The model is pretrained with the MLM objective, `mlm_probability=0.15`.  
It takes approximately 30 days to train on an NVIDIA Quadro RTX 8000 with 48GB of RAM.  

# Installation
The Romanian BERT model requires the Romanian WordPiece tokenizer `ro-wordpiece-tokenizer` (see above).  
You have to clone it and this repository (`ro-corola-bert-small`) in the same folder and

```bash
export PYTHONPATH=../ro-wordpiece-tokenizer
```

while in the `ro-corola-bert-small` folder.
