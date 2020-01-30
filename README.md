# VW Golf image classifier

CNN classifier to predict VW Golf models. Models currently supported:
* Golf 5
* Golf 6
* Golf 7
* Golf 7.5

For more info on the [dataset](https://github.com/GerardWalsh/golf-classifier/wiki/Dataset-discussion) and [models](https://github.com/GerardWalsh/golf-classifier/wiki/Model-discussion), check out the respective Wikis. 

# Setup

Tested and trained on Ubuntu 18.04, Driver Version: 430.64, CUDA 10.0, Tensforflow 2.0.0 and Keras 2.3.1. See install instructions [here](https://github.com/GerardWalsh/golf-classifier/wiki/Setup). 

Install the Python packages with:

```
$ (venv) pip install -r requirements.txt
```

# Training

To train the providided dataset, use: 

```
$ (venv) python train.py
```

# Testing

A pretrained model is to be included. 
