# VW Golf image classifier

CNN classifier to predict VW Golf models. Models currently supported:
* Golf 5
* Golf 6
* Golf 7
* Golf 7.5

# Setup

Tested and trained on Ubuntu 18.04, Driver Version: 430.64, CUDA 10.0, Tensforflow 2.0.0 and Keras 2.3.1.

## GPU

 If you want to use the GPU backend for Tensorflow (and subsequently Keras), first install the following:

* Driver: https://tech.amikelive.com/node-731/how-to-properly-install-nvidia-graphics-driver-on-ubuntu-16-04/

* CUDA toolkit: https://tech.amikelive.com/node-859/installing-cuda-toolkit-9-2-on-ubuntu-16-04-fresh-install-install-by-removing-older-version-install-and-retain-old-version/

* CUDNN: https://tech.amikelive.com/node-679/quick-tip-installing-cuda-deep-neural-network-7-cudnn-7-x-library-for-cuda-toolkit-9-1-on-ubuntu-16-04/

* Follow the install instructions at https://www.tensorflow.org/install/pip?lang=python3, use a virtual environment and install with:

```
$ (venv) pip install --upgrade tensorflow-gpu
```

If the above GPU toolkits did not install successfully, try the instructions at https://www.tensorflow.org/install/gpu.

## CPU

Follow the install instructions at https://www.tensorflow.org/install/pip?lang=python3, once again use virtual environments and install with:

```
$ (venv) pip install --upgrade tensorflow
```

## Finally

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

A pretrained model is included. 
