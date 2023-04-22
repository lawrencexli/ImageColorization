# Mirflickr-25k Image Colorization

By Lawrence Li (ll3598@columbia.edu) and Isabella Cao (sc3912@columbia.edu) for COMS 4995 applied computer vision project

Spring 2023, Columbia University

## About the project
This project implements an image colorization system based on the paper image-to-image translation with conditional adversarial networks, known as the pix2pix model. A UNet with DCGAN model is implemented with UNet as generator and a deep convolutional conditional discriminator to achieve realistic photogrgraph colorization.

## Dataset
The project uses Mirflickr-25000 image dataset, which contains 25000 photographs retrieved from the social photography site Flickr through its public API. Each image in the dataset was classified to an image category, such as people, street, nature, animal etc. The dataset is available here: https://press.liacs.nl/mirflickr/mirdownload.html

## Showcase

![](https://drive.google.com/uc?export=view&id=1uYzBZU_W2-Z4o2U9qC70eR1rFVSxCzAQ)

![](https://drive.google.com/uc?export=view&id=1REyKCB97D7uYAlft1Txr5xmU8ubP0OZD)

## Run Instructions

The following instructions are based on Linux and MacOS system. 

### Requirements

* `Python 3.7`
* `Pillow`
* `scikit-learn`
* `scikit-image`
* `torch`
* `torchvision`
* `torchmetrics`
* `fastai`
* `numpy`
* `matplotlib`

### Repository structure

* `dataset.py`: opens and preprocesses image dataset for training, converts images to Lab colorspace. 
* `model.py`: Builds UNet generator and conditional discriminator model.
* `pretrain.py`: Pretrains UNet generator through supervised learning.
* `train.py`: Train UNet DCGAN.
* `ImageColorization.ipynb`: Evaluates the model and showcases results.
* `saved_weights`: Saved model weights.

### Running

1. Download the image dataset, either from Mirflickr-25000 with link provided above or any other image dataset you like.
2. Specify your dataset folder in `dataset.py`. Your dataset folder should contain all `.jpg` images.
3. Pretrain UNet generator by running `pretrain.py`, change any hyperparameter based on your preferences. 
4. Train UNet DCGAN by running `train.py`, change any hyperparameter based on your preferences. 
