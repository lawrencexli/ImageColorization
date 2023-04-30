# Mirflickr-25k Image Colorization

By Lawrence Li (ll3598@columbia.edu) and Isabella Cao (sc3912@columbia.edu) for COMS 4995 applied computer vision project

Spring 2023, Columbia University

## Cloning this repository
All trained model weights are stored in the `saved_weights` directory through github large file storage (Git LFS). When cloning this repository, you need to use the `git lfs clone` command to have a complete clone of this repository. Otherwise, you may need to manually download the weights from our github repository source. More details about Git LFS and how to install can be found at https://git-lfs.com. 

## About the project
This project implements an image colorization system based on the paper image-to-image translation with conditional adversarial networks, known as the pix2pix model. A UNet with DCGAN model is implemented with UNet as generator and a deep convolutional conditional discriminator to achieve realistic photogrgraph colorization.

## Dataset

**This repository does not include the image dataset. You need to manually download it from the provided link below. **

The project uses Mirflickr-25000 image dataset, which contains 25000 photographs retrieved from the social photography site Flickr through its public API. Each image in the dataset was classified to an image category, such as people, street, nature, animal etc. The dataset can be downloaded here: https://press.liacs.nl/mirflickr/mirdownload.html.

## Showcase

![](https://drive.google.com/uc?export=view&id=1uYzBZU_W2-Z4o2U9qC70eR1rFVSxCzAQ)

![](https://drive.google.com/uc?export=view&id=1REyKCB97D7uYAlft1Txr5xmU8ubP0OZD)

![](https://drive.google.com/uc?export=view&id=1SxaGvstDsz1Vx4wxfKFgryahAKpMJiV9)

![](https://drive.google.com/uc?export=view&id=1FXw75FJTUEhWynxzYfCb3Pu4qzDMTTnt)

## Run Instructions

The following instructions are based on Linux and MacOS system. 

### Requirements

The following are required dependencies for this project.

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
* `saved_weights/`: Saved model weights. All trained models are stored here. 
* `custom_image/`: Custom image dataset containing jpg images. Minimum number of images is 2. 
* `evaluation_plots/`: Stores plots for different evaluation metrics during training.
* `PlotResult.ipynb`: Plot the evaluation metrics, including all generator and discriminator losses during training, and accuracy metrics such as SSIM and PSNR during evaluation.

### Running

1. Download the image dataset, either from Mirflickr-25000 with link provided above or any other image dataset you like. In our project we downloaded Mirflickr-25000 and extract the zip as `mirflickr25k` folder. We used the command `wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip` and `unzip mirflickr25k.zip` to retrieve the data.
2. Specify your dataset folder in `dataset.py`. Your dataset folder should contain all `.jpg` images. No subfolders!
3. Pretrain UNet generator by running `pretrain.py`, change any hyperparameter based on your preferences. 
4. You should see that a pretrained weight file is generated under the `saved_weights/` folder.
5. Train UNet DCGAN by running `train.py` using the pretrained weight file that is generated in `saved_weights/`, change any hyperparameter based on your preferences. 
6. You should see that a final weight file is generated alongside with a `.pkl` file storing all evaluation metrics tracking in `saved_weights/`. 
7. Use `PlotResult.ipynb` to plot the evaluation metrics over epochs and them under `evaluation_plots/`. Loads the performance tracker dictionary pickle file from `saved_weights/`.
8. Use `ImageColorization.ipynb` to visualize colored samples from your dataset and perform evaluation. Loads the generator UNet model with trained weights from `saved_weights/` and generate `result.png` file for colorization result. 

### Attributes

Part of the codes are modified and revised from https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8 by Moein Shariatnia and the pytorch implementation of deep concolutional discriminator available at google cloud tutorials. 

