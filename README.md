# nanop_GAN

This repository contains the code and report for the project "Image Denoising with Generative Adversarial Networks"

## Research Objectives
Traditional denoising methods use mean-squared error as a loss function, but due to its averaging nature, the resulting images often have problems such as blurriness or exhibiting phantom artifacts. To overcome these issues, we propose a generative adversarial network (GAN) based denoising method in which we first train a GAN to generate realistic microscope images, then optimize over the latent space to find a clean image that most resembles the noisy image we were given. Since well-trained GANs could only generate reasonable images, we hypothesize that our method can help avoid the problems brought by traditional denoising methods.
  1. train a generator model that can generate realistic microscope images from a latent vector of length 50, 100, and 150 respectively
  2. evaluate the generator output
  3. choose the best generator model, and optimize to denoise a simulated image from outside the training set
  4. using the best generator model, optimize to denoise a real microscope image

## Dataset
Three thousand simulated grey-scale electronic microscope images of shape 876 × 927 were used to train the Generative Adversarial Network. Since the Generator is designed to output 512 × 512 images, random 512 × 512 patches were cut from the input images. As opposed to resizing the images which might cause loss of information and skew the atomic structure of the atoms, cutting random patches avoids these issues and increase the robustness of the model. 

The real microscope data comes in video format with 40 frames of images with size 1215 × 1208 pixels. The images used for denoising herein are randomly cropped 512 × 512 pixel images of stills from that video with no additional scaling.

*Feel free to contact us if you want to access the dataset.*

## Codes
GAN:
* [nanop_GAN.ipynb](https://github.com/cc6580/nanop_GAN/blob/main/GAN_codes/nanop_GAN.ipynb): step-by-step walk through of loading and transforming input data, GAN architecture design, optimizer setup, and different training designs with various training schemes. 
* [nanop_gan.py](https://github.com/cc6580/nanop_GAN/blob/main/GAN_codes/nanop_gan.py): final python script with the most successful trianing scheme
* [loaded_GAN.ipynb](https://github.com/cc6580/nanop_GAN/blob/main/GAN_codes/loaded_GANS.ipynb): notebook that loads saved models and generates images based on random vectors

Optimizer:




