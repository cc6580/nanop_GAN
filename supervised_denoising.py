import argparse
# import os
# import sys
import pickle
import numpy as np
import math
import gc
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from torch.distributions import Poisson
# from skimage.metrics import peak_signal_noise_ratio
# from PIL import Image

# Supervised denoiser
# Load the denoiser
from torch.serialization import default_restore_location
import sys
sys.path.append('nanoparticle_denoising_master/nanoparticle_denoising_master/')
import nano_data as data
# import models as models
import nano_models as models
import utils

# This .py file denoises using the supervised denoising method and pre-trained model
# provided by Sreyas Mohan.


# Model args
class Args(object):
    def __init__(self):
        self.data_path = 'data/microscope/10/'
        self.dataset = 'ptceo2'
        self.batch_size = 1
        self.checkpoint_path = 'nanoparticle_denoising_master/nanoparticle_denoising_master/pretrained/checkpoint_best.pt'
        self.output_path = 'microscope_output/supervised/'
        self.no_cuda = True
args=Args()

# Load the model
device = torch.device("cuda") if (
    torch.cuda.is_available() and not args.no_cuda) else torch.device("cpu")
# Load arguments from previous checkpoint
state_dict = torch.load(
    args.checkpoint_path,
    map_location=lambda s,
    l: default_restore_location(
        s,
        "cpu"))
args = argparse.Namespace(
    **{**vars(args), **vars(state_dict["args"]), "no_log": True})

# Build data loader and load model

super_model = models.build_model(args).to(device)
super_model.load_state_dict(state_dict["model"][0])
super_model.eval()

# # To loop through a series of images
save_dir = 'microscope_output/comparison_test/'
filename = "denoise_images.dat"
with open(save_dir+filename, "rb") as f:
    im_list = pickle.load(f)

# d_im_list = []

# Reshape transforms
topil = transforms.ToPILImage()
resize = transforms.Resize(358)
totensor = transforms.ToTensor()
resize_big = transforms.Resize(512)

for i,im in enumerate(im_list):
    outfile = "sup_denoised_"+str(i)+"_70.pt"
    im_r = totensor(resize(topil(im[0,0].cpu()))).unsqueeze(0)
    print("im_r shape: ",im_r.shape)
    denoise_im = super_model(im_r)
    denoise_im.detach().cpu()
    im_out = totensor(resize_big(topil(denoise_im[0,0]))).unsqueeze(0)
    torch.save(im_out,save_dir+outfile)
    del denoise_im, im_r, im_out
    gc.collect()





# png_file = 'im_1_70_sup.png'
# im = torch.load(save_dir+'im_noise_1_70.pt')
# outfile = 'sup_denoised_real_1_70.pt'

# denoise_im = super_model(im.cpu())
# denoise_im.detach().cpu()
# torch.save(denoise_im,save_dir+outfile)
# plt.imshow(denoise_im[0,0].detach(),cmap='gray',vmin=0,vmax=1)
# plt.savefig(save_dir+png_file)