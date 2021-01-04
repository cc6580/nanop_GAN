import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.utils import save_image
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio
import torch.nn as nn
import torch.nn.functional as F
import torch



# Functions for optimizing the gradient
def grad_loop(im,x,generator,optimizer,scheduler=None,clean=None,denoise=False,n_epochs=10000,epsilon = 1):
    '''
    grad_loop finds the nearest image to im in the range of generator, 
    returns the loss curve and the generated image. 
    '''
    loss_list = []
    psnr_list = []
    if denoise:
        psnr_clean_list = []
    generator.eval()
    for epoch in range(n_epochs):

        optimizer.zero_grad()
        
        im_g = (generator(torch.clamp(x,-5,5))*0.5)+0.5
        loss = F.mse_loss(im_g,im,reduction = 'sum') / (im.size(2)*im.size(3))
        loss.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if epoch % 100 == 0:
            loss_list.append(loss.item())
            # Calculate PSNR as well
            psnr = peak_signal_noise_ratio(im.detach().to('cpu').numpy(),im_g.detach().to('cpu').numpy())
            psnr_list.append(psnr)
            if denoise:
                psnr_clean = peak_signal_noise_ratio(clean.detach().to('cpu').numpy(),im_g.detach().to('cpu').numpy(),data_range=2)
                psnr_clean_list.append(psnr_clean)
            
            x_norm = torch.norm(x)
            print("Epoch {}\t Loss: {:2.4f}\t X Norm: {:2.4f}".format(epoch,loss,x_norm))
        if loss < epsilon:
            break
    if denoise:
        return loss_list, psnr_list, psnr_clean_list, im_g, x
    else:
        return loss_list, psnr_list, im_g, x


def convergence_loop(op_list,lr_list,generator,x_keep,im,n_epochs=10000,epsilon=1):
    '''
    convergence_loop compares learning rates and optimizers for on the grad_loop function.
    '''
    # Optimizer comparison loop
    loss_dict = {}
    psnr_dict = {}
    im_dict = {}
    for op in op_list:
        for lr in lr_list:
            x = x_keep.clone().requires_grad_()
            optimizer = op([x],lr=lr)
            dict_key = str(op)[-6:]+'_'+str(lr)
            loss_list,psnr_list,im_out,_ = grad_loop(im,x,generator,optimizer,n_epochs=n_epochs,epsilon=epsilon)
            loss_dict[dict_key] = loss_list
            psnr_dict[dict_key] = psnr_list
            im_dict[dict_key] = im_out
            print("Opt: {}\tLR: {} Done".format(op,lr))
    plt.figure(figsize=[12,12])
    for k in loss_dict.keys():
        plt.plot(loss_dict[k],label=str(k))
    plt.legend(loc='upper right',fontsize=15)
    return loss_dict,psnr_dict,im_dict

def sample_and_optimize(im,generator,z=100,op=torch.optim.Adam,
                        lr=1e-2,num_samples=100,clean=None,denoise=False,num_early_epochs=1000,num_total_epochs=50000):
    '''For a single latent dimension, z, loop thru 
    i) trying num_samples samples for num_early_epochs epochs
    and ii) training the best for num_total_epochs epochs.
    Returns best image, best latent vector, loss_list for the chosen sample, big_loss_list for all the samples,
    big_x_list for all the latent vectors trained in the first phase, and the index of the chosen sample.
    '''

    # Accumulators
    big_loss_list = []
    big_psnr_list = []
    big_x_list = []
    if denoise:
        big_psnr_clean_list = []
    for i in range(num_samples):
        # Initialize x, the initial latent vector
        x_keep = Variable(torch.randn(1,z,1,1,device=device),requires_grad=False)

        generator_list = [generator]
        x_keep_list = [x_keep]
        im_list = [im]
        if denoise:
            loss_dict, psnr_dict, psnr_clean_dict, im_dict, x_dict = latent_dim_loop(generator_list,
                                                    x_keep_list,
                                                    im_list,
                                                    op,
                                                    lr,
                                                    schedule = False,clean=clean, denoise=True,
                                                    n_epochs=num_early_epochs)
        else:
            loss_dict, psnr_dict, im_dict, x_dict = latent_dim_loop(generator_list,
                                                    x_keep_list,
                                                    im_list,
                                                    op,
                                                    lr,
                                                    schedule = False,
                                                    n_epochs=num_early_epochs)
            big_psnr_clean_list.append(psnr_clean_dict)

        big_loss_list.append(loss_dict)
        big_psnr_list.append(psnr_dict)
        big_x_list.append(x_dict)
            
    dict_key = "z="+str(x_keep.shape[1])
    min_losses = []
    for i,loss in enumerate(big_loss_list):
        min_losses.append(np.min(loss[dict_key]))

    # Find the best of the samples, optimizer further
    idx = np.argmin(loss100)
    x_keep = big_x_list[idx]['z=100']
    # x = x_keep.clone().requires_grad_()
    optimizer = torch.optim.Adam([x_keep],lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,20000,40000], gamma=0.6)
    loss_list, psnr_list, im_g, x_out = grad_loop(im,x_keep,generator,optimizer,
                                       scheduler=scheduler, n_epochs=num_total_epochs)
    if denoise:
        return im_g, x_out, loss_list, psnr_list, big_x_list, big_loss_list, big_psnr_list, big_psnr_clean_list, idx
    else:
        return im_g, x_out, loss_list, psnr_list, big_x_list, big_loss_list, big_psnr_list, idx


def latent_dim_loop(generator_list,x_keep_list,im_list,op,lr,schedule=False,clean=None,denoise=False,n_epochs=10000):
    '''
    latent_dim_loop compares convergence rates for different latent dimensions. Only tested on microscope images.
    '''
    loss_dict = {}
    psnr_dict = {}
    im_dict = {}
    x_dict = {}
    if denoise:
        psnr_clean_dict = {}
    for i, generator in enumerate(generator_list):     
        x = x_keep_list[i].clone().requires_grad_()
        # Optimizer only considers the first vector
        optimizer = op([x],lr=lr)
        if schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,20000,40000], gamma=0.6) 
        else:
            scheduler=None
        dict_key = "z="+str(x.shape[1])
        
        if denoise:
            loss_list, psnr_list, psnr_clean_list, im_g,_ = grad_loop(im_list[i],x,generator,optimizer,
                                                                      scheduler, clean=clean, denoise=True,
                                                                      n_epochs=n_epochs,epsilon=1e-5)
            psnr_clean_dict[dict_key] = psnr_clean_list
        else:
            loss_list, psnr_list, im_g,_ = grad_loop(im_list[i],x,generator,optimizer,scheduler,n_epochs=n_epochs,epsilon=1e-5)
        loss_dict[dict_key] = loss_list
        psnr_dict[dict_key] = psnr_list
        im_dict[dict_key] = im_g
        x_dict[dict_key] = x
        print("Opt: {}\tLR: {} Done".format(op,lr))
    plt.figure(figsize=[12,12])
    for k in loss_dict.keys():
        plt.plot(loss_dict[k],label=str(k))
    plt.legend(loc='upper right',fontsize=15)
    if denoise:
        return loss_dict, psnr_dict, psnr_clean_dict, im_dict, x_dict
    else:
        return loss_dict, psnr_dict, im_dict, x_dict


####################
# PLOTTING FUNCTIONS
####################

def plot_psnr_multi(first,second,z,title,start_idx=0):    
    '''Given a loss_dict where the optimizer name and LR are the key, plots the learning curves'''
#     start_idx = round(start_idx,-2)
    plt.figure(figsize=[15,9])
    key = 'z='+str(z)
    for curve in first:        
        plt.plot(np.arange(0,100*len(curve[key]),100),curve[key])
    first_epoch_ct = 100*len(curve[key])
    plt.plot(np.arange(first_epoch_ct-100,first_epoch_ct-100+100*len(second),100),second)
    plt.vlines(1400,plt.gca().get_ylim()[0],plt.gca().get_ylim()[1],colors='gray',linestyle='dashed')
    plt.ylabel("PSNR",fontsize=12)
    plt.xlabel("Epoch",fontsize=12)
    plt.title(title,fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    print(plt.gca().get_ylim())

    
def plot_loss_curve_list(loss_list,title,start_idx=0):    
    '''Given a loss_dict where the optimizer name and LR are the key, plots the learning curves'''
    start_idx = round(start_idx,-2)
    plt.figure(figsize=[15,9])
    for k in range(len(loss_list)):
        plt.plot(np.arange(start_idx,100*len(loss_list[k]),100),loss_list[k][int(start_idx/100):])
    plt.ylabel("MSE Loss per pixel",fontsize=12)
    plt.xlabel("Epoch",fontsize=12)
    plt.title(title,fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    

def plot_loss_curve(loss_dict,title,keys=None,start_idx = 0):    
    '''Given a loss_dict where the optimizer name and LR are the key, plots the learning curves'''
    start_idx = round(start_idx,-2)
    if keys == None:
        keys = loss_dict.keys()
    plt.figure(figsize=[15,9])
    for k in keys:
        plt.plot(np.arange(start_idx,100*len(loss_dict[k]),100),loss_dict[k][int(start_idx/100):],label=str(k))
    plt.legend(loc='upper right',fontsize=15)
    plt.ylabel("MSE Loss per pixel",fontsize=12)
    plt.xlabel("Epoch",fontsize=12)
    plt.title(title,fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


def plot_img(im,grayscale=True,vmin=None,vmax=None):
    '''Plots image in gray or color. It doesn't consider vmin and vmax though.'''
    if im.min() < 0:
        im = im*0.5 + 0.5
    if grayscale:
        if vmin:
            plt.imshow(im.detach().to('cpu')[0,0],cmap='gray',vmin=vmin,vmax=vmax)
        else:
            plt.imshow(im.detach().to('cpu')[0,0],cmap='gray')
    else:
        if vmin:
            plt.imshow(im[0,:,:,:].detach().to('cpu').permute(1,2,0),vmin=vmin,vmax=vmax)
        else:
            plt.imshow(im[0,:,:,:].detach().to('cpu').permute(1,2,0))