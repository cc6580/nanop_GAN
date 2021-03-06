# imports-----------------------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import random
import glob
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
# Define Agg as Backend for matplotlib when no X server is running
mpl.use('Agg') # AGG backend is for writing to file, not for rendering in a window.
# if you load this in an interactive notebook, no figures would display

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#on HPC, we can check whether we have GPU support here---------------------------------------------------------------------------------------------
assert torch.cuda.is_available()
print(60*'-')
print('|','Found GPU at: {}'.format(torch.cuda.get_device_name(0)), '|')
print(60*'-')
torch.cuda.empty_cache()

# define inputs----------------------------------------------------------------------------------------------------------------------------------
# Root directory for dataset that has sub-image folder within
dataroot = "../../particle_data/images"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 36
  # using 64 results in a out-of-memory error for CUDA

# Spatial size of training images. All images will be cropped to this size using a transformer.
image_size = 512

# Number of channels in the training images. For color images this is 3, for our grayscale image is 1
nc = 1

# Size/length of z latent vector (i.e. size of generator input)
# nz = 100
nz = int(sys.argv[4])

# Size of feature maps in generator, i.e. # of channels of the kernel
ngf = 64

# Size of feature maps in discriminator, i.e. # of channels of the kernel
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0001
  # smaller batch number, slightly larger learning rate?

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1# Decide which device we want to run on

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("device:", device)

try:
  os.mkdir('result_figs')
  # for storing saved figures and plots
except:
  print('directory already exists')

# load data ----------------------------------------------------------------------------------------------------------------------------------
folder_data=[]
  # this is a list of all the file path that ends with .tif

start_folder = int(sys.argv[1])
end_folder = int(sys.argv[2]) + 1
  # note that all sys.argvs are read as strings
  # so you must convert them to the right type before using

print("extracting images from folder {} to folder {}".format(start_folder, end_folder-1))

for i in range(start_folder, end_folder):
  folder_data.extend(glob.glob(dataroot+'/'+str(i)+'/*.tif'))
  # add all image file-paths from different image folders to the list of image file paths

len_data = len(folder_data)
print("total of {} images in the training set".format(len_data))

class CustomDataset_crop(Dataset):
  def __init__(self, image_paths, train=True):   # initial logic happens like transform
    self.image_paths = image_paths
      # image_paths is a list where each element is a file path to 1 image
    # self.target_paths = target_paths
    self.transforms = transforms.Compose([
                                          transforms.RandomCrop(512),
                                            # randomly crops a 512x512 image from the input image
                                          transforms.ToTensor(),
                                          transforms.Normalize(0.5, 0.5) # normalize to mean=0.5 and std=0.5
                                            # this converts all values form 0~1 to -1~1 so scale you vmin and vmax accordingly
                                            # https://discuss.pytorch.org/t/how-should-i-convert-tensor-image-range-1-1-to-0-1/26792
                      ])

  def __getitem__(self, index):
    image = Image.open(self.image_paths[index])
    # mask = Image.open(self.target_paths[index])
    t_image = self.transforms(image)
    # print(self.image_paths[index])
    return t_image, 0
      # since we don't have a corresponding `y` here for each sample image, just fill in 0 for `y`

  def __len__(self):  # return count of sample we have
    return len(self.image_paths)

# create the dataset from all images
dataset = CustomDataset_crop(folder_data, train=True)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

cont = bool(int(sys.argv[3]))

if cont == 0:
  real_batch = next(iter(dataloader))
    # get 1 batch of samples, which is 36 samples per our defined batch_size

  # Plot some training images
  plt.figure(figsize=(8,8))
  plt.axis("off")
  plt.title("Training Images")
  grid = vutils.make_grid((real_batch[0][:16]), 
                          # [(real_batch[0].to(device)[i]) for i in range(16)], 
                          nrow=4,
                          padding=5, 
                          normalize=True)
  plt.imshow(grid[1], cmap='gray', vmin=0, vmax=1)
    # since our grid normalized all images, scale is from 0~1
  plt.savefig("result_figs/training_images")
  plt.close()
  print("generated training images")

# weight initialization ---------------------------------------------------------------------------------------------------------------------------

# custom weights initialization called on netG and netD
def weights_init(m):
    # m will be a model module that inherits the nn.Module class, so it will have the following attributes
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: 
        # if the class name contains the partial string 'Conv'
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # if the class name contains the partial string 'BatchNorm'
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Archi ---------------------------------------------------------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is latent vector Z of shape (nz x 1 x 1), going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
              # state size. (ngf*16) x 4 x 4, where 4 = n+k-1= 1+4-1
              # to figure out the dimension of convtrans, just ask what dimension with the current setting gets you the current dimension
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
              # state size. (ngf*8) x 8 x 8, where 8 = (n-1)*s+k-2p = 3*2+4-2
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
              # state size. (ngf*4) x 16 x 16, where 16 = (n-1)*s+k-2p = 7*2+4-2
            nn.ConvTranspose2d( ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
              # state size. (ngf*2) x 32 x 32, where 32 = (n-1)*s+k-2p = 15*2+4-2
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
              # state size. (ngf) x 64 x 64, where 64 = (n-1)*s+k-2p = 31*2+4-2
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
              # state size. (ngf) x 128 x 128, where 128 = (n-1)*s+k-2p = 63*2+4-2
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
              # state size. (ngf) x 256 x 256, where 256 = (n-1)*s+k-2p = 127*2+4-2
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
              # state size. (nc) x 512 x 512, where 512 = (n-1)*s+k-2p = 255*2+4-2
            nn.Tanh()
            
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Archi ---------------------------------------------------------------------------------------------------------------------------   
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is an real or generated image of shape (nc) x 512 x 512
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf) x 256 x 256, where 256 = (n+2p-k)/s+1 = (512+2-4)/2+1 
            nn.Conv2d(ndf, ndf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf) x 128 x 128, where 128 = (n+2p-k)/s+1 = (256+2-4)/2+1 
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf) x 64 x 64, where 64 = (n+2p-k)/s+1 = (128+2-4)/2+1 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*2) x 32 x 32, where 32 = (n+2p-k)/s+1 = (64+2-4)/2+1
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*4) x 16 x 16, where 16 = (n+2p-k)/s+1 = (32+2-4)/2+1
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*8) x 8 x 8, where 8 = (n+2p-k)/s+1 = (16+2-4)/2+1  
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
              # state size. (ndf*16) x 4 x 4, where 4 = (n+2p-k)/s+1 = (8+2-4)/2+1  
            nn.Conv2d(ndf * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
              # state size. 1 x 1 x 1, where 1 = (n+2p-k)/s+1 = (4+0-4)/1+1 
            # nn.Sigmoid()
              # removed since we are using BCEWithLogitsLoss
        )

    def forward(self, input):
        return self.main(input)

# initialize model and optimizer ---------------------------------------------------------------------------------------------------------------------

if cont == 0:
  print("starting fresh training...")
  # Create the generator
  netG = Generator(ngpu).to(device) # moves the model to the GPU
  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
  netG.apply(weights_init)
else:
  print("continue training from saved models...")
  path = "model_checkpoints/e"+str(int(sys.argv[5])-1)+"_G"
    # for ex, "model_checkpoints/e400_G"
  netG = torch.load(path).to(device) # moves the model to the GPU
  # the correct generator class (compatible with the model saved) must be defined in order for the saved model to be loaded

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
      # Implements data parallelism at the module level.

# Print the model structure
print(netG)

if cont == 0:
  # Create the Discriminator
  netD = Discriminator(ngpu).to(device)
  # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
  netD.apply(weights_init)
else:
  path = "model_checkpoints/e"+str(int(sys.argv[5])-1)+"_D"
    # for ex, "model_checkpoints/e400_D"
  netD = torch.load(path).to(device)# moves the model to the GPU
  # the correct discriminator class (compatible with the model saved) must be defined in order for the saved model to be loaded

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCEWithLogitsLoss()
  # This loss combines a Sigmoid layer and the BCELoss in one single class. 
  # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss 
  # as, by combining the operations into one layer,we take advantage of the log-sum-exp trick for numerical stability.
  # learn more here https://github.com/soumith/ganhacks/issues/36
# So we also have to go back to our discriminator to remove the last sigmoid layer

# Create a batch of 64 different latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 0.9
  # label smoothing, from pt7 in https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
real_label_G = 1.0
  # this is the real label we will use for G when trying to fool the D
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=float(sys.argv[6]), betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=float(sys.argv[7]), betas=(beta1, 0.999))
  # Two Time-Scale Update Rule
  # a higher learning rate for the discriminator and a lower one for the generator
  # from pt9 in https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628

# start training -----------------------------------------------------------------------------------------------------------------------------

try:
  os.mkdir('model_checkpoints')
  # for storing saved models
except:
  print('directory already exists')

print("we have {} samples and {} samples per batch, we will have {} batches overall".format(
    len(folder_data), batch_size, len(dataloader)))
print("there are {} samples in our fixed test set".format(len(fixed_noise)))

# Training Loop

# Lists to keep track of progress
if cont == 0:
  logs = dict()
  G_losses = []
  D_losses = []
  start_epoch = 1
  img_list = []
  last_imgs = []
else:
  with open(str(sys.argv[8]), 'r') as fp:
    logs = json.load(fp)
  G_losses = logs["G_losses"]
  D_losses = logs["D_losses"]
  # start_epoch = logs["start_epoch"]
  start_epoch = int(sys.argv[5])
  # img_list = torch.load("img_list.ts")
    # this takes too much RAM, we will load this in later if needed
  img_list = []
  last_imgs = torch.load("last_imgs.ts")

iters = 1
checker = 0
num_epochs=num_epochs

print("Starting Training Loop From Epoch {}".format(start_epoch))
# For each epoch
for epoch in range(start_epoch, start_epoch+num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # i is the batch number
        # data[0] is the list of samples containing batch_size many samples
        # data[1] is the list of corresponding labels but in this case is meaningless, we will create our own labels

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch----------------------------------------------------------------------------
        
        netD.zero_grad() 
          # zero the gradients of model netD, must be done at the start of each batch
        
        # Format batch:
        real_input = data[0].to(device);
          # real image inputs as input to the model 
        
        # if epoch == 1 and i == 0: print(real_input.size())
          
        b_size = real_input.size(0) 
          # number of samples in the current batch
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) 
          # create a tensor of real labels all as 1 same length as the number of samples in the current batch
          # we can't directly use batch_size here because in the last batch there might be less samples
        
        # Forward pass real batch through D:
        output = netD(real_input).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label) 
          # computes the loss itself
          # when updating the discriminator, we want to minimize the difference between the D output and 1 for real samples
        
        # Calculate gradients for D in backward pass
        errD_real.backward() 
          # computes the gradient of the loss w.r.t. all model parameters
          # this step also store the gradients in `parameter.grad` attribute for every parameter in netD
        D_x = output.mean().item()
          # compute the mean of the probability predictions for this batch of real images for checking

        ## Train with all-fake batch----------------------------------------------------------------------------
        
        # Generate batch of random latent vectors z, this is randomly generated for every new batch and not fixed
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        
        # Generate a fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label) # create a tensor of fake labels all as 0
        
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
          # computes the loss itself
          # when updating the discriminator, we want to minimize the difference between the D output and 0 for fake samples
        
        # Calculate the gradients for this batch
        errD_fake.backward()
          # computes the gradient of the loss w.r.t. all model parameters
          # this step also store the gradients in `parameter.grad` attribute for every parameter in netD
          # since torch accumulates gradients, and you didn't zero out your gradient after the update above
          # this simply addes the gradient values to the previous gradient values
          # since we want to minimize both fake and real error (w.r.t. to the different label tensors), their sum will also be a goal of us to minimize
        D_G_z1 = output.mean().item()
          # compute the mean of the probability predictions for this batch for checking
        
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
          # this is only for our viewing sake
          # when we called .backward() on both losses the, their gradients were automatically added in the model already
        
        # Update D
        optimizerD.step()
          # update each parameter value in netD according to the gradient values stored for each parameter

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG.zero_grad()
          # zero the gradients of model netG, must be done at the start of each batch
        label.fill_(real_label_G) 
          # fake images are considered real for the generator cost 
          # because we want the D's output to be close to the classifying them as real
          # remember, the loss is just a measure of how different our prediction is from our result
          # and we update our model to minimize that difference
          # when updating the generator model, we want to minimize the difference between the D output and 1
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        #Calculate G's loss based on this output
        errG = criterion(output, label)
        
        # Calculate gradients for G
        errG.backward()
          # computes the gradient of the loss w.r.t. all model parameters
          # this step also store the gradients in `parameter.grad` attribute for every parameter in netG
        D_G_z2 = output.mean().item()
          # compute the mean of the probability predictions for this batch for checking
        
        # Update G
        optimizerG.step()
          # update each parameter value in netD according to the gradient values stored for each parameter

        # Output training stats
        if i % 20 == 0 or (i+1) % len(dataloader) == 0: # every 20 batches and the last batch in every epoch
            print('epoch[%d/%d] batch[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, start_epoch+num_epochs-1, i+1, len(dataloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                  )
              # the .item() extract the actual numeric result out of the tensor(~, device='cuda:0') output
            if errD.item() == 0 or errG.item() == 0: # if model enters training failure mode
                checker+=1
            else: # reset checker
                checker=0

            if checker > 50:
                print("training stuck for more than 50 checks, stopping it early")
                break;

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        # for every 100 iterations, and the last batch on the last epoch
        # predict on fixed_noise and generate validation images to get an idea of how well the model is going
        if (iters % 100 == 0) or ((epoch == start_epoch+num_epochs-1) and (i == len(dataloader)-1)):
            print("validating on fixed noise: iteraton %d, epoch %d, batch number %d"% (iters,epoch, i+1))
            
            with torch.no_grad():
            # stops the autograd engine from calculating the gradients, saves time and space
            # which is the recommended way of doing validation
                fake = netG(fixed_noise).detach().cpu()
                  # send it to cpu here because we don't need anything from the gpu to deal with these results
            
            # print(torch.equal(vutils.make_grid(fake, padding=2, normalize=True)[0], vutils.make_grid(fake, padding=2, normalize=True)[1]))
            #   # just re-confirms that all channels of the grid image is the same b/c we are greyscale
            img_list.append(vutils.make_grid(fake, nrow=6, padding=5, normalize=True)[0])
              # make a grid out of the 64 generated images into 1 entire image and add it to list

        iters += 1
    # also save models as checkpoints after each epoch incase training gets interrupted 
    print("model saved on iteration {}, epoch {}, batch {} with G_loss ={}".format(iters-1 ,epoch, i+1, errG.item()))
    torch.save(netG, 'model_checkpoints/e{}_G'.format(epoch))
    torch.save(netD, 'model_checkpoints/e{}_D'.format(epoch))
    print(120*'-')

if cont == 0: last_imgs.append(img_list[0].cpu())
last_imgs.append(img_list[-1].cpu())

# save the losses and the validation images so we can re-visit them later
# torch.save(img_list, "img_list.ts")
torch.save(img_list, "img_list_e{}~{}.ts".format(start_epoch, start_epoch+num_epochs-1))
torch.save(last_imgs, "last_imgs.ts")
  # since these lists contains tensors, we cannot save with Json
logs["G_losses"] = G_losses
logs["D_losses"] = D_losses
# logs["start_epoch"] = epoch+1
with open(str(sys.argv[8]), 'w') as fp:
    json.dump(logs, fp)

print("number of grids saved in G's progression: ", len(img_list))

if cont == 0:
  plt.figure(figsize=(8,8))
  plt.imshow((img_list[0].cpu()), cmap='gray', vmin=0, vmax=1)
    # grid was normalized from make_grid so value is 0~1
    # since the images were output of the model on GPU, we have to convert them to host CPU memory for plt to work
  plt.savefig("result_figs/first_gen")
  plt.close()
  print("plotted the first generated image")

plt.figure(figsize=(8,8))
plt.imshow((img_list[-1].cpu()), cmap='gray', vmin=0, vmax=1)
  # grid was normalized so value is 0~1
plt.savefig("result_figs/last_gen")
plt.close()
print("plotted the last generated image")

plt.figure(figsize=(8,int(8*len(last_imgs))))
  # width 8, height 8*number of images
for i in range(0, len(last_imgs)):
  plt.subplot(len(last_imgs), 1, i+1)
    # add len(last_imgs) rows and 1 column of subplots, this will be the i-th subplot of them
    # the first subplot is indexed 1, not 0!
  plt.axis("off")
  plt.title("image generated after epoch {}".format(i*100))
  plt.imshow(last_imgs[i], cmap='gray', vmin=0, vmax=1)
plt.savefig("result_figs/last_imgs")
plt.close()
print("plotted the last imgs generated after every 100 epochs")

# results -----------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(int(10*len(last_imgs)),5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.yscale('log')
plt.legend()
plt.savefig("result_figs/GD_loss_plot")
plt.close()
print("plotted the model losses")

# inspect real vs fake -----------------------------------------------------------------------------------------------------------------------------
# Grab a batch (any batch) of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
  # add 1 row and 2 columns of subplots, this will be the first of them
plt.axis("off")
plt.title("Real Images")
plt.imshow((vutils.make_grid(real_batch[0][:batch_size], nrow=6, padding=5, normalize=True))[0], cmap='gray', vmin=0, vmax=1)

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
  # add 1 row and 2 columns of subplots, this will be the second of them
plt.axis("off")
plt.title("Fake Images")
plt.imshow(img_list[-1], cmap='gray', vmin=0, vmax=1)
plt.savefig("result_figs/real_vs_fake")
plt.close()
print("plotted real vs fake image")

# G's progression mp4 -----------------------------------------------------------------------------------------------------------------------------
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
  # `conda install -c conda-forge ffmpeg` on HPC in your environment for this to work
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow((i.cpu()), animated=True, cmap='gray')] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save("result_figs/G_progress_e{}~{}.mp4".format(start_epoch, start_epoch+num_epochs-1), writer=writer)
  # save the animation to a mp4 video file
plt.close()
print("plotted generator progression")