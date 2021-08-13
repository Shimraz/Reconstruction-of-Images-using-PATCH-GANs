#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.utils as vutils
import torch.nn.functional as nnf
torch.manual_seed(0)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.autograd import Variable

from torchvision.utils import save_image
from PIL import Image

from skimage.transform import rescale, resize
from skimage import io
from skimage.measure import compare_psnr

from models import *
from utils import *
from skimage import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def motion_blur(image,kernel_size = 3):
    channels = 3
    motion_filter = np.zeros((kernel_size, kernel_size), np.float32)
    motion_filter[int((kernel_size - 1)/2), :] = np.ones(kernel_size)  # horizontal blurr
    #motion_filter[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)  # vertical blurr
    motion_filter /= kernel_size
    motion_filter = torch.FloatTensor(motion_filter).unsqueeze(0).unsqueeze(0)
    motion_filter = torch.repeat_interleave(motion_filter, channels, dim=0)
    #print(motion_filter.shape)
    weight = nn.Parameter(data=motion_filter, requires_grad=False).to(device)
    image = nnf.conv2d(image, weight, padding=3, groups=channels).to(device)
    return image


def GPP_Color_solve(test_img='color_tiger',savedir='outs_gpp_color',USE_BM3D=False):
    savedir = 'motion_blurr'
    test_img = 'Gabrielle_blurred_h'

    # I_x = I_y = 512 #size of images (can be varied)
    I_x = 352
    I_y = 480
    d_x = d_y = 32 #size of patches (can be varied)
    n_measure = 0.1 #measurement rate (can be varied)
    nIter = 10001

    if USE_BM3D:
        from bm3d import bm3d, BM3DProfile
        from experiment_funcs import get_experiment_noise
        noise_type = 'g0'
        noise_var = 0.01 # Noise variance
        seed = 0  # seed for pseudorandom noise realization

        # Generate noise with given PSD
        noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, [256,256])
        ###

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dim_x = d_x*d_y
    batch_size = (I_x*I_y)//(dim_x)
    nz = 100  # latent dimensionality of GAN (fixed)

    dim_phi = int(n_measure*dim_x)

    n_img_plot_x = I_x//d_x
    n_img_plot_y = I_y//d_y
    workers = 2
    ngpu = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## measurement operator
    phi_np = np.random.randn(dim_x,dim_phi)
    phi_test = torch.Tensor(phi_np)

    iters = np.array(np.geomspace(10,10,nIter),dtype=int)
    fname = '../GPP/motion_blurr/{}.jpg'.format(test_img)
    image = io.imread(fname)
    x_test = resize(image, (I_x, I_y),anti_aliasing=True,preserve_range=True,mode='reflect')
    print('x_test da shape hove',x_test.shape)
    x_test_ = np.array(x_test)/255.
    print(x_test_.shape)

    x_test = []
    for i in range(n_img_plot_x):
        for j in range(n_img_plot_y):
            _x = x_test_[i*d_x:d_x*(i+1),j*d_y:d_y*(j+1)]
            x_test.append(_x)

    x_test = np.array(x_test)
    test_images = torch.Tensor(np.transpose(x_test[:batch_size,:,:,:],[0,3,1,2]))
    print(test_images.shape)
    io.imsave('{}/gt.png'.format(savedir),(255*x_test_).astype(np.uint8))

    genPATH = './all_models/generator.pt'

    netG = Generator(ngpu=ngpu,nc=3).to(device)
    netG.apply(weights_init)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    if os.path.isfile(genPATH):
        if device.type == 'cuda':
            netG.load_state_dict(torch.load(genPATH))
        elif device.type=='cpu':
            netG.load_state_dict(torch.load(genPATH,map_location=torch.device('cpu')))
        else:
            raise Exception("Unable to load model to specified device")

        print("************ Generator weights restored! **************")
        netG.eval()



    criterion = nn.MSELoss()
    z_prior = torch.zeros(batch_size,nz,1,1,requires_grad=True,device=device)

    optimizerZ = optim.RMSprop([z_prior], lr=5e-3)

    real_cpu = test_images.to(device)
    print(real_cpu.shape)

    for iters in range(nIter):
        optimizerZ.zero_grad()

        z2 = torch.clamp(z_prior,-1.,1.)
        fake = 0.5*netG(z2)+0.5
        fake = motion_blur(fake, 7)  # apply the mask
        cost = 0
        for i in range(3):
            y_gt = real_cpu[:,i,:,:]
            y_est = fake[:,i,:,:]
            cost += criterion(y_gt,y_est)

        cost.backward()
        optimizerZ.step()
        if (iters % 100 == 0):

            with torch.no_grad():
                z2 = torch.clamp(z_prior,-1.,1.)
                fake = 0.5*netG(z2).detach().cpu() + 0.5
                fake = nnf.interpolate(fake, size=(d_x, d_y), mode='bilinear', align_corners=False)
                G_imgs = np.transpose(fake.detach().cpu().numpy(),[0,2,3,1])

            imgest = merge(G_imgs,[n_img_plot_x,n_img_plot_y])
            psnr = compare_psnr(x_test_,imgest,data_range=1.0)

            if USE_BM3D:
                merged_clean = bm3d(imgest,psd)
                psnr1 = compare_psnr(x_test_,merged_clean,data_range=1.0)
                print('Iter: {:d}, Error: {:.3f}, PSNR-raw: {:.3f}, PSNR-bm3d: {:.3f}'.format(iters,cost.item(),psnr,psnr1))
                io.imsave('{}/inv_solution_bm3d_iters_{}.png'.format(savedir,str(iters).zfill(4)),(255*merged_clean).astype(np.uint8))
            else:
                print('Iter: {:d}, Error: {:.3f}, PSNR-raw: {:.3f}'.format(iters,cost.item(),psnr))
                io.imsave('{}/inv_solution_iters_{}.png'.format(savedir,str(iters).zfill(4)),(255*imgest).astype(np.uint8))



if __name__ == '__main__':
    # test_images = ['barbara', 'Parrots','lena256','foreman','cameraman','house','Monarch']
    GPP_Color_solve()




