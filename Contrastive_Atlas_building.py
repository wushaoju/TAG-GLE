
from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
from Diffeo_losses import NCC, MSE, Grad
from Diffeo_networks import DiffeoDense  
from SitkDataSet import SitkDataset as SData
import lagomorph as lg
# from adopt import ADOPT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None


##################Data Loading##########################
def load_and_preprocess_data(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    im_temp = sitk.ReadImage(f'{data_dir}/{data[keyword][0]["Data"]}')
    temp_scan = sitk.GetArrayFromImage(im_temp)
    xDim, yDim, zDim = temp_scan.shape
    return xDim, yDim, zDim, im_temp


def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    net = DiffeoDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32], [ 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    net_rest = DiffeoDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32], [32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net_rest = net_rest.to(dev)

    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()

    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
    elif para.model.optimizer == 'Adopt':
        optimizer = optim.ADOPT(net.parameters(), lr=para.solver.lr) 
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, net_rest, criterion, optimizer

def cosine_similarity_loss_3d(x1, x2):
    # Flatten the spatial dimensions (32x32x32) into 1D vectors for each channel while keeping batch size intact
    x1_flat = x1.view(x1.size(0), x1.size(1), -1)  # Flatten spatial dimensions (32x32x32) into 1D for each channel
    x2_flat = x2.view(x2.size(0), x2.size(1), -1)  # Flatten spatial dimensions (32x32x32) into 1D for each channel

    # Normalize the flattened vectors (L2 normalization across spatial dimensions)
    x1_flat = F.normalize(x1_flat, p=2, dim=-1)  # Normalize over the last dimension (spatial)
    x2_flat = F.normalize(x2_flat, p=2, dim=-1)  # Normalize over the last dimension (spatial)

    # Cosine similarity: dot product of the normalized vectors for each channel
    cos_sim = torch.sum(x1_flat * x2_flat, dim=-1)  # Sum over the flattened spatial dimensions for each channel

    # Contrastive loss: We want positive pairs to have high similarity and negative pairs to have low similarity
    loss = 1 - cos_sim  # This will make the loss smaller for more similar pairs
    
    return loss.mean()
def train_network(trainloader, aveloader, train_restloader, net, net_rest, para, criterion, optimizer, at_lr, DistType, RegularityType, weight_dist, weight_reg, xDim, yDim, zDim, im_info, dev):
    running_loss = 0
    total = 0
    output_folder = './check_atlas'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize atlas
    for ave_scan in aveloader:
        atlas = ave_scan[0]
    atlas.requires_grad = True
    opt = optim.Adam([atlas], lr=at_lr)
    scheduler_opt = lr_scheduler.CosineAnnealingLR(optimizer, T_max=para.solver.epochs)
    scheduler_at_opt = lr_scheduler.CosineAnnealingLR(opt, T_max=para.solver.epochs)
    fluid_params = [2.0, 0, 1]
    lddmm_metirc = lg.FluidMetric(fluid_params)
    for epoch in range(para.solver.epochs):
        net.train()
        net_rest.train()  # Ensure both networks are in training mode
        print('epoch:', epoch)

        # Loop through the first dataloader
        for j, tar_bch in enumerate(trainloader):
            b, c, w, h, l = tar_bch.shape
            optimizer.zero_grad()
            opt.zero_grad()

            atlas_bch = torch.cat(b * [atlas]).reshape(b, c, w, h, l).to(dev).float()
            tar_bch = tar_bch.to(dev).float()

            # Forward pass for the first network
            deformed_bch, flow, latent, mome = net(atlas_bch, tar_bch, registration=True)
            print (latent.shape)

            # Compute loss for the first network
            # Dist = criterion(deformed_bch, tar_bch)
            # Reg = Grad(penalty=RegularityType)
            # Reg_loss = Reg.loss(flow)

            # # Total loss for the first network
            # loss_total = weight_dist * Dist + weight_reg * Reg_loss
            
            h = lg.expmap(lddmm_metirc, mome, num_steps= para.solver.Euler_steps)
            Idef = lg.interp(atlas_bch, h)
            v = lddmm_metirc.sharp(mome)
            reg_term = (v*mome).mean()
            loss_total = (1/(0.02*0.02))*NCC().loss(Idef, tar_bch) + reg_term
            
            loss_total.backward(retain_graph=True)  # Compute gradients for atlas and net

            # Update atlas parameters
            opt.step()
            opt.zero_grad()

        # Loop through the second dataloader
        for k, tar_bch_rest in enumerate(train_restloader):
            b, c, w, h, l = tar_bch_rest.shape
            optimizer.zero_grad()

            tar_bch_rest = tar_bch_rest.to(dev).float()

            # Forward pass for the second network
            deformed_bch_rest, flow_rest, latent_rest, mome_rest = net_rest(atlas_bch, tar_bch_rest, registration=True)
            # if (para.model.shooting =='svf'):
            # # Compute loss for the second network
            #Dist_rest = criterion(deformed_bch_rest, tar_bch_rest)
            #Reg_loss_rest = Grad(penalty=RegularityType).loss(flow_rest)

            # Total loss for the second network
            #loss_total_rest = weight_dist * Dist_rest + weight_reg * Reg_loss_rest
            #else: 
            h = lg.expmap(lddmm_metirc, mome_rest, num_steps= para.solver.Euler_steps)
            Idef = lg.interp(atlas_bch, h)
            v = lddmm_metirc.sharp(mome_rest)
            reg_term = (v*mome_rest).mean()
            loss_total_rest = (1/(0.02*0.02))*NCC().loss(Idef, tar_bch_rest) + reg_term
            
            loss_total_rest.backward(retain_graph=True)  # Compute gradients for net_rest only



        contrastive_loss_value = para.model.contrastive_weight*cosine_similarity_loss_3d(latent, latent_rest)

        # Total loss including the contrastive loss
        total_loss =  contrastive_loss_value
        print (loss_total.item(), "    ", loss_total_rest.item(), "    ", contrastive_loss_value.item())
        total_loss.backward()  # Backpropagate total loss
        if (j%1==0):
            src = atlas_bch[0,...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
            save_path = f'./check_atlas/atlas_{epoch}_{j}.nii.gz'
            src_im = sitk.GetImageFromArray(src, isVector=False)
            src_im.CopyInformation(im_info)
            sitk.WriteImage(src_im, save_path, False)
        # Scheduler updates
        optimizer.step()
        opt.step()

        running_loss += total_loss.item()
        total += running_loss
        running_loss = 0.0

        # Update learning rates
        scheduler_opt.step()
        scheduler_at_opt.step()
        print(f'Epoch {epoch} - Total loss: {total_loss.item()}')

    print('Training completed.')


def main():
    print(torch.__version__)
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = './'
    json_file = 'data'
    keyword = 'train'
    xDim, yDim, zDim, im_info= load_and_preprocess_data(data_dir, json_file, keyword)
    print (xDim, yDim, zDim)
    dataset = SData('./data.json', "train")
    dataset_rest = SData('./data.json', "train_rest")
    ave_data = SData('./data.json', 'linear')
    trainloader = DataLoader(dataset, batch_size= para.solver.batch_size, shuffle=True)
    train_restloader = DataLoader(dataset_rest, batch_size= para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(ave_data, batch_size= 1 , shuffle=False)
    combined_loader = zip(trainloader, train_restloader, aveloader )
    net, net_rest, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
    print (xDim, yDim, zDim)
    train_network(trainloader, aveloader, train_restloader, net, net_rest, para, criterion, optimizer, para.solver.atlas_lr, NCC, 'l1', para.model.dist, para.model.reg, xDim, yDim, zDim, im_info, dev)

if __name__ == "__main__":
    main()








       
    
 
        


