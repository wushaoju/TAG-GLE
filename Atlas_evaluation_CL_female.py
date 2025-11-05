#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:09:22 2024

@author: Shaoju WU
"""

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
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter
from medpy.metric.binary import hd, volume_correlation, ravd, asd
import scipy.io
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
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7, #change this step to be 1 as elastix
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()

    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, criterion, optimizer



def dice(y_pred, y_true):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.1):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img


def calculate_non_positive_jacobian_percentage_numpy(jacobian_determinant_array):
    """
    Calculate the percentage of voxels with a non-positive Jacobian determinant using NumPy.

    Parameters:
        jacobian_determinant_array (numpy.ndarray): A 3D NumPy array of Jacobian determinants.

    Returns:
        float: Percentage of voxels with a non-positive Jacobian determinant.
    """
    # Count voxels with non-positive Jacobian determinant
    non_positive_count = np.sum(jacobian_determinant_array <= 0)
    total_voxels = jacobian_determinant_array.size

    # Calculate percentage
    percentage = (non_positive_count / total_voxels) * 100

    return percentage

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def test_network(index, output_folder, testloader, atlas_path, net, para, criterion, DistType, RegularityType, weight_dist, weight_reg, xDim, yDim, zDim, im_info, dev, case_num):
    total_loss = 0
    total_samples = 0
    output_folder = output_folder + 'check_atlas_target_'+str(case_num)+'_rest_4_to_12_CL_pos_'  + str(index)
    result_output_folder = output_folder + '/Dice_evaluation_cross_age/'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(result_output_folder, exist_ok=True)

    # Load the atlas from the average loader
    #for ave_scan in aveloader:
    #    atlas = ave_scan[0]
    
    atlas = sitk.ReadImage(atlas_path)
    atlas = torch.from_numpy(sitk.GetArrayFromImage(atlas)).unsqueeze(0)
    atlas = atlas.to(dev).float()  # Ensure the atlas is on the correct device
    Dice_total = []
    ASD_total = []
    VD_total = []
    
    neg_percent_total = []
    net.eval()  # Set the network to evaluation mode
    with torch.no_grad():  # Disable gradient computations for testing
        for j, tar_bch in enumerate(testloader):
            b, c, w, h, l = tar_bch.shape

            # Repeat atlas for the batch
            atlas_bch = torch.cat(b * [atlas]).reshape(b, c, w, h, l).to(dev).float()
            tar_bch = tar_bch.to(dev).float()

            '''Run registration'''
            deformed_bch, flow, latent, mome = net(atlas_bch, tar_bch, registration=True)

            # Compute loss
            Dist = criterion(deformed_bch, tar_bch)
            Reg = Grad(penalty=RegularityType)
            Reg_loss = Reg.loss(flow)

            loss_total = weight_dist * Dist + weight_reg * Reg_loss
            total_loss += loss_total.item()
            total_samples += b
            
            
            
            vel_field = flow[0].detach().cpu().numpy() # Compute the determine of jacobian 
            det_jac = jacobian_determinant_vxm(vel_field)
            negative_percentage = calculate_non_positive_jacobian_percentage_numpy(det_jac)
            neg_percent_total.append(negative_percentage)
            
            '''Save results for visualization and analysis'''
            if j % 1 == 0:
                src = atlas_bch[0, ...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                save_path = f'{result_output_folder}/atlas_{j}.nii.gz'
                src_im = sitk.GetImageFromArray(src, isVector=False)
                src_im.CopyInformation(im_info)
                sitk.WriteImage(src_im, save_path, False)

                deformed = deformed_bch[0, ...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                smooth_deformed = smooth_seg(deformed, sigma=0.0, thresh=0.1)
                save_path = f'{result_output_folder}/deformed_{j}.nii.gz'
                def_im = sitk.GetImageFromArray(deformed.astype(float), isVector=False)
                def_im.CopyInformation(im_info)
                sitk.WriteImage(def_im, save_path, False)
                
                
                save_path = f'{result_output_folder}/deformed_mask{j}.nii.gz'
                def_im = sitk.GetImageFromArray(smooth_deformed.astype(float), isVector=False)
                def_im.CopyInformation(im_info)
                sitk.WriteImage(def_im, save_path, False)

                target = tar_bch[0, ...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                smooth_target = smooth_seg(target, sigma=0.0, thresh=0.1)
                
                Dice_coefficient = dice(smooth_deformed, smooth_target)
                Dice_total.append(Dice_coefficient)
                VD_distance = ravd(smooth_deformed, smooth_target)
                ASD_distance = asd(smooth_deformed, smooth_target)

                VD_total.append(VD_distance)
                ASD_total.append(ASD_distance)
                
                save_path = f'{result_output_folder}/target_{j}.nii.gz'
                tar_im = sitk.GetImageFromArray(target.astype(float), isVector=False)
                tar_im.CopyInformation(im_info)
                sitk.WriteImage(tar_im, save_path, False)
                
                
                save_path = f'{result_output_folder}/target_mask{j}.nii.gz'
                tar_im = sitk.GetImageFromArray(smooth_target.astype(float), isVector=False)
                tar_im.CopyInformation(im_info)
                sitk.WriteImage(tar_im, save_path, False)

    # Print the average loss for testing
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    print('Total testing loss:', total_loss)
    print('Average testing loss per sample:', avg_loss)
    print("Dice score:", np.mean(Dice_total))
    print("std for Dice score:", np.std(Dice_total))
    
    print("Non-negative jac %:", np.mean(neg_percent_total))
    print("std for Non-negative jac %:", np.std(neg_percent_total))
    
    print("VD score:", np.mean(np.abs(VD_total))*100.0)
    print("std for VD score:", np.std(np.abs(VD_total))*100.0)
    
    print("ASD score:", np.mean(ASD_total)*2.0)
    print("std for ASD score:", np.std(ASD_total)*2.0)
    
    return np.mean(Dice_total), np.mean(np.abs(VD_total))*100.0, np.mean(ASD_total)*2.0


def train_network(output_folder, trainloader, aveloader, net, para, criterion, optimizer, at_lr, DistType, RegularityType, weight_dist, weight_reg,  xDim, yDim, zDim, im_info, dev, case_num):
    running_loss = 0
    total = 0
    output_folder = output_folder + 'male_raw_check_atlas_00_' + str(case_num)
    os.makedirs(output_folder, exist_ok=True)

    for ave_scan in aveloader:
        atlas = ave_scan[0]
    # print (atlas.sum())
    atlas.requires_grad=True
    opt= optim.Adam([atlas], lr = at_lr) 
    scheduler_opt = lr_scheduler.CosineAnnealingLR(optimizer, T_max=para.solver.epochs)
    scheduler_at_opt = lr_scheduler.CosineAnnealingLR(opt, T_max=para.solver.epochs)

    for epoch in range(para.solver.epochs):
        net.train()
        print('epoch:', epoch)

        for j, tar_bch in enumerate (trainloader):
            b, c, w, h, l = tar_bch.shape
            optimizer.zero_grad()

            atlas_bch = torch.cat(b*[atlas]).reshape(b , c, w , h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch = tar_bch.to(dev).float() 

            '''Mask out lesions and run registration'''
            deformed_bch, flow = net(atlas_bch, tar_bch ,  registration=True)

            Dist = criterion(deformed_bch, tar_bch) 
            Reg = Grad( penalty= RegularityType)
            Reg_loss  = Reg.loss(flow)


            loss_total = weight_dist * Dist + weight_reg* Reg_loss
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

            '''Save and checking results'''
            if (j%5==0):
                # velo = flow[0,...].reshape(3, xDim, yDim, zDim).permute(1, 2, 3, 0)
                # velo = velo.detach().cpu().numpy()
                # save_path = f'./check_atlas/velo_{epoch}_{j}.nii.gz'

                # sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=True), save_path, False)

                # defim = deformed_bch[0,...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                # save_path = f'./check_atlas/deform_{epoch}_{j}.nii.gz'
                # def_im =sitk.GetImageFromArray(defim, isVector=False)
                # def_im.CopyInformation(im_info)
                # sitk.WriteImage(def_im, save_path, False)

                # tar = tar_bch[0,:,:,:,:].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                # save_path = f'./check_atlas/tar_{epoch}_{j}.nii.gz'
                # tar_im =sitk.GetImageFromArray(tar, isVector=False)
                # tar_im.CopyInformation(im_info)
                # sitk.WriteImage(tar_im, save_path, False)

                src = atlas_bch[0,...].reshape(xDim, yDim, zDim).detach().cpu().numpy()
                save_path = f'./check_atlas/atlas_{epoch}_{j}.nii.gz'
                src_im = sitk.GetImageFromArray(src, isVector=False)
                src_im.CopyInformation(im_info)
                sitk.WriteImage(src_im, save_path, False)

            opt.step()
            opt.zero_grad()
        scheduler_opt.step()
        scheduler_at_opt.step()
        print('Total training loss:', total)

def main():

    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = './'
    Dice_score = []
    VD_score = []
    ASD_score = []
    
    contrastive_weight_index = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    i = 13
    for i in range(4,19):
        
        json_file = 'data'
        keyword = 'train'
        xDim, yDim, zDim, im_info= load_and_preprocess_data(data_dir, json_file, keyword)
        print (xDim, yDim, zDim)
        folder_name = './female_raw_only_Smooth_data_00_'
        #folder_name = './data_json/data_'
        output_folder = './check_atlas_female/'

        atlas_path = './check_atlas_female/check_atlas_target_'+str(i)+'_rest_LOOCLSIG' + '/atlas_950_1.nii.gz' 
        #atlas_path = './check_atlas_male/check_atlas_target_'+str(i)+'_rest_4_to_18_CL_'+ str(j) + '/atlas_950_2.nii.gz' 
        ave_data = SData(folder_name+str(i)+'.json', 'linear')
        dataset = SData(folder_name+str(i)+'.json', "train")
        trainloader = DataLoader(dataset, batch_size= 1, shuffle=True)
        aveloader = DataLoader(ave_data, batch_size= 1 , shuffle=False)
        #combined_loader = zip(trainloader, aveloader )
        net, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
        checkpoint = torch.load('./check_atlas_female/check_atlas_target_'+str(i)+'_rest_LOOCLSIG'+'/trained_model/model_atlas_weights.pth') ###########Change this
        #checkpoint = torch.load('./check_atlas_male/check_atlas_target_'+str(i)+'_rest_4_to_18_CL_'+ str(j) +'/trained_model/model_cross_weights.pth') ###########Change this
        net.load_state_dict(checkpoint)
        net = net.to(dev)
        print (xDim, yDim, zDim)
        #train_network(output_folder, trainloader, aveloader, net, para, criterion, optimizer, para.solver.atlas_lr, NCC, 'l2', para.model.dist, para.model.reg, xDim, yDim, zDim, im_info, dev, i)
        Dice, VD, ASD  = test_network(0, output_folder, trainloader, atlas_path, net, para, criterion, NCC, 'l2', para.model.dist, para.model.reg, xDim, yDim, zDim, im_info, dev, i)
        Dice_score.append(Dice)
        VD_score.append(VD)
        ASD_score.append(ASD)
        
    scipy.io.savemat("Contrastive_learning_evaluation_female.mat", {"Dice_score": Dice_score, "VD_score": VD_score, "ASD_scoree": ASD_score})
    print("VD score for all the cases:", np.mean(VD_score))
    print("STD of VD score for all the cases:", np.std(VD_score))
    print("ASD score for all the cases:", np.mean(ASD_score))
    print("STD of ASD score for all the cases:", np.std(ASD_score))
    print("Dice score for all the cases:", np.mean(Dice_score))
    print("STD of Dice score for all the cases:", np.std(Dice_score))
if __name__ == "__main__":
    main()








       
    
 
        


