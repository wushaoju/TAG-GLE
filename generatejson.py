import SimpleITK as sitk
import os
import glob
import json
import numpy as np
import re
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

def smooth_3d_array(input_array, sigma=1):
    smoothed_array = gaussian_filter(input_array, sigma=sigma)
    return smoothed_array

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1 

filelist =  sorted(glob.glob('./data_file/female_9_seg/*.nii.gz'),  key=extract_number)   ### Current age 
filelist_rest = sorted(glob.glob('./data_file/female_10_seg/*.nii.gz'),  key=extract_number) ### All other ages 
keyword1 = 'train'
keyword2 = 'train_rest'
keyword3 = 'linear'
dictout = {keyword1:[], keyword2:[],keyword3:[]}
target_shape = (128, 128, 128)
total_array = np.zeros((128, 128, 128))
smooth_factor = 1.5

# Create "Smooth_data" folder if it doesn't exist
output_folder = './Smooth_data'
os.makedirs(output_folder, exist_ok=True)

for i in range(0, len(filelist)):
    smalldict = {}
    print(filelist[i])
    file_name = "Hip_current_" + str(i) + "_.nii.gz"
    print (file_name)
    image = sitk.ReadImage(filelist[i])

    image_array = sitk.GetArrayFromImage(image)
    smoothed_data = smooth_3d_array(image_array, sigma=smooth_factor)
    total_array += smoothed_data
    # smoothed_data = resize(smoothed_data, target_shape)
    smooth_image = sitk.GetImageFromArray(smoothed_data)
    smooth_image.SetOrigin(image.GetOrigin())
    smooth_image.SetSpacing(image.GetSpacing())
    smooth_image.SetDirection(image.GetDirection())
    output_file_path = os.path.join(output_folder, file_name)
    sitk.WriteImage(smooth_image, output_file_path)
    smalldict['Data'] = './Smooth_data/' + file_name
    dictout[keyword1].append(smalldict)

for i in range(0, len(filelist_rest)):
    smalldict2 = {}
    file_name = "Hip_rest_" + str(i) + "_.nii.gz"
    print (file_name)
    image = sitk.ReadImage(filelist[i])

    image_array = sitk.GetArrayFromImage(image)
    smoothed_data = smooth_3d_array(image_array, sigma=smooth_factor)
    total_array += smoothed_data
    smooth_image = sitk.GetImageFromArray(smoothed_data)
    smooth_image.SetOrigin(image.GetOrigin())
    smooth_image.SetSpacing(image.GetSpacing())
    smooth_image.SetDirection(image.GetDirection())
    output_file_path = os.path.join(output_folder, file_name)
    sitk.WriteImage(smooth_image, output_file_path)
    smalldict2['Data'] = './Smooth_data/' + file_name
    dictout[keyword2].append(smalldict2)

ave_data = total_array / i
ave_data = resize(ave_data, target_shape)
ave_image = sitk.GetImageFromArray(ave_data)
ave_image.SetOrigin(image.GetOrigin())
ave_image.SetSpacing(image.GetSpacing())
ave_image.SetDirection(image.GetDirection())
ave_file_name = "Mean_hip" + ".nii.gz"
ave_file_path = os.path.join(output_folder, ave_file_name)
sitk.WriteImage(ave_image, ave_file_path)

smalldict3 = {}
smalldict3['Data'] = './Smooth_data/' + ave_file_name
dictout[keyword3].append(smalldict3)




savefilename = './data' + '.json'
with open(savefilename, 'w') as fp:
    json.dump(dictout, fp)
