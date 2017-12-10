#the github https://github.com/kaustubh0mani/Indian_Pines/blob/master/Code/Data_Preparation_Image_Split.py
# maybe 其他的数据集 http://lesun.weebly.com/hyperspectral-data-set.html

import scipy
import scipy.io as io

import numpy as np
#input_image = io.loadmat('19920612_AVIRIS_IndianPine_Site3.tif')   #shape [145,145,220]
#target_image = io.loadmat('Indian_pines_gt.mat')['indian_pines_gt'] # shape[145, 145]

#
# train_image = input_image[:73]
# train_labels = target_image[:73]
# test_image = input_image[73:]
# test_labels = target_image[73:]
framedim = [2048,2048]
nb_elem = framedim[0]*framedim[1]
offset = 4096
formatdata = np.uint16
f = open('19920612_AVIRIS_IndianPine_Site3.tif', 'rb')
f.seek(offset)#TODO: only header size for tiff !!
d = np.fromfile(f, dtype=formatdata, count=nb_elem).reshape(framedim)


print(d.shape)
