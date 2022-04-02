################################################################################
# CSE 253: Programming Assignment 1
# Code snippet by Michael
# Winter 2020
################################################################################
# We've provided you with the dataset in PA1.zip
################################################################################
# To install PIL, refer to the instructions for your system:
# https://pillow.readthedocs.io/en/5.2.x/installation.html
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict


def load_data(data_dir="./aligned/"):
	""" Load all PNG images stored in your data directory into a list of NumPy
	arrays.

	Args:
		data_dir: The relative directory path to the CompCar image directory.
	Returns:
		images: A dictionary with keys as car types and a list containing images associated with each key.
		cnt: A dictionary that stores the # of images in each car type
	"""
	images = defaultdict(list)

	# Get the list of car type directory:
	for e in listdir(data_dir):
		# excluding any non-directory files
		if not os.path.isdir(os.path.join(data_dir, e)):
			continue
		# Get the list of image file names
		all_files = listdir(os.path.join(data_dir, e))

		for file in all_files:
			# Load only image files as PIL images and convert to NumPy arrays
			if '.jpg' in file:
				img = Image.open(os.path.join(data_dir, e, file))
				images[e].append(np.array(img))

	print("Car types: {} \n".format(list(images.keys())))

	cnt = defaultdict(int)
	for e in images.keys():
		print("{}: {} # of images".format(e, len(images[e])))
		cnt[e] = len(images[e])
	return images, cnt

align = load_data("C:/Users/brand/Desktop/PA1/aligned")
resized = load_data("C:/Users/brand/Desktop/PA1/resized")

print(align[0]['Convertible'][0])

dataset1 = []
dataset2 = []        
dataset3 = []
dataset4 = []
dataset5 = []
dataset6 = []
dataset7 = []
dataset8 = []
dataset9 = []
dataset10 = []

count = 0

for l in [dataset1, dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8,dataset9,dataset10]:
    
    for i in ["Convertible","Minivan","Pickup","Sedan"]:
        for j in range(15):
            j = count * 15 + j
            if j > 147:
                pass
            else :
                l.append(align[0][i][j])
                
    count = count + 1

dataset10.append(align[0]['Convertible'][148])    
dataset10.append(align[0]['Pickup'][148])    
dataset10.append(align[0]['Pickup'][149])        
dataset10.append(align[0]['Sedan'][148])    
dataset10.append(align[0]['Sedan'][149])    

                
def PCA(X, n_components):
	"""
	Args:
		X: has shape Mxd where M is the number of images and d is the dimension of each image
		n_components: The number of components you want to project your image onto. 
	
	Returns:
		projected: projected data of shape M x n_components
		mean_image: mean of all images
		top_sqrt_eigen_values: singular values
		top_eigen_vectors: eigenvectors 
	"""

	mean_image = np.average(X, axis = 0)

	msd = X - mean_image # M x d

	smart_cov_matrix = np.matmul(msd, msd.T)
	eigen_values, smart_eigen_vectors = np.linalg.eig(smart_cov_matrix)

	idx = eigen_values.argsort()[::-1]   
	eigen_values = eigen_values[idx]
	smart_eigen_vectors = smart_eigen_vectors[:,idx]

	eigen_vectors = (np.matmul(msd.T, smart_eigen_vectors)).T # M x d

	row_norm = np.sum(np.abs(eigen_vectors)**2,axis=-1)**(1./2) # M

	normalized_eigen_vectors = eigen_vectors/(row_norm.reshape(-1, 1)) # M x d

	top_eigen_vectors = normalized_eigen_vectors[:n_components].T
	top_sqrt_eigen_values = np.sqrt(eigen_values[:n_components])

	projected = np.matmul(msd, top_eigen_vectors)/top_sqrt_eigen_values

	return projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors            

