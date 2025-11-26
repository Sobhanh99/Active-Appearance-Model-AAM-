from shape_model import shape_model
from texture_model import texture_model
from visualize_pointset import get_coordinates
from image_warp import normalize_shape
from fit_test_sample import fit_shape, fit_texture, fit_total_image
from combine_variation_modes import get_combine_variation_modes

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave, imread
import cv2

number_of_images = 2819  # total number of samples
number_of_landmarks = 487 # total number of landmarks for each sample
mean_image = 319 # picture number of mean face

def prepare_data(data_dir, precomputed_shape_normalized_texture, save_texture_data_dir = 'C:\\Users\\Lenovo\\Desktop\\aam\\AAM_code\\gray_scale\\warped', mean_image):

	pointset_data = []
	shape_normalized_texture_data, shape_normalized_texture_data_full_res = [], []
	data_list = []

	mean_shape_img_path = os.path.join(data_dir, f'{mean_image}.jpg')

	for i, img_path in enumerate(sorted(os.listdir(data_dir))):
		if not '.jpg' in img_path:
			continue

		individual_img_path = os.path.join(data_dir, img_path)
		x_coordinates, y_coordinates, connect_from, connect_to = get_coordinates(individual_img_path)  #[0]
		x_coordinates = 256*np.expand_dims(x_coordinates, 1); y_coordinates = 256*(1-np.expand_dims(y_coordinates, 1))

		coordinates = np.expand_dims(np.concatenate((x_coordinates, y_coordinates), axis=1), 0)
		pointset_data.append(coordinates)

		# Align individual image to shape normalized coordinates

		if not precomputed_shape_normalized_texture:
			shape_normalized_texture = normalize_shape(individual_img_path, mean_shape_img_path, show=False)
			#print("shape_normalized_texture:",shape_normalized_texture)
			plt.imsave(os.path.join(save_texture_data_dir, img_path), shape_normalized_texture)
		else:
			shape_normalized_texture = plt.imread(os.path.join(save_texture_data_dir, img_path))

		shape_normalized_texture_resize = cv2.resize(shape_normalized_texture,
			(shape_normalized_texture.shape[0]//4, shape_normalized_texture.shape[1]//4))
		shape_normalized_texture_data.append(np.expand_dims((shape_normalized_texture_resize), 0))    #rgb2gray
		shape_normalized_texture_data_full_res.append(np.expand_dims(shape_normalized_texture,0))

		data_list.append(individual_img_path)

	pointset_data = np.concatenate(pointset_data, axis=0)
	shape_normalized_texture_data = np.concatenate(shape_normalized_texture_data, axis=0)
	shape_normalized_texture_data_full_res = np.concatenate(shape_normalized_texture_data_full_res, axis=0)

	return pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res, \
	 connect_from, connect_to, np.array(data_list)


def main(data_dir='...\\Images and Annotations', precomputed_shape_normalized_texture=False):

	# Prepare data
	pointset_data, shape_normalized_texture_data, shape_normalized_texture_data_full_res, \
	connect_from, connect_to, data_list = prepare_data(data_dir, precomputed_shape_normalized_texture)

	N = number_of_images # Total Data size
	Ntest = pointset_data.shape[0]

	# Split Data into Train and Test Set
	test_indices = list(set(range(Ntest)))
	train_indices = not_used_img

	# # Test fitting ability of shape model
	print('----------------------------------Test Shape Model Fits----------------------------------')
	shape_mean = np.load('current_results\\shape_mean.npy')
	shape_cov_matrix = np.load('current_results\\shape_cov.npy')
	shape_eig_values = np.load('current_results\\shape_eigvalues.npy')
	shape_eig_vecs = np.load('current_results\\shape_eigvecs.npy')

	fit_shape(pointset_data,shape_mean, shape_cov_matrix, shape_eig_values, shape_eig_vecs,
		connect_from, connect_to, save_dir='current_results', number_of_images)


	shape_mean = np.load('current_results\\shape_mean.npy')
	shape_cov_matrix = np.load('current_results\\shape_cov.npy')
	shape_eig_values = np.load('current_results\\shape_eigvalues.npy')
	shape_eig_vecs = np.load('current_results\\shape_eigvecs.npy')

	print('----------------------------------Test Texture Model Fits----------------------------------')

	texture_mean = np.load('current_results\\texture_mean.npy')
	texture_cov_matrix = np.load('current_results\\texture_cov.npy')
	texture_eig_values = np.load('current_results\\texture_eigvalues.npy')
	texture_eig_vecs = np.load('current_results\\texture_eigvecs.npy')

	print('Evaluating Texture Model fits on Test-Set...')
	fit_texture(shape_normalized_texture_data, texture_mean, texture_cov_matrix, texture_eig_values, texture_eig_vecs,
		connect_from, connect_to, save_dir='current_results')


if __name__ == '__main__':
	np.random.seed(1243)
	main(precomputed_shape_normalized_texture=True)
