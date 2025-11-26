import os
import numpy as np
import matplotlib.pyplot as plt

#from shape_utils import *
#from visualize_pointset import *

def shape_model(data, connect_from, connect_to, save_plot_dir = 'current_results', data_list = None):

	# Shape of Data is Nxnxd. N images. n points per each image pointset. d dimensions of the points.

	os.makedirs(save_plot_dir, exist_ok=True)
	N = data.shape[0]
	colors = np.random.rand(N,3)


	# -------------------------- Compute and plot Mean Shape --------------------------- #
	plt.imshow(np.zeros((2,2)))
	for i in range(N):
		plt.scatter(data[i,:,0]*5, data[i,:,1]*5,s=1 ,color=colors[i])
	plt.title('Plot of all initital pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'unaligned_data_scatter.png'))

	plt.clf()

	mean, z_aligned = compute_mean(data)
	#print("data:",data[0])
	#print("z_aligned:",z_aligned[0])
	#print("mean:",mean[0])
	plt.imshow(np.zeros((2,2)))
	for i in range(2):   #40
		plt.scatter(z_aligned[i,:,0]*5, z_aligned[i,:,1]*5,s=1, alpha=0.4)
	plt.scatter(mean[0,:,0]*5, mean[0,:,1]*5,s=1)
	plt.title('Aligned pointsets')
	plt.savefig(os.path.join(save_plot_dir, 'mean_and_aligned_data.png'))
	plt.clf()

	np.save('current_results\\shape_mean.npy', mean)


	# -------------------------- Plot EigenValues --------------------------- #

	try:
		cov_matrix = np.load('current_results\\shape_cov.npy')   #adderess changed
		eig_values = np.save('current_results\\shape_eigvalues.npy')   #adderess changed
		eig_vecs = np.save('current_results\\shape_eigvecs.npy')   #adderess changed
	except:
		cov_matrix = compute_covariance_matrix(z_aligned, mean) # ndxnd matrix
		eig_values, eig_vecs = np.linalg.eig(cov_matrix)
		np.save('current_results\\shape_cov.npy', cov_matrix)      #adderess changed
		np.save('current_results\\shape_eigvalues.npy', eig_values)   #adderess changed
		np.save('current_results\\shape_eigvecs.npy', eig_vecs)   #adderess changed

	idx = eig_values.argsort()[::-1]
	eig_values = eig_values[idx]
	eig_vecs = eig_vecs[:,idx]

	plt.plot(np.real(eig_values[::-1]))
	plt.title('Eigenvalues (in y axis) plot (sorted in ascending order)')
	plt.savefig(os.path.join(save_plot_dir, 'eigen_values.png'))
	plt.clf()


	# -------------------------- Plot modes of variations --------------------------- #

	def get_modes_of_variation(i, scale=3):

		var_plus = mean + scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		var_minus = mean - scale*np.sqrt(np.real(eig_values[i]))*np.real(eig_vecs[:,i]).reshape(mean.shape)
		return var_plus, var_minus

	plt.imshow(np.zeros((2,2)))
	for i in range(1):  #40
		plt.scatter(z_aligned[i,:,0]*5, z_aligned[i,:,1]*5,s=1, alpha=0.15)

	var_1_plus, var_1_minus = get_modes_of_variation(0,scale=3)

	plt.scatter(mean[0,:,0]*5, mean[0,:,1]*5,s=1, label='Mean')
	plt.scatter(var_1_plus[0,:,0]*5, var_1_plus[0,:,1]*5,s=1, label= 'Mean + 3 S.D', color='red')
	plt.scatter(var_1_minus[0,:,0]*5, var_1_minus[0,:,1]*5,s=1, label='Mean - 3 S.D',color='blue')

	plt.title('1st Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean_and_first_mode.png'))
	plt.clf()

	var_2_plus, var_2_minus = get_modes_of_variation(1, scale=5)  #5

	plt.imshow(np.zeros((2,2)))
	for i in range(1):   
		plt.scatter(z_aligned[i,:,0]*5, z_aligned[i,:,1]*5,s=1, alpha=0.15)

	plt.scatter(mean[0,:,0]*5, mean[0,:,1]*5,s=1,label='Mean')
	plt.scatter(var_2_plus[0,:,0]*5, var_2_plus[0,:,1]*5 ,s=1,label= 'Mean + 5 S.D', color='red')
	plt.scatter(var_2_minus[0,:,0]*5, var_2_minus[0,:,1]*5,s=1, label='Mean - 5 S.D',color='blue')

	plt.title('2nd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean_and_second_mode.png'))
	plt.clf()

	var_3_plus, var_3_minus = get_modes_of_variation(2, scale=7)

	plt.imshow(np.zeros((2,2)))
	for i in range(1):   #40
		plt.scatter(z_aligned[i,:,0]*5, z_aligned[i,:,1]*5,s=1 , alpha=0.15)

	plt.scatter(mean[0,:,0]*5, mean[0,:,1]*5,s=1,label='Mean')
	plt.scatter(var_3_plus[0,:,0]*5, var_3_plus[0,:,1]*5,s=1 , label= 'Mean + 7 S.D', color='red')
	plt.scatter(var_3_minus[0,:,0]*5, var_3_minus[0,:,1]*5,s=1, label='Mean - 7 S.D',color='blue')

	plt.title('3rd Mode of Variation with all the aligned pointsets')
	plt.legend()
	plt.savefig(os.path.join(save_plot_dir, 'mean-and-third-mode.png'))
	plt.clf()


	# -------------------------- Compare modes of variations  --------------------------- #

	z_closest_mean, index_mean = get_closest_pointset(z_aligned, mean)  # I changed data to z_aligned
	print("index mean:",index_mean)
	z_closest_var_1_plus, index_mean_plus = get_closest_pointset(z_aligned, var_1_plus)			# I changed data to z_aligned
	print("index mean plus:",index_mean_plus)
	z_closest_var_1_minus, index_mean_minus = get_closest_pointset(z_aligned, var_1_minus)		# I changed data to z_aligned
	print("index_mean_minus",index_mean_minus)

	#print("data_list[index_mean]: ",data_list[index_mean])
	visualize_checkpoints(data_list[index_mean], show=False)
	plt.title('Image closest to the the mean shape')
	plt.savefig(os.path.join(save_plot_dir, 'closest_mean.png'))
	plt.clf()

	visualize_checkpoints(data_list[index_mean_plus], show=False)
	plt.title('Image closest to Mean shape +3 S.D along the top mode of variation')
	plt.savefig(os.path.join(save_plot_dir, 'closest_var_plus.png'))
	plt.clf()

	visualize_checkpoints(data_list[index_mean_minus], show=False)
	plt.title('Image closest to Mean shape -3 S.D along the top mode of variation')
	plt.savefig(os.path.join(save_plot_dir, 'closest_var_minus.png'))
	plt.clf()
