#visualize_pointset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Delaunay

def get_coordinates(img_lm):
    x_values=[]
    y_values=[]

    if type(img_lm)==str:
        s=os.path.splitext(os.path.basename(img_lm))[0]
        landmarks=arr_SH[int(s)-1]

    else:
        img_lm=np.array(img_lm)
        landmarks=img_lm #.reshape(-1,2)

    for i in range(len(landmarks)):
        x_values.append(landmarks[i][0])
        y_values.append(landmarks[i][1])
    x_values=np.asarray(x_values)
    y_values=np.asarray(y_values)



    # Create a Delaunay triangulation of the landmarks
    tri = Delaunay(np.column_stack((x_values, y_values)))

    # Get the simplices of the triangulation
    simplices = tri.simplices

    # Define the connections between the landmarks using the simplices

    connect_from = np.concatenate([tri.simplices[:, 0], tri.simplices[:,1], tri.simplices[:,2]],axis=0)
    connect_to = np.concatenate([tri.simplices[:, 1], tri.simplices[:,2], tri.simplices[:,0]], axis=0)

    return x_values, y_values, connect_from, connect_to

def plot_pointset_with_connections(x_values, y_values, connect_from, connect_to, label='connections', color='black', number_of_landmarks):
    if color == 'black':
        plt.scatter(x_values, y_values)
    else:
        plt.scatter(x_values, y_values, color=color)

    for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
        if i == number_of_landmarks:
            break
        plt.plot([x_values[c_f], x_values[i]], [y_values[c_f], y_values[i]], color=color)
        plt.plot([x_values[i], x_values[c_t]], [y_values[i], y_values[c_t]], color=color)


def visualize_checkpoints(img_path, show=True):
	if type(img_path)==str:
		img = plt.imread(img_path)
		s = os.path.splitext(os.path.basename(img_path))[0]
	if type(img_path)==np.ndarray:    #I added
		img=img_path   # I added
		for i in range(len(data)):
			if (data[i]-img).any()==0:
				s=int(i)+1
				#print(s)
				break
	else:
		img_path=str(f"{img_path}")
		s = os.path.splitext(os.path.basename(img_path))[0]
		img = plt.imread(img_path)



	x_values, y_values, connect_from, connect_to = get_coordinates(arr_SH[int(s)-1])  #img_lm
	x_values *= img.shape[1]; y_values *= img.shape[0]
	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.scatter(x_values, y_values)
	for i, (c_f, c_t) in enumerate(zip(connect_from, connect_to)):
		ax.plot([x_values[c_f], x_values[c_t]], [y_values[c_f], y_values[c_t]], color='red')
	if show:
		plt.show()

if __name__ == '__main__':
    img_path = "...\\Images and Annotations/126.jpg"
    visualize_checkpoints(img_path)
