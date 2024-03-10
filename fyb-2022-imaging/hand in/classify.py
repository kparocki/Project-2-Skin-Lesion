import pickle
import cv2
import groupXY_functions as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def classify(img, mask):
	with open('pickle_knn.pickle','rb') as pickle_in:
		knn_loaded = pickle.load(pickle_in)

	with open('pickle_scalar.pickle','rb') as pickle_in:
		scalar_loaded = pickle.load(pickle_in)
	
	df = pd.DataFrame()

	img, mask = util.scale_image_res(img,mask)
	
	img_plt, mask_plt = plt.imread(img_path), plt.imread(img_path)
	img_plt, mask_plt = util.scale_image_res(img_plt,mask_plt)	
 
	# remove skin from lesion pictures.
	res = cv2.bitwise_and(img,img,mask = mask)
	notmask = cv2.bitwise_not(mask)
	notres = cv2.bitwise_and(img,img,mask = notmask)
	
	# get bounding box
	a,b,c,d = util.extract_bboxes(np.array(mask))

	index = 0
	
	df.at[index, ['color_lesion_r','color_lesion_g','color_lesion_b']] = util.averageColorsSimple(res)

	df.at[index, ['color_skin_r','color_skin_g','color_skin_b']] = util.averageColorsSimple(notres)

	df.at[index, ['area','perimeter']] = util.measure_area_perimeter(mask)

	df.at[index, ['min_symmetry', 'avg_symmetry']] = util.all_deg(mask[a:b,c:d])

	# df.at[index, ['border_dark', 'border_skin']] = util.boundry_check2(img_plt,mask_plt)

	df.at[index, ['color_variance']] = util.color_variance(img_plt, mask_plt)

	df['std_min_symmetry'] = df['min_symmetry'] / (df['area'] / 255)
	df['std_avg_symmetry'] = df['avg_symmetry'] / (df['area'] / 255)

	df.loc[ df['std_min_symmetry'] > 254, 'std_min_symmetry'] = 1
	df.loc[ df['std_avg_symmetry'] > 254, 'std_avg_symmetry'] = 1

	df['circularity'] = (4*np.pi*df['area'])/(df['perimeter'])**2

	df.drop(['min_symmetry', 'avg_symmetry', 'area', 'perimeter', 'color_lesion_b', 'color_lesion_g', 'color_skin_g', 'color_skin_b'], axis=1, inplace=True)

	
	scaled_features = scalar_loaded.transform(df)

	scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


	y = knn_loaded.predict(scaled_features_df[['color_variance', 'std_avg_symmetry']])

	print(y)

if __name__ == "__main__":

	img_path = 'data\example_image\ISIC_0014055.jpg'
	mask_path = 'data\example_segmentation\ISIC_0014055_segmentation.png'


	#please load images as cv2 to be in correct color space (BGR)
	img = cv2.imread(img_path)
	mask = cv2.imread(mask_path,0)


	classify(img,mask)