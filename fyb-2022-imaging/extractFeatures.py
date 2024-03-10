import cv2
import csv
import pandas as pd
import numpy as np
import path
import groupXY_functions as util
import matplotlib.pyplot as plt



df = pd.read_csv(path.ground_truth)


for index, line in df.iterrows():
	id_ = line['image_id'] 
	
	img = cv2.imread(path.images+f'{id_}.jpg')
	mask = cv2.imread(path.segmentation+f'{id_}_segmentation.png',0)

	img, mask = util.scale_image_res(img,mask)
	
	img_plt, mask_plt = plt.imread(path.images+f'{id_}.jpg'), plt.imread(path.segmentation+f'{id_}_segmentation.png')
	img_plt, mask_plt = util.scale_image_res(img_plt,mask_plt)	
 
	res = cv2.bitwise_and(img,img,mask = mask)
	notmask = cv2.bitwise_not(mask)
	notres = cv2.bitwise_and(img,img,mask = notmask)
	
	a,b,c,d = util.extract_bboxes(np.array(mask))
 
	print(index)
	
	#df.at[index, 'symmetry'] = util.symetryScore(mask[a:b,c:d])

	df.at[index, ['color_lesion_r','color_lesion_g','color_lesion_b']] = util.averageColorsSimple(res)

	df.at[index, ['color_skin_r','color_skin_g','color_skin_b']] = util.averageColorsSimple(notres)

	df.at[index, ['area','perimeter']] = util.measure_area_perimeter(mask)

	df.at[index, ['min_symmetry', 'avg_symmetry']] = util.all_deg(mask[a:b,c:d])

	# df.at[index, ['border_dark', 'border_skin']] = util.boundry_check2(img_plt,mask_plt)

	df.at[index, ['color_variance']] = util.color_variance(img_plt, mask_plt)

df.to_csv(path.out+'out_rescaled.csv')
