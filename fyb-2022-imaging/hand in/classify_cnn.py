import pickle
import cv2
import groupXY_functions as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


def classify(img_arr, mask_arr):
    # Set dimensions for cropping and reshaping
    rescale_width = 350
    rescale_height = 260

    # Scale image and mask
    img_arr, mask_arr = util.scale_image_res(img_arr, mask_arr, dim = (rescale_height, rescale_width))

    # Mask out lesion from image
    masked_img = cv2.bitwise_and(img_arr, img_arr, mask = mask_arr)

    # Add image to array and reshape array and standardize pixel values (0-1)
    X = []
    X.append(masked_img)
    X = np.array(X).reshape(-1, rescale_width, rescale_height, 3)   
    X = X / 255

    # Load in saved CNN model and do prediction based on given image and mask 
    cnn_loaded = load_model('melanoma_model_training_end_07_04_2022-18_03_58')
    y_pred = cnn_loaded.predict(X)

    print(y_pred[0][0])
    return y_pred[0][0]


if __name__ == "__main__":

	img_path = 'data/example_image/ISIC_0014055.jpg'
	mask_path = 'data/example_segmentation/ISIC_0014055_segmentation.png'

	#please load images as cv2 to be in correct color space (BGR)
	img = cv2.imread(img_path)
	mask = cv2.imread(mask_path,0)


	classify(img,mask)