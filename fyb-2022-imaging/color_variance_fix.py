import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode
import cv2
from scipy import ndimage
import skimage.morphology as morphology
from skimage.measure import label as sklabel
from sklearn.cluster import MiniBatchKMeans
from math import sqrt
import random

from tqdm import tqdm

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def scale_image_res(img, mask, dim = (1024, 768)):
    
    # Rezises each image to be the same dimensions as the smallest image (1024,768), making the code run faster.
    
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    resized_mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)

    return resized_img, resized_mask

def color_variance(image, mask):
    '''function to quantify the color variantion of the lesion.
    The input is the original image name. The function loads the segmentation mask.
    It uses extract_mask_roi() function given below.
    The function returns a number approximately between 60 and 200.
    The greater the number, the greater the color variance'''

    #grab the width and height of an image
    (h, w) = image.shape[:2]

    #cut the working space closer to the lesion borders using extract_mask_roi function given below
    image, _ = extract_mask_roi(image, mask, 20, 20)

    #convert the image from the RGB color space to L*a*b* for euclidean distance
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means can be applied (2 dimensions instead of 3)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #REDUCING COLORS
    #apply k-means using the specified number of clusters and create a quantized image based on predictions

    #using MiniBatchKMeans instead of KMeans for speed
    clt = MiniBatchKMeans(n_clusters = 16, random_state=1)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    #filtering only to unique colors
    colors = np.unique(quant, axis = 0)

    #filtering out all black pixels
    #threshold <= 2 prevents taking into account not-perfect black (but still black)
    #[0] is the luminanace index in L*a*b*
    colors_ = np.delete(colors, np.where(colors <= 2)[0], axis=0) 

    #CALCULATIONS 
    #the calculations take luminance into consideration

    #conversion to a python list for iteration
    color_variance = colors_.tolist()
    max_distance = 0
    for i in color_variance:
        for j in color_variance:
            distance = sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)
            if distance > max_distance:
                max_distance = distance

    return max_distance

def extract_mask_roi(image, mask, lower_bound, upper_bound):
    '''
    Function to extract the region of interest from the image and mask.
    The input is the original image, the original mask and the threshold value.
    The function returns the region of interest as a new numpy array.
    '''
    # NOTE: Images that are taken as very close ups (ISIC_0014712.jpg for example) will return the most prominent color as the color in the legion, and thus not cut out the skin color
    
    # Get the most dominant color in the image
    # https://stackoverflow.com/a/50900143/8660908
    a2D = image.reshape(-1,image.shape[-1])
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    color_range = np.unravel_index(np.bincount(a1D).argmax(), col_range)
    # Add some upper and lower bounds for the color range
    # Make sure lower and upper are within the color range
    lower = np.maximum(np.array(color_range) - lower_bound, 0)
    upper = np.minimum(np.array(color_range) + upper_bound, 255)


    # Show only the original mask on the entire image
    im_roi_opened = image.copy()
    im_roi_opened[mask==0] = 0

    # Remove everything that is not in the range
    # This should amount to the skin being removed
    cv_mask = cv2.inRange(im_roi_opened, lower, upper)
    # cv_image = cv2.bitwise_not(im_roi_opened, im_roi_opened, mask = cv_mask)
    cv_image = im_roi_opened.copy()
    cv_image[cv_mask!=0] = 0

    # Get the biggest connected component
    # https://stackoverflow.com/a/63842045/8660908
    img_bw = cv_image > 0
    labels = sklabel(img_bw, return_num=False)
    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat))

    # Apply the maxCC_nobcg mask to the cv_image
    cv_image_nobcg = cv_image.copy()
    cv_image_nobcg[maxCC_nobcg==0] = 0

    # Turn cropped mask image into grayscale
    cv_image_nobcg_gray = cv2.cvtColor(cv_image_nobcg, cv2.COLOR_BGR2GRAY)
    # Turn into black and white
    cv_image_nobcg_mask = cv_image_nobcg_gray > 0 # > 0 because of the grayscale

    # Return image and mask
    return cv_image_nobcg, cv_image_nobcg_mask


    
path = "ISIC_0001769"
image = plt.imread(f'data/example_image/{path}.jpg')
mask = plt.imread(f'data/example_segmentation/{path}_segmentation.png')

im, msk = scale_image_res(image, mask)
print(color_variance(im, msk))
