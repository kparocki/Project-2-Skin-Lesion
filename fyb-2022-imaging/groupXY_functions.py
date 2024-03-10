import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import morphology
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
    
    # Rezises each image to be the same dimensions as the smallest image (768,1024), making the code run faster.
    
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

#NOTE: THIS IS OLD FUNCTION
# def color_variance(image, mask):
#     '''function to quantify the color variantion of the lesion.
#     The input is the original image name. The function loads the segmentation mask.
#     It uses extract_mask_roi() function given below.
#     The function returns a number approximately between 60 and 200.
#     The greater the number, the greater the color variance'''
    
#     random.seed(10)
#     # load the image and grab its width and height
#     (h, w) = image.shape[:2]

#     #cut the working space to lesion borders (or smaller). Mask is based on the original image.
#     #lesion should return the image with reduced colors trimmed by the original mask

#     _, mask2 = extract_mask_roi(image, mask, 20, 20)
#     image[mask2==0] = 0 #if the mask is black at some spot, set image color to black at that spot

#     # convert the image from the RGB color space to the L*a*b*
#     # color space -- since we will be clustering using k-means
#     # which is based on the euclidean distance, we'll use the
#     # L*a*b* color space where the euclidean distance implies
#     # perceptual meaning 

#     #the important thing is this color space is perceptually uniform - the distances between colors are stable, and we can say
#     #that similar colors always lay within a threshold

#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#     # reshape the image into a feature vector so that k-means
#     # can be applied. Basically, we lose the shape of the picture (1 axis of our 3), but preserve color infos
#     #for K clustering, which works on a 2-dimensional array. Think of scattering the points from their xy coordinates into a random space
#     image = image.reshape((image.shape[0] * image.shape[1], 3))

#     # apply k-means using the specified number of clusters (8) and
#     # then create the quantized image based on the predictions

#     clt = MiniBatchKMeans(n_clusters = 16)
#     labels = clt.fit_predict(image)
#     quant = clt.cluster_centers_.astype("uint8")[labels]

#     # reshape the feature vectors to images
#     quant = quant.reshape((h, w, 3))
#     image = image.reshape((h, w, 3))

#     # convert from L*a*b* to BGR and then to RGB gor matplotlib plotting
#     quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
#     image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

#     quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     #graph = np.hstack([image, quant]) #hstack just makes them appear one after another for comparison
#     plt.imshow(quant)
#     plt.tight_layout()

#     #converting back to LAB color space to measure color variance later uniformly
#     lesion_ = cv2.cvtColor(quant, cv2.COLOR_RGB2BGR)
#     lesion_ = cv2.cvtColor(quant, cv2.COLOR_BGR2LAB)

#     #losing xy values to focus on colors
#     colors = lesion_.reshape((lesion_.shape[0] * lesion_.shape[1], 3))

#     #filtering only to unique colors - this takes most time
#     colors = np.unique(colors, axis = 0)

#     #filtering out all black pixels
#     colors_ = np.delete(colors, np.where(colors <= 2)[0], axis=0) #colors[0] is the luminanace row in LAB

#     #CALCULATIONS - the calculations take luminance into consideration

#     #conversion to a python list
#     color_variance = colors_.tolist()
#     max_distance = 0
#     for i in color_variance:
#         for j in color_variance:
#             distance = sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2)
#             # print(i,j,distance)
#             if distance > max_distance:
#                 max_distance = distance

#     return max_distance

def get_color_variability(filtered_image, measure='variance'): # image needs to be filteredd for lesion
    r, g, b = filtered_image.split() # converting to separate channels  
    r= np.array(r) 
    g= np.array(g)
    b= np.array(b) 
    if measure == 'variance': rgb=(np.var(r[r > 0]),np.var(g[g > 0]),np.var(b[b > 0]))
    elif measure == 'standard_deviation': rgb=(np.std(r[r > 0]),np.std(g[g > 0]),np.std(b[b > 0]))
    else: return 
    return np.mean(rgb)

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

def boundry_check2(image,mask):
    
    ## Creates two masks from the extract_mask_roi,dark_mask is ran with larger variables to cut away more, allowing us to later
    ## extract a mask representing the boundry, however this mask is innacurate and does not always properly crop the boundry
    boundry_unfiltered, boundry_mask = extract_mask_roi(image,mask, 20,20)
    darkest_part, dark_mask = extract_mask_roi(image,mask, 40,40)
    
    
    ## Ensures the mask is only containing values of 0 and 1
    boundry_mask[boundry_mask > 0] = 1
    dark_mask[dark_mask > 0] = 1    
    boundry_mask_proper = ((np.abs((boundry_mask)*1)).astype(int))
    darkest_part_mask = (np.abs((dark_mask)*1)).astype(int)
    
    
    
    ## Crops so that we only get border, and skin
    true_boundry = boundry_mask_proper - darkest_part_mask
    only_skin_mask = np.invert(boundry_mask)
    
    
    ## imposes on the picure to only show the desired part
    skin = image.copy()
    skin[only_skin_mask.astype(int) == 0] = [0,0,0]
    
    boundry = image.copy()
    boundry[true_boundry.astype(int) == 0] = [0,0,0]
    
    dark_part = image.copy()
    dark_part[darkest_part_mask.astype(int)== 0] = [0,0,0]
    
    
    ## gets the average RGB values of all the wanted parts
    average_dark = modified_average(dark_part)
    average_boundry = modified_average(boundry)
    average_skin = modified_average(skin)
    
    
    ## Converts the average colors to RGB so that we can compare the colours easily
    average_dark_rgb = sRGBColor(average_dark[0],average_dark[1],average_dark[2])   
    average_boundry_rgb = sRGBColor(average_boundry[0],average_boundry[1],average_boundry[2])
    average_skin_rgb = sRGBColor(average_skin[0],average_skin[1],average_skin[2])

    color_dark_lab = convert_color(average_dark_rgb, LabColor)
    color_boundry_lab = convert_color(average_boundry_rgb, LabColor)
    color_skin_lab = convert_color(average_skin_rgb, LabColor)
    
    ## Returns the colour difference
    return delta_e_cie2000(color_dark_lab, color_boundry_lab), delta_e_cie2000(color_boundry_lab,color_skin_lab)

def modified_average(img):
    non_black = np.any(img != [0, 0, 0], axis=-1)
    average_ = np.average(img[non_black], axis=0)
    return average_

# def extract_mask_roi(image, mask, thresh=95):
#     # NOTE: This is the old function
#     '''
#     Function to extract the region of interest from the image and mask.
#     The input is the original image, the original mask and the threshold value.
#     The function returns the region of interest as a new numpy array.
#     '''
#     # Get region of interest from original mask
#     # Replace the non-lesion pixels
#     im_roi = image.copy()
#     # If the mask is black at a pixel, set image color to black at that same pixel
#     im_roi[mask==0] = 0

#     # Turn the region of interest into gray
#     im_gray = cv2.cvtColor(im_roi, cv2.COLOR_BGR2GRAY)

#     # Create our custom mask with pixels that are lower than threshold
#     custom_mask = im_gray < thresh
#     # Generate a flask disk-shaped footprint
#     struct_el = morphology.disk(10)

#     # Apply the footprint to the custom mask
#     opened = opening(custom_mask, struct_el)

#     # Show only the opened mask on the entire image
#     im_roi_opened = im_roi.copy()
#     im_roi_opened[opened==0] = 0

#     return im_roi_opened

def averageColorsSimple(img,):
    average_ = np.average(img, axis=0)
    return np.average(average_, axis=0)

def middle(mask):
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

def symetryScore(mask):

    lst = [45,90,180,270,55]
    pivot = middle(mask)
    padX = [mask.shape[1] - pivot[0], pivot[0]]
    padY = [mask.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(mask, [padY, padX], 'constant')
    res = []
    for i,x in enumerate(lst):
        rotated = ndimage.rotate(mask, x)
        if rotated.shape[1] % 2 == 1:
            rotated = np.pad(rotated, [0,1], 'constant')
        mid = rotated.shape[1]//2
        
        fold = rotated[:,:mid] - np.fliplr(rotated[:,mid:])
        res.append(np.sum(fold))
    
    return np.average(res)


def extract_bboxes(mask):
    # Finds the values rmin and rmax and cmin cmax to be the first and last value where the mask is not 0 on both axis,
    # enables us to crop the mask, saving time
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rotate_bound(image ,angle):
    
    ## This code is from the Imutils library, and used to rotate an image without cutting away any of the mask
    
    
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = middle(image)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def sym_detect(img):
    # Ensures the mask is only consisting of values being 0 or 255, as they change when rotated
    img[img > 0] = 255

    # finds the center of mass of the mask
    center = middle(img)
    
    # Cuts the image vertically from where the x axis is equals to the center
    first_half = img[:,:int(center[0])]
    second_half = img[:,int(center[0]):-1]
    
    
    # Adds additional coloumns that are filled with 0 values to the smaller picture so that they can be overlapped

    _, first_size = first_half.shape
    _, second_size = second_half.shape
    if first_size > second_size:
        padd_rows = first_size-second_size
        second_half = np.pad(second_half,((0,0),(0,padd_rows)),"constant",constant_values=(0))
    elif first_size < second_size:
        padd_rows = abs(first_size-second_size)
        first_half = np.pad(first_half,((0,0),(padd_rows,0)),"constant",constant_values=0)
        
    # Flips the first half image across the vertical axis
    flipped = cv2.flip(first_half, 1)

    # overlaps the images and removes every overlaying pixel, sum_mask then being a mask only containing the non overlay parts
    sum_mask = np.array(flipped) ^ np.array(second_half)
    
    # Sets values in mask back to 1 so that sum is representative of each pixel
    sum_mask[sum_mask > 0]= 1

    # Returns the sum of the mask where the flipped image did not overlap the second half.
    return np.sum(sum_mask)

def edge_check(mask):
    
    ## Returns True if a mask touches the edge
    top = np.unique(mask[0,:])
    left = np.unique(mask[:,0])
    right = np.unique(mask[:,-1])
    bottom = np.unique(mask[-1,:])
    if (len(top) > 1) or  (len(left) > 1) or  (len(right) > 1) or  (len(bottom) > 1):
        return True
    else:
        return False

def all_deg(thresh):
    # This function itterates the sym_detect for 180 degrees, returning the average value, aswell as the minimum symmetry
    if edge_check(thresh) == True:
        return np.sum(thresh), np.sum(thresh)
    # List_ stores every symmetry score, so that we can average after all 180 degrees
    a,b,c,d = extract_bboxes(np.array(thresh))
    thresh = thresh[a:b,c:d]
    list_ = []
    
    # sets a high value of last_sym so that when 
    last_sym = 100000000000
    
    # Itterates all 180 degrees
    for i in range(0, 180):
        rotated_image = rotate_bound(thresh, i)
        
        sym = sym_detect(rotated_image)
        list_.append(sym)
        
        # If the sym_detect function returns a lower value than any of the previous values, swap last_sym to be that value,
        # After itterating all degrees, last_sym will be the minimum sym score found for the mask
        if sym < last_sym:
            last_sym = sym

    # returns minimum sym value found and the average sym value found
    return (last_sym, np.average(list_))

def measure_area_perimeter(mask):
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    # Now we have the perimeter image, the sum of all white pixels in it
    perimeter = np.sum(image_perimeter)

    return area, perimeter