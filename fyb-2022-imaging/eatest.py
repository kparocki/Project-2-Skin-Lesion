import cv2
import csv
from cv2 import rotate
from cv2 import CV_8S
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import ast


import path
import imutils
from tqdm import tqdm



df_features = pd.read_csv('features/features.csv')

df = pd.read_csv('data/example_ground_truth.csv')

df_out = pd.read_csv("out.csv")
df_out[['center_im_ea']] = None
df_out[["center_mask_ea"]] = None
#for id_ in df['image_id']:
#    img = Image.open(f'data/example_image/{id_}.jpg')
#    img.save(f"data/red95/{id_}.jpg", quality=95)


thresh = cv2.imread(f'data/example_segmentation/ISIC_0001769_segmentation.png',0)


def center(img, cropped = []):
    if len(cropped) != 0:
        img = img[cropped[0]:cropped[1],cropped[2]:cropped[3]]
        
    contours,hierarchy = cv2.findContours(img,2,1)
    cnt = contours

    for i in range (len(cnt)):
        (x,y),radius = cv2.minEnclosingCircle(cnt[i])
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(thresh,center,radius,(0,255,0),2)
    return center, img

"""
for index, line in df.iterrows():
    id_ = line["image_id"]
    thresh = cv2.imread(f'data/example_image/{id_}.jpg',0)
    df_out.at[index,"center_im_ea"] = center(thresh, list(map(int, df_out.iloc[index][["rmin","rmax","cmin","cmax"]])))



for index, line in df.iterrows():
    id_ = line["image_id"]
    thresh = cv2.imread(f'data/example_segmentation/{id_}_segmentation.png', 0)
    print(id_)
    df_out.at[index,"center_im_ea"] = center(thresh, list(map(int, df_out.iloc[index][["rmin","rmax","cmin","cmax"]])))

df_out.to_csv('out.csv')
"""

cv2.imshow("thresh",thresh)
thresh_center, thresh = center(thresh, list(map(int, df_out.iloc[0][["rmin","rmax","cmin","cmax"]])))
def image_rotator(img, center, angle):
    rows, cols = img.shape
    transform_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img,transform_matrix,(cols,rows))
    return rotated

def rotate_bound(image,center ,angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = center

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
    _,th1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    center = [np.average(indices) for indices in np.where(th1 >= 255)]
    first_half = img[:,:int(center[0])]
    second_half = img[:,int(center[0]):-1]
    _, first_size = first_half.shape
    _, second_size = second_half.shape
    if first_size > second_size:
        padd_rows = first_size-second_size
        second_half = np.pad(second_half,((0,0),(padd_rows,0)),"constant",constant_values=(0))
    elif first_size < second_size:
        padd_rows = abs(first_size-second_size)
        first_half = np.pad(first_half,((0,0),(padd_rows,0)),"constant",constant_values=0)
    #cv2.imshow("first_half", first_half)
    cv2.imshow("second_half", second_half)
    flipped = cv2.flip(first_half, 1)
    cv2.imshow("flipped 1", flipped )
    sum_mask = cv2.bitwise_and(flipped,flipped, mask=second_half)
    print(sum_mask, "Hellos")
    cv2.imshow("sum_mask", sum_mask )
    return np.sum(sum_mask)


rotated_image = rotate_bound(thresh,thresh_center, 191)
_,th1 = cv2.threshold(thresh,100,255,cv2.THRESH_BINARY)
center = [np.average(indices) for indices in np.where(th1 >= 255)]
sym_detect(rotated_image)
cv2.imshow("test_sym",rotated_image)
print(int(center[0]), int(center[1]))


"""
last_sym = 100000000000
for i in tqdm(range(0, 359)):
    rotated_image = rotate_bound(thresh,thresh_center, i)
    sym = sym_detect(rotated_image)
    print(sym)
    if sym < last_sym:
        last_sym = sym
        deg = i
    
print(last_sym, deg)
"""
cv2.waitKey(0)

cv2.destroyAllWindows()

quit()

for index, line in df.iterrows():
    id_ = line['image_id'] 
	
    img = cv2.imread(path.images+f'{id_}.jpg')
    mask = cv2.imread(path.segmentation+f'{id_}_segmentation.png',0)

    
    a,b,c,d = map(int, df_out.iloc[index][["rmin","rmax","cmin","cmax"]])
    
    cropped_image = thresh[a:b,c:d]


    contours,hierarchy = cv2.findContours(cropped_image,2,1)
    print (len(contours))
    cnt = contours
    for i in range (len(cnt)):
        (x,y),radius = cv2.minEnclosingCircle(cnt[i])
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(thresh,center,radius,(0,255,0),2)
    df.at[index,"center_notmask_cropped"] = [x,y]

print(df[['center_notmaks_cropped']])
cv2.waitKey(0)

cv2.destroyAllWindows()