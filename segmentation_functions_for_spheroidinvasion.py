
import re
from glob import glob
from scipy.ndimage import zoom
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.filters import threshold_otsu,threshold_li,threshold_local,threshold_sauvola,threshold_niblack
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass,find_objects
from scipy.ndimage.morphology import binary_erosion, binary_dilation, distance_transform_edt,binary_fill_holes
from scipy.optimize import least_squares
from scipy import ndimage
from PIL import Image,ImageFilter
from skimage import restoration
import os
from copy import deepcopy
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects,skeletonize
from scipy import interpolate
from scipy.stats import gaussian_kde
from skimage.filters import sobel, laplace, sobel_h, sobel_v, gaussian
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import regionprops
from scipy.ndimage import label
from scipy.ndimage.measurements import find_objects
try:
    from functions_for_spheroid_invasion import *
except:
    pass


def segmentation_gradient_dog(img,include_mask,nuc_size=4):

    '''

    :param img: grey scale image of spheroid
    :param include_mask: optional mask. ONly detections in this region are considered
    :param nuc_size:
    :return mask_overlapp2: segmentation mask
    :return img1: blurred and normalized image
    :return com: center of mass of the mask
    '''

    include_mask=include_mask.astype(bool)
    img1 = img - np.percentile(img[include_mask], 0.1)  # 1 Percentile
    img1 = img1 / np.percentile(img1[include_mask], 99.9 )  # norm to 99 Percentile
    img1[img1 < 0] = 0.0
    img1[img1 > 1] = 1.0
    img1 =gaussian_filter(img1,0.4)


    img_laplace=laplace(img1)
    img_dog=img-gaussian(img,nuc_size)

    #thresh_laplace = threshold_otsu(img_laplace[200:-200,200:-200][img_laplace[200:-200,200:-200]>0])
    thresh_laplace = threshold_otsu(img_laplace[include_mask][img_laplace[include_mask] > 0])

    mask = img_laplace > thresh_laplace
    mask=np.logical_and(mask,include_mask)
    mask=binary_closing(mask)
    mask=binary_fill_holes(mask)
    mask=remove_small_objects(mask,2)
    mask_dog=img_dog>threshold_otsu(img_dog)
    mask_dog=np.logical_and(mask_dog,include_mask)
    mask_dog=remove_small_objects(mask_dog,10)
    mask_overlapp=mask_dog*2+mask

    mask_overlapp2=copy.deepcopy(mask_overlapp)
    regions=find_objects(label(mask_overlapp>0)[0])
    for r in regions:
        if np.sum((mask_overlapp[r]==1)+(mask_overlapp[r]==3))==0:
            mask_overlapp2[r]=0

    mask_overlapp2=mask_overlapp2 > 0
    mask_overlapp2[~include_mask]=False

    #com=center_of_mass(mask_overlapp2) # fining com from all segmented cells
    #com=center_of_mass(img1) # finding com form whole image intenisty
    #blob=find_blob_fl(img) # finding com from segmented blob only
    #com=center_of_mass(blob) # finding com from segmented blob only
    com = center_of_mass(np.logical_and(img1>0.5,include_mask)) # finding com by mask on high intensity region
    #### could also try com by "unsharp regions" (local entropy?)

    return mask_overlapp2, img1, com

def find_blob_fl(img):
    '''
    function to find blob (body of spheroid) from flourescence labled images
    :param img:
    :return:
    '''

    img2 = img - gaussian_filter(img, 100)
    img2 = img2 - np.percentile(img2, 1)  # 1 Percentile
    img2 = img2 / np.percentile(img2, 99)  # norm to 99 Percentile
    img2 = gaussian_filter(img2, 10)
    img2[img2 < 0] = 0.0
    img2[img2 > 1] = 1.0
    thres = threshold_otsu(img2)

    blob = img2 > thres
    blob = binary_erosion(blob, iterations=5)
    blob = binary_dilation(blob, iterations=10)
    blob = binary_erosion(blob, iterations=5)
    blob = gaussian_filter(blob.astype("float"), 10)
    blob[blob > 0.9] = 1
    blob[blob < 0.9] = 0

    obj, num_features = ndimage.measurements.label(blob)

    # delete all objects except biggest blob

    amax = 0
    for i in range(1, np.max(obj) + 1):
        a = np.sum(obj == i)
        if a > amax:
            big_blob = i
            amax = a
    blob = (obj == big_blob)
    blob = binary_fill_holes(blob)  # fills a disconnected hole in th blob, sometimes occures in the center

    return blob



def find_blob_bf(img,include_mask,pixelsize_bf):
    '''
    function to find blob (body of spheroid) from bright files images
    :param img:
    : param include mask: only used for thresholding
    :return:
    '''
    #normalizing
    include_mask=include_mask.astype(bool)
    img2 = img - np.percentile(img[include_mask], 1)  # 1 Percentile
    img2 = img2 / np.percentile(img2[include_mask], 99)  # norm to 99 Percent
    img2[img2 < 0] = 0.0
    img2[img2 > 1] = 1.0
    img2=1-img2


    ## high sigma costst calculation time
    img3 = gaussian_filter(img2, 3)-gaussian_filter(img2, 60)  ### more usefull would be local thershold with sampling
    mask=img3>threshold_otsu( img3[include_mask])
    mask=binary_fill_holes(mask)
    minsize = pixelsize_bf * 50  # minimal size of spheroid in pixels
    mask=remove_small_objects(mask,min_size=minsize)


    labels, num_features = ndimage.measurements.label(mask)
    intensities=np.array([r.mean_intensity for r in regionprops(labels,intensity_image =img2 )]) # selecting detection with highest intentity on bf image

    #regions=[r.coords for r in regionprops(labels,intensity_image =img3 )]
    if sum(intensities>0.8)>1:
        areas=np.array([r.area for r in regionprops(labels,intensity_image =img2 )]) #if stateente to deal with large nucleus fragements
        max_area=np.max(areas[intensities>0.8])
        select=np.where(max_area==areas)[0]
        print("###### possible spheroid fragmentation ######")
    else:
        select=np.argmax(intensities)
    blob= labels==(select+1) # selecting area with highest mean intensity

    com=center_of_mass(img2,blob)

    #plt.figure()
    #plt.imshow(blob,alpha=0.5)
   # plt.imshow(img, alpha=0.5)

    return blob,com



