
import re
import warnings
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






# Method to create folder
def normalize(img):
    img1 = img - np.percentile(img, 0.1)  # 1 Percentile
    img1 = img1 / np.percentile(img1, 99.9)  # norm to 99 Percentile
    img1[img1 < 0] = 0.0
    img1[img1 > 1] = 1.0
    return img1
def convert_to_unit8(img):
    img1 = img - np.min(img)
    img1 = img1 /np.max(img1)  # norm to 99 Percentile
    img1*=255
    img1=img.astype("uint8")
    return img1

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def find_edge(image,width):
    image_edge_x = np.zeros_like(image)

    for i in range (image_edge_x.shape[0]):
        if sum(np.where(image[i,:])[0])==0:
            continue
        a=min(np.where(image[i,:])[0])
        b=max(np.where(image[i,:])[0])
        image_edge_x[i,[range(a,a+width),range(b-width+1,b+1)]] = 1

    image_trans = np.transpose(image)
    image_edge_y = np.zeros_like(image_trans)

    for i in range(image_edge_y.shape[0]):
        if sum(np.where(image_trans[i, :])[0])==0:
            continue
        a = min(np.where(image_trans[i, :])[0])
        b = max(np.where(image_trans[i, :])[0])
        image_edge_y[i,[range(a,a+width),range(b-width+1,b+1)]] = 1
    image_edge=image_edge_x + np.transpose(image_edge_y)

    return image_edge

def get_group(s=None,group_number=1):
    try:
        return s.group(group_number)
    except:
        return ""

def get_meta_info(filename, im_list, inputfolder_path,subdir):
    info = ["Mic", "pos", "rep", "well","x", "y", "z", "date", "file_path_info","path","file_path", "mode"]                     # list of all information keywords
    meta_info_dict = dict((key, "") for key in info)                                          # extraction info from file name

    for key in meta_info_dict:
        meta_info_dict[key]=get_group(re.search(".*_" + key + "(\d+).*", filename),1)

    meta_info_dict["date"]=get_group(re.search("(\d{8}-\d{6})", filename),1)

    # information about well, by searching the first sub directory that matches well
    subdir_copy=deepcopy(subdir)
    subdir_part_name=""
    while not re.match("well\d+",subdir_part_name):
        subdir_part_name =os.path.split( subdir_copy)[1]
        subdir_copy=os.path.split( subdir_copy)[0]

    meta_info_dict["well"] = subdir_part_name
    meta_info_dict["mode"]=get_group(re.search("mode(.{0,6}\d+)_", filename),1)
    meta_info_dict["rep"] = len(im_list)                                                      # number of repetitions, equall to number of images used for mean blending
    meta_info_dict["file_path_info"] = os.path.split(inputfolder_path)[1] + re.search(        # extracting path to input files
        re.escape(inputfolder_path) + "(.*)", subdir).group(1)
    meta_info_dict["path"]=subdir                                                        # getting full path to file
    return meta_info_dict




def get_meta_info2(file_list,well="",pos="",experiment=""):
    info = ["Mic", "pos", "rep", "well","x", "y", "z", "date","file_path", "mode"]

    path,filename=os.path.split(file_list[0]) # just using one exaample filenmae

    # list of all information keywords
    meta_info_dict = dict((key, "") for key in info)                                          # extraction info from file name

    for key in meta_info_dict:
        meta_info_dict[key]=get_group(re.search(".*_" + key + "(\d+).*", filename),1)

    meta_info_dict["date"]=get_group(re.search("(\d{8}-\d{6})", filename),1)
    meta_info_dict["well"] =well
    meta_info_dict["pos"]=pos
    meta_info_dict["mode"]=get_group(re.search("mode(.{0,6}\d+)_", filename),1)
    meta_info_dict["rep"] = len(file_list)                                                      # number of repetitions, equall to number of images used for mean blending
    meta_info_dict["file_path"] = path
    meta_info_dict["Mic"] =get_group(re.search("(Mic\d+)_", filename),1)

    return meta_info_dict


def collaps_dict(dict):  ### maybe better solution is pandas??
    '''

    :param dict:
    :return:list of tuples
    This function collapses a three layers deep dictionay int o a list of tuples. each tuple has the structure:
    key1,key2,key3, values (values as a seperate list)
    '''
    dict_as_list = []
    for key1, value1 in dict.items():
        for key2, value2 in dict[key1].items():
            for key3, value3 in dict[key1][key2].items():
                dict_as_list.append((key1, key2, key3, value3))
    return dict_as_list



def plotting_segementation(res, meta_info, outputfolder_path, filename):
    p, rad, inv_front, blob, mask, dens,dt, img, img1 = res
    plt.ioff()
    plt.figure()
    plt.imshow(blob,alpha=0.5)
    plt.imshow(mask,alpha=0.5)
    plt.pause(0.05)
    plt.savefig(os.path.join(outputfolder_path,"overlay.png"))
    plt.close()

def remove_empty_keys(dict):
    '''

    :param dict:
    :return: dict
    this function removes empty keys in a three layers deep dictionary.
    '''

    dict_copy = deepcopy(dict)
    for key1, values1 in dict.items():
        for key2, values2 in values1.items():
            if len(values2) == 0:
                dict_copy[key1].pop(key2, None)
    dict_copy2 = deepcopy(dict_copy)
    for key1, values1 in dict_copy.items():
        if len(values1) == 0:
            dict_copy2.pop(key1, None)
    return dict_copy2

def crop_image(img,factor_x,factor_y):
    '''

    :param img: numpy array
    :param factor: fraction of image to be cut at each side. Only use values from 0 to 0.5
    :return:
    '''

    shape=np.shape(img)
    img_cut=img[int(shape[0]*factor_y):int(-shape[0]*factor_y),int(shape[1]*factor_x):int(-shape[1]*factor_x)]
    return img_cut



def add_plot_inv_profile(fig, rad, px_um, dens, p, inv_front):

    # plotting the invasion profile

    ax = fig.add_axes([0.075, 0.2, 0.25, 0.6])

    #  ax1.subplot(grid[14:86, 0:2])
    max_ind = int(1000 / px_um) if int(1000 / px_um) < len(dens) else len(dens)
    x = np.array(range(int(np.ceil(rad / px_um)), max_ind)) * px_um
    x_fit = np.array(range(int(np.ceil(rad / px_um)), max_ind)) * px_um

    ax.semilogy(x, dens[int(np.ceil(rad / px_um)):max_ind], linewidth=2, color='C0')  # show all data
    ax.semilogy(x_fit, p[2] / (1 + np.exp((x_fit - p[0]) / p[1])), linewidth=2,
                 color='C1')  # show fit, but only for x>rad
    ax.fill_between([0, rad], [0, 0], [1, 1], facecolor=(0.729, 0.729, 0.729), edgecolor="none",
                     linewidth=0)  # fills area in plot represetning the blob
    ax.fill_between([rad, p[0]], [0, 0], [1, 1], facecolor=("#C3E994"), edgecolor="none",
                     linewidth=0)  # fills area in plot  representing distance to lambda point

    ax.set_ylim([1e-4, 1])
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    ax.text(610, 0.1, '$\lambda$=%.0f $\mu$m' % p[1], fontsize=16)

    ax.text(610, 0.3, 'd/2=%.0f $\mu$m' % inv_front, fontsize=16)
    ax.set_xlabel('distance from center ($\mu$m)')
    ax.set_ylabel('cell density')

    return ax

def add_plot_mask_overlay(fig, mask, img, custom_cmap1, blob=None, factor_x=0.1, factor_y=0.1, pixel_size=None, plot_scalebar=False):
    # plotting the image and segmentation

    if not blob is None:
        mask = mask | blob
        mask = mask * 1  # adding (with logical "or") operation mask and blob
        mask[blob] = 2
    mask = mask.astype(int)
    overlay_mask = deepcopy(mask)
    overlay_mask = np.array(overlay_mask, dtype="float64")  # values are nan (bakcground), 1 (single) cells, 2 (blob)
    overlay_mask[np.where(overlay_mask == 0)] = np.nan  # create matrix of nan-values (these are transparent in plots)

    #  overlay_mask[mask] = 1.  # fill in everything except spheroid region (invert by deleting the "~"-sign)

    ax = fig.add_axes([0.375, 0.2, 0.25, 0.6])
    ax.imshow(crop_image(img, factor_x, factor_y), cmap='gray')  # display image in greyscale
    # display overlay, partially transparent
    ax.imshow(crop_image(overlay_mask, factor_x, factor_y), cmap=custom_cmap1, alpha=0.5)

    if plot_scalebar:
        ax.arrow(img.shape[1] - (img.shape[1] * 2 * factor_x) - 300,
                  img.shape[1] - (img.shape[1] * 2 * factor_x) - 400, 500 / pixel_size, 0, color="white",
                  length_includes_head=False, head_width=0., width=10)
        ax.text(img.shape[1] - (img.shape[1] * 2 * factor_x) - 300,
                img.shape[0] - (img.shape[0] * 2 * factor_y) - 60, '500 $\mu$m', color='white', fontsize=12)
    ax.axis('off')
    return ax

def plot_r_d(fig, img1, blob, dt, p, px_um, custom_cmap2, custom_cmap3, factor_x=0.1, factor_y=0.1, plot_scalebar=True):
    # plotting the filtered image and center and d/2 outlines

    # plotting d/2 line:
    ring=np.zeros(np.shape(blob))
    ring[np.where(np.isclose(dt, p[0]/px_um, rtol=0.01))]=1 # finds interval where distance is close to d/2 distances
    skelet_ring=skeletonize(ring)  # reduces to exactely one pixel
    skelet_ring=binary_dilation(skelet_ring,iterations=3)  # expands to diameter of exactely 6 pixels

    d_half_ring=np.zeros(np.shape(blob)) + np.nan
    d_half_ring[np.where(skelet_ring>0)]=1.

    # plotting blob outlines
    blob_edge = find_edge(image=blob, width=4)  # function to find edges of binary image
    blob_edge_overlay = np.zeros_like(blob_edge) + np.nan
    blob_edge_overlay[blob_edge] = 1.

    ax = fig.add_axes([0.675, 0.2, 0.25, 0.6])
    ax.imshow(crop_image(img1, factor_x, factor_y), cmap='gray')  # display image in greyscale
    ax.imshow(crop_image(blob_edge_overlay,factor_x,factor_y), cmap=custom_cmap2, alpha=0.5) # display the edge of the detected blob
    ax.imshow(crop_image(d_half_ring,factor_x,factor_y), cmap=custom_cmap3, alpha=0.8)

#    ax3.arrow([img1.shape[1]-(img1.shape[1]*2*factor_x) - 250, img1.shape[1]-(img1.shape[1]*2*factor_x) - 250 + 500 / px_um], [img_norm.shape[0] - 30,img_norm.shape[0] - 30], lw=4,
 #            c='white')

    ax.text(img1.shape[1]-(img1.shape[1]*2*factor_x) - 431, img1.shape[0]-(img1.shape[0]*2*factor_y) - 755, 'spheroid edge', color="red", fontsize = 14)
    ax.text(img1.shape[1]-(img1.shape[1]*2*factor_x) - 370, img1.shape[0]-(img1.shape[0]*2*factor_y) - 680, 'd/2 distance', color="#C3E994", fontsize = 14)

    if plot_scalebar:
        ax.arrow(img1.shape[1] - (img1.shape[1] * 2 * factor_x) - 210,
                  img1.shape[1] - (img1.shape[1] * 2 * factor_x) - 340, 500 / px_um, 0, color="white",
                  length_includes_head=False, head_width=0., width=10)
        ax.text(img1.shape[1] - (img1.shape[1] * 2 * factor_x) - 210,
                img1.shape[0] - (img1.shape[0] * 2 * factor_y) - 20, '500 $\mu$m', color='white', fontsize=12)
    ax.axis('off')
    return ax

def add_info_box(meta_info):

    # writing info box in plot
    meta_info_sel = dict((k, meta_info[k]) for k in
                         ("rep", "pos", "well", "Mic", "file_path"))  # possible selection to be written in info box
    if len(meta_info_sel["file_path"]) > 30:  # inserting "\n" in path because it could be to long to fit in the grafik
        meta_info_sel["file_path"] = meta_info_sel["file_path"][:30] + meta_info_sel["file_path"][30:].replace("\\",
                                                                                                               "\\\n",
                                                                                                               1)
        meta_info_sel["file_path"] = meta_info_sel["file_path"][:30] + meta_info_sel["file_path"][30:].replace("/",
                                                                                                               "\n", 1)

    list_length = len(list(meta_info_sel.items()))  # setting index for splitting metha info in half
    infostr1 = "\n".join(['%s: %s' % (key, value) for (key, value) in list(meta_info_sel.items())[0:list_length // 2]])
    plt.gcf().text(0.39, 0.18, infostr1, fontsize=10, verticalalignment='top')  # print first half of meta_info to plo
    infostr2 = "\n".join(
        ['%s: %s' % (key, value) for (key, value) in list(meta_info_sel.items())[list_length // 2:list_length]])
    plt.gcf().text(0.46, 0.18, infostr2, fontsize=10, verticalalignment='top')  # print second half of meta info to plot
    # plt.suptitle(meta_info_dict["file_path_info"])

def set_standart_plot_parameters():

    font = {'family': 'sans-serif',
            'sans-serif': ['Arial'],
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', titlesize=18)


def plotting_invasion_profiles1(res, meta_info, outputfolder_path, filename):


    p, rad, inv_front, blob, mask, dens, dt, img, img1, px_um=res


    # ----------general fonts for plots and figures----------
    set_standart_plot_parameters()


    # used for coloring blob outlines ("", ["C8","red"])
    custom_cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#DBDC3E", "red"])
    # used for coloring mask of cells blob "", ["red", "white"]
    custom_cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "white"])
    # used for coloring mask of cells blob "", ["red", "white"]
    custom_cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#C3E994", "white"])
    # draw and plot


    plt.ioff()
    fig = plt.figure(figsize=(14, 4))
    # plotting the invasion profile
    ax1 = add_plot_inv_profile(fig, rad, px_um, dens, p, inv_front)
    # plotting the image and segmentation
    ax2 = add_plot_mask_overlay(fig, mask, img, custom_cmap1, blob=blob)
    # plotting the filtered image and center and d/2 outlines
    ax3 = plot_r_d(fig, img1, blob, dt, p, px_um, custom_cmap2, custom_cmap3)

    add_info_box(meta_info)

    path_file = os.path.split(filename)
    print(outputfolder_path,('Inv_profile_' + path_file[1][:-4] + '.png'))
    fig.savefig(os.path.join(outputfolder_path,('Inv_profile_' + path_file[1][:-4] + '.png')), dpi=600)#
    fig.close()









def plotting_invasion_profiles2(res, meta_info, outputfolder_path, filename, magnification, pixel_size):
    p, rad, inv_front, blob, mask, dens, dt, img, img1 = res
    # ----------general fonts for plots and figures----------
    font = {'family': 'sans-serif',
            'sans-serif': ['Arial'],
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=10)
    plt.rc('axes', titlesize=18)
    px_um = pixel_size / magnification
    nuc_size = 3

    # custom color map, this defines
    # the colors used for showing and overlaying masks, pass this to "cmap" argument in plt.imshow()

    custom_cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#DBDC3E",
                                                                            "red"])  # used for coloring blob outlines ("", ["C8","red"])
    custom_cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red",
                                                                            "white"])  # used for coloring mask of cells blob "", ["red", "white"]
    custom_cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#C3E994",
                                                                            "white"])  # used for coloring mask of cells blob "", ["red", "white"]
    # draw and plot

    # plotting blob outlines
    blob_edge = find_edge(image=blob, width=4)  # function to find edges of binary image
    blob_edge_overlay = np.zeros_like(blob_edge) + np.nan
    blob_edge_overlay[blob_edge] = 1.

    plt.ioff()
    f, ax = plt.subplots(1, 1, figsize=(14, 4))
    grid = plt.GridSpec(100, 6, hspace=0.1, wspace=0.2)
    plt.subplot(grid[14:86, 0:2])

    #################ohhh no an error with lengths
    x = np.array(range(int(rad / px_um), int(1000 / px_um))) * px_um
    x_fit = np.array(range(int(rad / px_um), int(1000 / px_um))) * px_um

    plt.semilogy(x, dens[int(rad / px_um):int(1000 / px_um)], linewidth=2, color='C0')  # show all data

    plt.semilogy(x_fit, p[2] / (1 + np.exp((x_fit - p[0]) / p[1])), linewidth=2,
                 color='C1')  # show fit, but only for x>rad

    plt.fill_between([0, rad], [0, 0], [1, 1], facecolor=(0.729, 0.729, 0.729), edgecolor="none",
                     linewidth=0)  # fills area in plot represetning the blob
    plt.fill_between([rad, p[0]], [0, 0], [1, 1], facecolor=("#C3E994"), edgecolor="none",
                     linewidth=0)  # fills area in plot  representing distance to lambda point

    plt.ylim([1e-4, 1])
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    plt.text(620, 0.1, '$\lambda$=%.0f $\mu$m' % p[1], fontsize=16)

    plt.text(620, 0.3, 'd/2=%.0f $\mu$m' % inv_front, fontsize=16)
    plt.xlabel('distance from center ($\mu$m)')
    plt.ylabel('cell density')

    mask = mask | blob
    mask = mask * 1  # adding (with logical "or") operation mask and blob
    mask[blob] = 2
    overlay_mask = deepcopy(mask)
    overlay_mask = np.array(overlay_mask, dtype="float64")  # values are nan (bakcground), 1 (single) cells, 2 (blob)
    overlay_mask[np.where(overlay_mask == 0)] = np.nan  # create matrix of nan-values (these are transparent in plots)

    #  overlay_mask[mask] = 1.  # fill in everything except spheroid region (invert by deleting the "~"-sign)
    plt.subplot(grid[0:, 2:4])
    plt.imshow(img1, cmap='gray')  # display image in greyscale
    plt.imshow(overlay_mask, cmap=custom_cmap1, alpha=0.5)  # display overlay, partially transparent
    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    plt.subplot(grid[0:, 4:])
    plt.imshow(img1, cmap='gray')  # display image in greyscale
    plt.imshow(blob_edge_overlay, cmap=custom_cmap2, alpha=0.5)  # display the edge of the detected blob


    plt.plot([img1.shape[1] - 250, img1.shape[1] - 250 + 500 / px_um], [img1.shape[0] - 30, img1.shape[0] - 30], lw=4,
             c='white')
    plt.text(img1.shape[1] - 230, img1.shape[0] - 50, '500 $\mu$m', color='white', fontsize=12)

    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # writing info box in plot
    meta_info_sel = dict((k, meta_info[k]) for k in
                         ("rep", "pos", "well", "Mic", "file_path"))  # possible selection to be written in info box
    if len(meta_info_sel["file_path"]) > 30:  # inserting "\n" in path because it could be to long to fit in the grafik
        meta_info_sel["file_path"] = meta_info_sel["file_path"][:30] + meta_info_sel["file_path"][30:].replace("\\",
                                                                                                               "\\\n",
                                                                                                               1)
        meta_info_sel["file_path"] = meta_info_sel["file_path"][:30] + meta_info_sel["file_path"][30:].replace("/",
                                                                                                               "\n", 1)

    list_length = len(list(meta_info_sel.items()))  # setting index for splitting metha info in half
    infostr1 = "\n".join(['%s: %s' % (key, value) for (key, value) in list(meta_info_sel.items())[0:list_length // 2]])
    plt.gcf().text(0.39, 0.2, infostr1, fontsize=10, verticalalignment='top')  # print first half of meta_info to plo
    infostr2 = "\n".join(
        ['%s: %s' % (key, value) for (key, value) in list(meta_info_sel.items())[list_length // 2:list_length]])
    plt.gcf().text(0.46, 0.2, infostr2, fontsize=10, verticalalignment='top')  # print second half of meta info to plot
    # plt.suptitle(meta_info_dict["file_path_info"])
    path_file = os.path.split(filename)
    print(outputfolder_path, ('Inv_profile_' + path_file[1][:-4] + '.png'))
    f.savefig(os.path.join(outputfolder_path, ('Inv_profile_' + path_file[1][:-4] + '.png')), dpi=600)



