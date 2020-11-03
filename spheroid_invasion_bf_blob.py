

'''
Skript to perform spheroid invasion analysis by using bright field images to find central blob.
Folder Structure...
SphForce and SphInvasion folders are required in inputfolder
within SphForce: well folder
within SphInvasion: BF and Fluo1 folder
within BF and Fluo1 folder: SphForce corresponding well folders
'''


import re
import clickpoints
from scipy.ndimage import zoom
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import distance_transform_edt
from scipy.optimize import least_squares
from PIL import Image
import os

from skimage.measure import regionprops
from skimage.morphology import remove_small_objects,skeletonize
from scipy import interpolate
from skimage.filters import sobel, laplace, sobel_h, sobel_v, gaussian
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import regionprops
from scipy.ndimage import label
from scipy.ndimage.measurements import find_objects
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from skimage.draw import circle
try:
    from functions_for_spheroid_invasion import *
    from segmentation_functions_for_spheroidinvasion import *
except:
    pass



# ------------ Data input/output settings section  ------------

def collect_files(inputfolder_path,selector_path="SphInv",selectors_file=["rep","Fluo"],negative_selectors_file=[]):
    files_dict = {}
    for subdir, dirs, files in os.walk(inputfolder_path):
        if "Analyzed_Data" in subdir:
            continue
        # checking if sphforce folder also exists

        bf_check = sum(["SphForce" in x for x in dirs] + ["SphInv" in x for x in dirs])
        if bf_check == 2 and not subdir in files_dict:  # also cheks if key already exists
            experiment = subdir
            files_dict[experiment] = {}
            print(experiment)

        ## stops iteration if not in correct folder tree
        if not bf_check == 2 and "SphForce" not in subdir and "SphInv" not in subdir:  #
            continue
        ## stops iteration if keyword is missing: diffrentiate between images from Sph Force and SphINv series
        if not all([x in subdir for x in selector_path]):
            continue

        # finds well identifier, doesnt search for files if well is not in the current subdirectory
        if not "well" in os.path.split(subdir)[1]:
            continue

        well = re.match("(well\d+)", os.path.split(subdir)[1])
        print(well)
        well_id = well.group(1)

        if well and not well_id in files_dict[experiment]:
            files_dict[experiment][well_id] = {}

        file_list = os.listdir(subdir)  # list all files (not directories) in any subdirectory
        if len(negative_selectors_file)>0:
            file_list_f = [x for x in file_list if x.endswith('.tif') and all(selector in x for selector in selectors_file) and not any(selector in x for selector in negative_selectors_file)]  # select all .tif images with rep and fluo in their name
        else:
            file_list_f = [x for x in file_list if
                           x.endswith('.tif') and all(selector in x for selector in selectors_file)]
        ## extract position number:

        search_list = [re.search(".*_pos(\d+).*", x) for x in file_list_f]

        # extracts the whole string of the filename except "rep" and all following numbers
        positions = [x.group(1) for x in search_list]
        positions = list(set(positions))  # list all unique strings

        try:  # sorting by position argument for better representation in output txt. file
            positions.sort(key=lambda x: re.search("(\d{1,3})", x).group(1))
        except:
            print("no position found in ", subdir)
        # creating a dictinary where keys are the future output filenames and values are all associated input filenames (parts of them) (for mean blending)
        for pos in positions:
            files_dict[experiment][well_id][pos.zfill(3)] = [os.path.join(subdir,file) for file in file_list_f if "pos"+pos in file]

    files_dict = remove_empty_keys(files_dict)
    return files_dict




def analyze_profiles(img_fl,img_bf, pixelsize_bf,pixelsize_fl_mic3,pixelsize_fl_mic5,Mic="Mic5",nuc_size=3):



    #zoom factors:
    bf_mic5=pixelsize_bf
    fl_mic3=pixelsize_fl_mic3
    fl_mic5=pixelsize_fl_mic5
    zoom_factor=bf_mic5/ fl_mic3 # factor to zoom bf images to same scale as fl images

    # read image
    img_fl = img_fl.astype(float)
    img_bf = img_bf.astype(float)

    ## correct zoom diffrence by zoming out bf image
    if Mic == "Mic3":
        img_bf_zoom = zoom(img_bf, zoom_factor)
        center = np.array(np.shape(img_bf_zoom )) / 2
        bf_crop_mask = circle(int(center[0]), int(center[1]), int(np.min(center)))
        IQR = np.percentile(  img_bf_zoom , 75) - np.percentile(  img_bf_zoom , 25)
        img_bf_crop = np.random.normal(loc=np.median(  img_bf_zoom ), scale=IQR / 13, size=np.shape( img_bf_zoom ))
        img_bf_crop[bf_crop_mask] = img_bf_zoom[bf_crop_mask]
        px_um=pixelsize_fl_mic3



    # cutting circular shaped form
    else:
        center = np.array(np.shape(img_bf)) / 2
        bf_crop_mask = circle(int(center[0]), int(center[1]), int(np.min(center)))
        IQR = np.percentile(img_bf, 75) - np.percentile(img_bf, 25)
        img_bf_crop = np.random.normal(loc=np.median(img_bf), scale=IQR / 13, size=np.shape(img_bf))
        img_bf_crop[bf_crop_mask] = img_bf[bf_crop_mask]
        px_um=pixelsize_fl_mic5


    center = np.array(np.shape(img_fl)) / 2
    fl_crop_mask=np.zeros(img_fl.shape)
    fl_crop_mask[circle(int(center[0]), int(center[1]), int(np.min(center)))] = 1
    fl_crop_mask = fl_crop_mask.astype(bool)

    #IQR=np.percentile(img_fl,75)-np.percentile(img_fl,25)
    #img_fl_crop=np.random.normal(loc=np.median(img_fl),scale=IQR/13,size=np.shape(img_fl))
    #img_fl_crop[fl_crop_mask] = img_fl[fl_crop_mask]
    #img_fl_crop=img_fl[200:-200,200:-200]



    # segementation of blob and cells
    mask, img1, com_fl =  segmentation_gradient_dog(img_fl,fl_crop_mask,nuc_size)
    blob_pre, com_bf = find_blob_bf(img_bf_crop,pixelsize_bf)


    # correcting for blob shift
    ## finds "box around blob
    rectangle=find_objects(blob_pre,1)
    lengths=[rectangle[0][0].stop-rectangle[0][0].start,rectangle[0][1].stop-rectangle[0][1].start]
    center_of_blob= [rectangle[0][0].start+lengths[0]/2,rectangle[0][1].start+lengths[1]/2]  #  center of rectangle object in coordinates of full mask
    displacement_blob_to_com_bf= [center_of_blob[0]-com_bf[0],center_of_blob[1]-com_bf[1]]
    # set in around com of fl image:
    corrected_blob=np.zeros(np.shape(img1))

    new_cords1=[int(com_fl[0]-lengths[0]/2 + displacement_blob_to_com_bf[0]),
                int(com_fl[0]+lengths[0]/2 + displacement_blob_to_com_bf[0]),
                int(com_fl[1]-lengths[1]/2 + displacement_blob_to_com_bf[1]),
                int(com_fl[1]+lengths[1]/2 + displacement_blob_to_com_bf[1])
                ]
    corrected_blob[new_cords1[0]: new_cords1[1], new_cords1[2]:new_cords1[3]]=blob_pre[rectangle[0][0],rectangle[0][1]]
    blob= corrected_blob>0


    # distance transform
    dt = distance_transform_edt(~blob)
    dt_inner = distance_transform_edt(blob)
    rad = np.max(dt_inner)       # radius is minimal distance to blob edge from most central points
    print(rad)
    dt = dt - dt_inner + rad
    dens = np.zeros(int(np.max(dt)))

    for i in range(0, int(np.max(dt))):
        dens[i] = np.mean(mask[np.where((dt >= i) & (dt <= i + nuc_size))])  # density is density for list in radial slices around the blob


    # lsq function
    def sigmoid(p):
        rad_index=int(np.floor(rad)) # index where blob ends
        ind1=np.where(dens[rad_index:] > 0)[0][0]+rad_index# first non zero index
        zero_index=np.where(dens[ind1:] == 0)[0][0]+ind1 # index of first zerop putside of blob
        ind = range(ind1, zero_index)  # al fitted indices
        x_ind = px_um * np.asarray(ind) #in µm
        y = p[2] / (1 + np.exp((x_ind - p[0]) / p[1]))
        return np.log(y) - np.log(dens[ind])

    pstart = np.zeros(3)

    bounds = ([rad*px_um, 0, 0], [np.inf, np.inf,1])
    pstart[0] = rad*px_um
    pstart[1] = 10
    pstart[2] = 0.3
    p = least_squares(fun=sigmoid, x0=pstart, bounds=bounds, method="trf",
                      max_nfev=10000, xtol=1e-8, ftol=1e-8, args=())["x"]  # trust region algorithm,


    rad = rad * px_um  # radius in micrometer


    inv_front = p[0] - rad
    print('radius of spheroid blob   = %.1f' % rad, 'um')  # radius
    print('radius of constant cell density    = %.2f' % p[0], 'um')  # relation to inv front.. think about this??
    print('characteristic invasion depth beyond radius of constant density = %.1f' % p[1], 'um')

    return (p, rad, inv_front,  blob, mask, dens,dt,img_fl,img1,px_um)




def spheroid_analysis_with_bf_core(inputfolder_path, save_images,use_existing_mean_images, analyze,
                                   pixelsize_bf,pixelsize_fl_mic3,pixelsize_fl_mic5,
                                   outputfolder_mode,cdb):


    fl_images = collect_files(inputfolder_path, selector_path=["SphInv","Fluo1"],
                              selectors_file=["rep","z0"],negative_selectors_file=[])  # finding all fl-images
    bf_images = collect_files(inputfolder_path, selector_path=["SphForce"],
                              selectors_file=["rep0000", "modeBF"],negative_selectors_file=["above", "below"])  # finding all bf images at t=0


    if outputfolder_mode == "mode1":  # will create outpfolder by replacing a folder "raw data" with "analyzed data and copy the folder structure deeper
        outputfolder_path = re.sub("Raw Data|Raw_data", "Analyzed_Data", inputfolder_path, flags=re.I)
        outputfolder_path_profile = os.path.join(outputfolder_path, "Invasion_Profiles")
        outputfolder_path_mean = os.path.join(outputfolder_path, "Mean_images")

    if outputfolder_mode == "mode2":
        outputfolder_path = os.path.join(inputfolder_path, "Analyzed_Data")
        outputfolder_path_profile = os.path.join(outputfolder_path, "Invasion_Profiles")
        outputfolder_path_mean = os.path.join(outputfolder_path, "Mean_images")

    if save_images:  # creating output folder if they don't exist already
        createFolder(outputfolder_path_profile)
    if analyze:
        createFolder(outputfolder_path_mean)

    with open(os.path.join(outputfolder_path, 'invasion_analysis.txt'), 'w') as f:  # setting up tab delimited file
        f.write("input_path\twell\tmode\tpos\tradius\td/2\tlambda\tradius_of_constant_cell_density\tpixelsize_fl_image" + "\n")

    for key1, value1 in fl_images.items():
        for key2, value2 in fl_images[key1].items():
            for key3, value3 in fl_images[key1][key2].items():
                files_fl = value3
                try:
                    file_bf = bf_images[key1][key2][key3][0]  ## check if this looses me something
                except:
                    print("\n--------------------------------------->no complementary file to:")
                    print(files_fl[0])
                    print("!!! Position skipped !!!")
                    continue
                print("using", files_fl[0], file_bf)


                if not ("pos014" in files_fl[0]):
                    continue


                meta_info_dict = get_meta_info2(files_fl, well=key2, pos=key3)
                # default output file for mean blended images
                output_filename_path = os.path.join(outputfolder_path_mean,
                                                    ('Mean_Blend_' + meta_info_dict["date"] + "_" +
                                                     meta_info_dict["well"] + "_" +
                                                     meta_info_dict["pos"]) + ".tif")
                #reading fl images
                if os.path.exists(output_filename_path) and use_existing_mean_images: # use existing mean blended images
                   img_16bit =np.array(Image.open(output_filename_path))
                else:    # generate new mean blended images
                    im_list = [np.array(Image.open(file)) for file in files_fl]
                    stack = np.array(im_list,
                                     ndmin=3)  # stacking al images, dimensions fixed to 3 , otherwise problems with single images
                    mean_img = np.mean(stack, axis=0)  # mean blending of imgaes
                    img_16bit = mean_img.astype("uint16")

                    if save_images:  # saving mean blended image
                        im = Image.fromarray(img_16bit)
                        im.save(output_filename_path)

                # read bright field image
                img_bf = plt.imread(file_bf)



                if analyze:  # performing analysis on mean blended images
                    # try:
                    res = analyze_profiles(img_fl=img_16bit, img_bf=img_bf,
                                   pixelsize_bf=pixelsize_bf,pixelsize_fl_mic3=pixelsize_fl_mic3,
                                           pixelsize_fl_mic5=pixelsize_fl_mic5, Mic=meta_info_dict["Mic"],nuc_size=nuc_size)
                    p, rad, inv_front, blob, mask, dens, dt, img, img1,px_um = res
                    plotting_invasion_profiles1(res, meta_info_dict,
                                                outputfolder_path_profile,
                                                output_filename_path)


                    #if "well1" in files_fl[0] or "well2" in files_fl[0] or "015" in files_fl[0] or "016" in files_fl[0] or "017" in files_fl[0] or "018" in files_fl[0]:
                        #pass


                    #plotting_segementation(res, meta_info_dict, outputfolder_path_profile, output_filename_path)

                    # writing information to text file
                    output_text = [meta_info_dict["file_path"], meta_info_dict["well"], meta_info_dict["mode"],
                                   meta_info_dict["pos"], str(rad.round(2)),str(inv_front.round(2)), str(p[1].round(2)),
                                    str(p[0].round(2)),str(px_um)]
                    with open(os.path.join(outputfolder_path, 'invasion_analysis.txt'), 'a') as f:
                        f.write("\t".join(output_text) + "\n")
                if cdb and analyze and save_images:
                    path_file = os.path.split(output_filename_path)
                    db = clickpoints.DataFile(os.path.join(outputfolder_path_profile, path_file[1][:-4] + ".cdb"), "w")
                    db.setImage(output_filename_path, frame=0)
                    db.setMaskType("mask_cells",color="#1fff00",index=1)
                    db.setMaskType("mask_blob", color="#ff0f1b", index=2)
                    cdb_mask=copy.deepcopy(mask)*1
                    cdb_mask[blob]=2
                    db.setMask(image=db.getImages()[0],data=np.array(cdb_mask,dtype="uint8"))
                    db.db.close()



inputfolder_path = r'E:\Raw_data\C02_NaGr_002-005-04_CellScr_2020-01-20'    # Insert input folder directory (Folder of Experiment, where SphInvasion and SphForce can be found)
save_images = True     #True save mean blended images
analyze = True         # perform analysis

# use existing mean blended images
#  You still need the same data structure and bright field images
use_existing_mean_images = True       #True, if mean_blend images already exist

# general settings of measurement on microscope
pixelsize_bf= 4.0965 / 4  # µm/pixel / magnification   ## same with bf_mic5
pixelsize_fl_mic3= 6.45 / 2.5  # µm/pixel / magnification
pixelsize_fl_mic5= 4.0965 / 4  # µm/pixel / magnification   ## same with bf_mic5
#outputfolder_mode="mode1" # will create outpfolder by replacing a folder "raw data" with "analyzed data and copy the folder structure deeper
# mode2 will create outputfolder in the input directory
outputfolder_mode="mode1"
nuc_size = 4 # maximal expected cell size in pixel
cdb=True  #True set if cdb file with flourescent image and mask should be created

if __name__ == '__main__':
    spheroid_analysis_with_bf_core(inputfolder_path, save_images, use_existing_mean_images, analyze,
                                   pixelsize_bf,pixelsize_fl_mic3,pixelsize_fl_mic5,
                                   outputfolder_mode,cdb=True)












