








import re
from glob import glob
from scipy.ndimage import zoom


'''
Skript to perform spheroid invasion analysis by using only the flourescence images. 
Folder Structure...
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
def collect_files(inputfolder_path, selector_path="SphInv", selectors_file=["rep", "Fluo"]):
    files_dict = {}
    for subdir, dirs, files in os.walk(inputfolder_path):
        if "Analyzed_Data" in subdir:
            continue
        if not selector_path in subdir:
            continue

        # checking if sphforce folder also exists
        bf_check = sum(["Fluo1" in x for x in dirs] + ["SphInv" in x for x in dirs])
        if "SphInv" in subdir and bf_check>0:  # also cheks if key already exists
            experiment = subdir
            files_dict[experiment] = {}
            print(experiment)

        if "SphInv" not in subdir:  #
            continue


        # finds well identifier, doesnt search for files if well is not in the current subdirectory
        if not "well" in os.path.split(subdir)[1]:
            continue

        well = re.match("(well\d+)", os.path.split(subdir)[1])
        well_id = well.group(1)

        if well and not well_id in files_dict[experiment]:
            files_dict[experiment][well_id] = {}

        file_list = os.listdir(subdir)  # list all files (not directories) in any subdirectory
        file_list_f = [x for x in file_list if x.endswith('.tif') and all(
            selector in x for selector in selectors_file)]  # select all .tif images with rep and fluo in their name

        ## extract position number:
        search_list = [re.search(".*_(pos\d+).*", x) for x in file_list_f]

        # extracts the whole string of the filename except "rep" and all following numbers
        positions = [x.group(1) for x in search_list]
        positions = list(set(positions))  # list all unique strings

        try:  # sorting by position argument for better representation in output txt. file
            positions.sort(key=lambda x: re.search("pos(\d+)", x).group(1))
        except:
            print("no position found")
        # creating a dictinary where keys are the future output filenames and values are all associated input filenames (parts of them) (for mean blending)
        for pos in positions:
            files_dict[experiment][well_id][pos] = [os.path.join(subdir, file) for file in file_list_f if pos in file]

    files_dict = remove_empty_keys(files_dict)
    return files_dict


def analyze_profiles(img_fl, px_um, Mic="Mic5", nuc_size=3):
    # zoom factors:
    bf_mic5 = 4.0965 / 4  # µm/pixel   ## same with bf_mic5
    fl_mic3 = 6.45 / 2.5  # µm/pixel
    zoom_factor = bf_mic5 / fl_mic3  # factor to zoom bf images to same scale as fl images

    # read image
    img_fl = img_fl.astype(float)
    # cutting circular shaped form
    center=np.array(np.shape(img_fl))/2
    fl_crop_mask=circle(int(center[0]),int(center[1]),int(np.min(center)))
    img_fl_crop =np.zeros(np.shape( img_fl))+np.median( img_fl)
    img_fl_crop[fl_crop_mask]= img_fl [fl_crop_mask]



   # img_fl_crop = img_fl[200:-200, 200:-200]

    # segementation of blob and cells
    mask, img1, com_fl = segmentation_gradient_dog(img_fl_crop, nuc_size)
    blob = find_blob_fl(img_fl_crop)






    # distance transform
    dt = distance_transform_edt(~blob)
    dt_inner = distance_transform_edt(blob)
    rad = np.max(dt_inner)  # radius is minimal distance to blob edge from most central points
    print(rad)
    dt = dt - dt_inner + rad
    dens = np.zeros(int(np.max(dt)))

    for i in range(0, int(np.max(dt))):
        dens[i] = np.mean(mask[np.where(
            (dt >= i) & (dt <= i + nuc_size))])  # density is density for list in radial slices around the blob

    # lsq fit
    def sigmoid(p):
        zero_density = np.asarray(np.where(dens == 0))[0]
        fit_to = np.min(np.asarray(np.where(zero_density > rad))[0])
        fit_to = zero_density[fit_to]
        ind = range(int(rad),
                    fit_to - 1)  # only fit to values, outside the spheroid radius ## completely ignores all densities inside the blob
        x_ind = px_um * np.asarray(ind)
        y = p[2] / (1 + np.exp((x_ind - p[0]) / p[1]))
        return np.log(y) - np.log(dens[ind])

    pstart = np.zeros(3)
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
    pstart[0] = rad
    pstart[1] = 10
    pstart[2] = 0.3
    p = least_squares(fun=sigmoid, x0=pstart, bounds=bounds, method="trf",
                      max_nfev=10000, xtol=1e-8, ftol=1e-8, args=())["x"]  # trust region algorithm,

    rad = rad * px_um  # radius in micrometer
    inv_front = p[0] - rad
    print('radius of spheroid blob   = %.1f' % rad, 'um')  # radius
    print('radius of constant cell density    = %.2f' % p[0], 'um')  # relation to inv front.. think about this??
    print('characteristic invasion depth beyond radius of constant density = %.1f' % p[1], 'um')

    return (p, rad, inv_front, blob, mask, dens, dt, img_fl, img1)


def spheroid_analysis_with_fl_core(inputfolder_path, save_images, folder_selector, analyze, magnification, pixel_size,
                                   outputfolder_mode,cdb):
    fl_images = collect_files(inputfolder_path, selector_path="SphInv",
                              selectors_file=["rep", "Fluo"])  # finding all fl-images
    bf_images = collect_files(inputfolder_path, selector_path="SphForce",
                              selectors_file=["rep0000", "modeBF"])  # finding all bf images at t=0

    if outputfolder_mode == "mode1":  # will create outpfolder by replacing a folder "raw data" with "analyzed data and copy the folder structure deeper

        outputfolder_path = re.sub("Raw Data|Raw_data", "Analyzed_Data", inputfolder_path, flags=re.I)
        outputfolder_name_profile = os.path.join(outputfolder_path, "Invasion_Profiles")
        outputfolder_path_invasion = os.path.join(outputfolder_path, "Mean_images")

    if outputfolder_mode == "mode2":
        outputfolder_path = os.path.join(inputfolder_path, "Analyzed_Data")
        outputfolder_path_profile = os.path.join(outputfolder_path, "Invasion_Profiles")
        outputfolder_path_mean = os.path.join(outputfolder_path, "Mean_images")

    if save_images:  # creating output folder if they don't exist already
        createFolder(outputfolder_path_profile)
    if analyze:
        createFolder(outputfolder_path_mean)

    with open(os.path.join(outputfolder_path, 'invasion_analysis.txt'), 'w') as f:  # setting up tab delimited file
        f.write("input_path\twell\tmode\tpos\tradius\td/2\tlambda\tradius of constant cell density" + "\n")

    for key1, value1 in fl_images.items():
        for key2, value2 in fl_images[key1].items():
            for key3, value3 in fl_images[key1][key2].items():
                files_fl = value3

                print("using", files_fl[0])


                im_list = [plt.imread(file) for file in files_fl]
                stack = np.array(im_list,ndmin=3)  # stacking al images, dimensions fixed to 3 , otherwise problems with single images
                mean_img = np.mean(stack, axis=0)  # mean blending of imgaes
                img_16bit = mean_img.astype("uint16")
                meta_info_dict = get_meta_info2(files_fl, well=key2, pos=key3)
                output_filename_path = os.path.join(outputfolder_path_mean,
                                                    ('Mean_Blend_' + meta_info_dict["date"] + "_" +
                                                     meta_info_dict["well"] + "_" +
                                                     meta_info_dict["pos"]) + ".tif")

                if save_images:  # saving image
                    im = Image.fromarray(img_16bit)
                    im.save(output_filename_path)

                if analyze:  # performing analysis on mean blended images
                    res = analyze_profiles(img_fl=img_16bit, px_um=px_um, Mic=meta_info_dict["Mic"],
                                           nuc_size=nuc_size)
                    p, rad, inv_front, blob, mask, dens, dt, img, img1 = res
                    plotting_invasion_profiles1(res, meta_info_dict,
                                                outputfolder_path_profile,
                                                output_filename_path, magnification, pixel_size)
                    #plotting_segementation(res, meta_info_dict, outputfolder_path_profile, output_filename_path)

                    # writing information to text file
                    output_text = [meta_info_dict["file_path"], meta_info_dict["well"], meta_info_dict["mode"],
                                   meta_info_dict["pos"], str(rad.round(2)), str(p[1].round(2)),
                                   str(inv_front.round(2)), str(p[0].round(2))]
                    with open(os.path.join(outputfolder_path, 'invasion_analysis.txt'), 'a') as f:
                        f.write("\t".join(output_text) + "\n")
                    if cdb and analyze and save_images:
                        path_file = os.path.split(output_filename_path)
                        db = clickpoints.DataFile(os.path.join(outputfolder_path_profile, path_file[1][:-4] + ".cdb"),
                                                  "w")
                        db.setImage(output_filename_path, frame=0)
                        db.setMaskType("mask_cells", color="#1fff00", index=1)
                        db.setMaskType("mask_blob", color="#ff0f1b", index=2)
                        cdb_mask = copy.deepcopy(mask) * 1
                        cdb_mask[blob] = 2
                        db.setMask(image=db.getImages()[0], data=np.array(cdb_mask, dtype="uint8"))
                        db.db.close()


# ------------ Data input/output settings section  ------------

inputfolder_path = r'E:\Raw_data\C02_NaGr_002-002-02_CellScr_2019-03-04'  # Insert input folder directory (Folder of Experiment including SphInvasion folder)
save_images = True  # save mean blended images
folder_selector = "SphInvasion"  # a string to limit the analyzed subfolders #SphInvasion
analyze = True  # perform analysis
# general settings of measurement on microscope
magnification = 2.5  # 4
pixel_size = 6.45  # (µm per 1 pixel)     MicInc: 4.0954;  Mic3: 6.45
# outputfolder_mode="mode1" # will create outpfolder by replacing a folder "raw data" with "analyzed data and copy the folder structure deeper
outputfolder_mode = "mode2"  # will create outputfolder in the input directory
px_um = pixel_size / magnification
nuc_size = 4
cdb=True # set weather cdb file with mask and fl images will be shown





if __name__ == '__main__':
    spheroid_analysis_with_fl_core(inputfolder_path, save_images, folder_selector, analyze, magnification, pixel_size,
                                   outputfolder_mode,cdb)












