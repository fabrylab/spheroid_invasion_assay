
from skimage.draw import circle
import sys
sys.path.insert(0,"/home/user/Software/spheroid_invasion_assay")
from functions_for_spheroid_invasion import *
from segmentation_functions_for_spheroidinvasion import *



def analyze_profiles(img_fl, pixel_size, nuc_size=3):

    center = np.array(np.shape(img_fl )) / 2
    include_coords = circle(int(center[0]), int(center[1]), int(np.min(center)))
    fl_crop_mask = np.zeros(np.shape( img_fl))
    fl_crop_mask[include_coords]= 1
    fl_crop_mask = fl_crop_mask.astype(bool)

    mask, img1, com_fl = segmentation_gradient_dog(img_fl, fl_crop_mask, nuc_size)

    # distances to cell center (in this case we have no spheroid surface)
    y, x = np.indices(img_fl.shape,dtype=float)
    y -= com_fl[0]
    x -= com_fl[1]

    dt = np.linalg.norm(np.array([y,x]),axis=0)

    rad = 0
    dens = np.zeros(int(np.max(dt)))
    for i in range(0, int(np.max(dt))):
        dens[i] = np.mean(mask[np.where((dt >= i) & (dt <= i + nuc_size))])  # density is density for list in radial slices around the blob


    # lsq function
    def sigmoid(p):
        rad_index = int(np.floor(rad)) # index where blob ends
        ind1 = np.where(dens[rad_index:] > 0)[0][0] + rad_index# first non zero index
        zero_index = np.where(dens[ind1:] == 0)[0][0] + ind1 # index of first zero outside of blob
        ind = range(ind1, zero_index)  # al fitted indices
        x_ind = pixel_size * np.asarray(ind) #in µm
        y = p[2] / (1 + np.exp((x_ind - p[0]) / p[1]))
        return np.log(y) - np.log(dens[ind])

    pstart = np.zeros(3)

    bounds = ([rad * pixel_size, 0, 0], [np.inf, np.inf,1])
    pstart[0] = rad * pixel_size
    pstart[1] = 10
    pstart[2] = 0.3
    p = least_squares(fun=sigmoid, x0=pstart, bounds=bounds, method="trf",
                      max_nfev=10000, xtol=1e-8, ftol=1e-8, args=())["x"]  # trust region algorithm,


    rad = rad * pixel_size  # radius in micrometer

    inv_front = p[0] - rad

    return (p, rad, inv_front, mask, dens, dt, img_fl, img1)

if __name__ == "__main__":
    im = plt.imread("test_data/Mean_Blend__20180913-181321_Mic3_pos00_x0_y0_modeFluo1_z0.tif")
    pixel_size = 2
    p, rad, inv_front, mask, dens, dt, img_fl, img1 = analyze_profiles(im, pixel_size, nuc_size=3)


    # ----------general fonts for plots and figures----------
    set_standart_plot_parameters()

    # used for coloring blob outlines ("", ["C8","red"])
    custom_cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#DBDC3E", "red"])


    fig = plt.figure(figsize=(14, 4))
    # plotting the invasion profile
    ax1 = add_plot_inv_profile(fig, rad, pixel_size, dens, p, inv_front)
    # plotting the image and segmentation
    ax2 = add_plot_mask_overlay(fig, mask, im, custom_cmap1, pixel_size=pixel_size,plot_scalebar=True)
