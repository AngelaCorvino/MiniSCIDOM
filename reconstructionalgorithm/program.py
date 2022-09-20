# pylint: disable=invalid-name, redefined-outer-name
"""Main module where all the classes are called"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from image_loader import Image_loader
from camera_picture_viewer import Camera_picture_viewer
from reconstructor import Reconstructor
from tomograpic_viewer import Tomograpic_viewer
from correction_matrix_producer import Correction_matrix_producer
from calibrationfactor import calibrationfactor
from letcorrection3D import letcorrection
from getprofile import Get_profile

#######################For-real-camera-pictures#######################################
# Directory measured data


directory = "pictures/2021-09-01/"
# directory='pictures/spatialresolution/'
picture_name = "29.png"
picture2_name = "background60s.png"  # background

# Define path for measured pictures
path_pic_front = directory + picture_name
path_pic_top = directory + picture_name
path_pic_120 = directory + picture_name
path_pic_240 = directory + picture_name

path_pic2_front = directory + picture2_name
path_pic2_top = directory + picture2_name
path_pic2_120 = directory + picture2_name
path_pic2_240 = directory + picture2_name

shape_front = (170, 130)  # you can not reconstruct somethin bigger than 190
shape_side = (140, 130)


s = 0.073825  # spatialresolution proton measurements at oncoray
deltaz = 140  # proton measurements
rot_angle = 89.9

x_0 = int(0)
y_0 = int(-5)

y1_front = y_0 + 451
y2_front = y_0 + 621
x1_front = x_0 + 458
x2_front = x_0 + 588

y1_top = y_0 + 655
y2_top = y_0 + 795
x1_top = x_0 + 459
x2_top = x_0 + 589

y1_120 = y_0 + 650
y2_120 = y_0 + 790
x1_120 = x_0 + 267
x2_120 = x_0 + 397

y1_240 = y_0 + 650
y2_240 = y_0 + 790
x1_240 = x_0 + 649
x2_240 = x_0 + 779
###shift
shift_image_front = [0, 0]
shift_image_top = [0, 0]
shift_image_120 = [8, 0]
shift_image_240 = [-8, 0]


mask_border_front = [0, 0, 0, 0]
mask_border_top = [0, 0, 0, 0]
mask_border_120 = [10, 4, 0, 0]
mask_border_240 = [4, 10, 0, 0]


##Load images
color_channel = "grey"

cam_pic2_front = Image_loader(
    path_pic2_front,
    color_channel,
    None,
    None,
    rot_angle,
    [y1_front, y2_front, x1_front, x2_front],
    None,
    None,
    shift_image_front,
    mask_border_front,
    False,
    10 / 9,
).image
cam_pic2_top = Image_loader(
    path_pic2_top,
    color_channel,
    None,
    None,
    rot_angle,
    [y1_top, y2_top, x1_top, x2_top],
    None,
    None,
    shift_image_top,
    mask_border_top,
    False,
    1,
).image
cam_pic2_120 = Image_loader(
    path_pic2_120,
    color_channel,
    None,
    1,
    rot_angle,
    [y1_120, y2_120, x1_120, x2_120],
    None,
    None,
    shift_image_120,
    mask_border_120,
    False,
    10 / 9,
).image
cam_pic2_240 = Image_loader(
    path_pic2_240,
    color_channel,
    None,
    1,
    rot_angle,
    [y1_240, y2_240, x1_240, x2_240],
    None,
    None,
    shift_image_240,
    mask_border_240,
    False,
    10 / 9,
).image


cam_pic_front = Image_loader(
    path_pic_front,
    color_channel,
    None,
    None,
    rot_angle,
    [y1_front, y2_front, x1_front, x2_front],
    None,
    None,
    shift_image_front,
    mask_border_front,
    False,
    10 / 9,
).image  # DELTAX,DELTAY SHIFTING
cam_pic_top = Image_loader(
    path_pic_top,
    color_channel,
    None,
    None,
    rot_angle,
    [y1_top, y2_top, x1_top, x2_top],
    None,
    None,
    shift_image_top,
    mask_border_top,
    False,
    1,
).image
cam_pic_120 = Image_loader(
    path_pic_120,
    color_channel,
    None,
    1,
    rot_angle,
    [y1_120, y2_120, x1_120, x2_120],
    None,
    None,
    shift_image_120,
    mask_border_120,
    False,
    10 / 9,
).image
cam_pic_240 = Image_loader(
    path_pic_240,
    color_channel,
    None,
    1,
    rot_angle,
    [y1_240, y2_240, x1_240, x2_240],
    None,
    None,
    shift_image_240,
    mask_border_240,
    False,
    10 / 9,
).image


#
correction_matrix = Correction_matrix_producer(
    shape_front,
    shape_side,
    shift_image_top,
    shift_image_120,
    shift_image_240,
    shift_image_front,
    mask_border_front,
    mask_border_top,
    mask_border_120,
    mask_border_240,
).correction_matrix

# correction_matrix=scipy.ndimage.gaussian_filter(correction_matrix,sigma=1.5) #xrays
# correction_matrix = np.ones(np.shape(correction_matrix))
#
Tomograpic_viewer(correction_matrix, False, 4, s)


#######################################directory for saving reconstruction file
save_directory = "pictures/2021-09-01/"
#####################################################################################


#######################################################Subtracting the backgroung

cam_pic_front = cam_pic_front - cam_pic2_front
cam_pic_top = cam_pic_top - cam_pic2_top
cam_pic_120 = cam_pic_120 - cam_pic2_120
cam_pic_240 = cam_pic_240 - cam_pic2_240


cam_pic_front[cam_pic_front < 0] = 0
cam_pic_top[cam_pic_top < 0] = 0
cam_pic_120[cam_pic_120 < 0] = 0
cam_pic_240[cam_pic_240 < 0] = 0


########################################################Show camera pictures
smin = Camera_picture_viewer(
    cam_pic_front, cam_pic_top, cam_pic_120, cam_pic_240, False, 16
).sminval  # here you can enable the logaritmic scale

cam_pic_front[cam_pic_front < smin] = 0
cam_pic_top[cam_pic_top < smin] = 0
cam_pic_120[cam_pic_120 < smin] = 0
cam_pic_240[cam_pic_240 < smin] = 0
# Camera_picture_viewer(cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,False,16)


########################################################number of iterations
max_it = 5


########################################################Reconstruction
reconstructor = Reconstructor(
    max_it,
    cam_pic_front,
    cam_pic_top,
    cam_pic_120,
    cam_pic_240,
    correction_matrix,
    True,
)  # the last one is the median filter
reconstructor.perform_MLEM()
rec_light_dist = reconstructor.rec_light_dist
# rec_light_dist=rec_light_dist.astype(int)
np.save(save_directory + "rec_light_dist_shot29", rec_light_dist)

# Show reconstruction
Tomograpic_viewer(
    rec_light_dist / rec_light_dist.max(), False, 1, s
)  # here you can enable the logaritmic scale


dist_3D = np.sum(rec_light_dist, axis=(1, 2))
dist_3D_mask = np.zeros(np.shape(rec_light_dist))


"""MASK"""


def create_circular_mask(h, w, center=None, radius=None):
    """Function that creates  a circular mask.

    Parameters
    ----------
    h : integer
        image height.
    w : integer
        image width.
    center : tuple
        Coordinates of image center .
    radius : type
        Description of parameter `radius`.

    Returns
    -------
    mask: boolean 1D array
        1DArray of 0 and 1 .

    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    YY, XX = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


mean_array = np.zeros((deltaz))
std = np.zeros((deltaz))
err = np.zeros((deltaz))
for index,element in enumerate(mean_array):
    img = rec_light_dist[index, :, :]
    h, w = img.shape[:2]

    innerradius = 34  # 34 40 #5mm diamter inner circle
    # innerradius=55 #7mm
    mask = create_circular_mask(
        h, w, radius=innerradius
    )  # create a boolean circle mask
    dist_3D_mask[index, mask] = img[mask]
    element = np.mean(img[mask], axis=0) #element=mean_array[index]
    std[index] = np.std(img[mask], axis=0)
    err[index] = std[index] / np.sqrt(len(img[mask]))


Tomograpic_viewer(
    dist_3D_mask / dist_3D_mask.max(), False, 1, s
)  # here you can enable the logaritmic scale


TOF = [0, 0, 0]
mean_array = np.append(mean_array, TOF)

# directory for saving mean dose array

save_directory = "pictures/2021-09-01/notnormalized/"
# save_directory='pictures/spatialresolution/'
#####################################################################################
np.save(save_directory + "notnormalizedmean_array" + "29", mean_array)
np.save(save_directory + "notnormalizederr" + "29", err)
np.save(save_directory + "notnormalizedmean_array_notmasked" + "29", dist_3D)

#############################################################################################

dist_1D_120 = np.flip(np.sum(cam_pic_120, axis=(1)))

np.save(save_directory + "dist_1D_120_" + "29", dist_1D_120)
plt.plot(
    np.arange(0, deltaz, 1) * s, dist_1D_120 / np.max(dist_1D_120), ".", label="120 "
)
dist_1D_240 = np.flip(np.sum(cam_pic_240, axis=(1)))
np.save(save_directory + "dist_1D_240_" + "29", dist_1D_240)
plt.plot(
    np.arange(0, deltaz, 1) * s, dist_1D_240 / np.max(dist_1D_240), ".", label=" 240"
)
dist_1D_top = np.flip(np.sum(cam_pic_top, axis=(1)))
np.save(save_directory + "dist_1D_top_" + "29", dist_1D_top)
plt.plot(
    (np.arange(0, deltaz, 1) * s), dist_1D_top / np.max(dist_1D_top), ".", label="top"
)

plt.plot(
    np.arange(0, deltaz, 1) * s,
    dist_3D / np.max(dist_3D),
    ".",
    label="3D reconstruction",
)


mean_array = mean_array[:-3]  # eliminate the TOF data
plt.errorbar(
    np.arange(0, len(mean_array), 1) * s,
    mean_array / np.max(mean_array),
    yerr=err / np.max(mean_array),
    xerr=None,
    fmt="",
    ecolor=None,
    elinewidth=None,
    label="3D masked image",
)


plt.title(
    "Depth dose distribution 1D  projection vs 3D reconstruction ",
    fontdict=None,
    loc="center",
    fontsize=22,
)
plt.legend(fontsize="large")
plt.xlabel("Depth[mm]", fontsize=18)
plt.ylabel("Intensity", fontsize=18)

plt.grid(b=True, which="major", color="k", linestyle="-", alpha=0.2)
plt.grid(b=True, which="minor", color="k", linestyle="-", alpha=0.2)
plt.minorticks_on()
plt.show()


#######################################################LET CORRECTION OF 3D DATA IN Depth

outputfile_topas = "/home/corvin22/Desktop/precoiusthings/SOBP/12PC/152.5MeV/LET_fluenceweighted_152500KeVproton_PVT_12PC_1Dscorer.csv"
header = pd.read_csv(outputfile_topas, nrows=7)
df = pd.read_csv(outputfile_topas, comment="#", header=None)
topas_datamatrix = np.array(df)  # convert dataframe df to array
letprofile = Get_profile(topas_datamatrix, 149, 1)
zletprofile = letprofile.zmeanprofile
zletprofile = zletprofile[::-1]


letcorrection = letcorrection(mean_array, zletprofile, s)

lightcorrection_value = letcorrection.lightcorrection

matrix_3D_corrected = np.zeros(
    np.shape(rec_light_dist)
)  # ((deltaz, 186, 152) z is the first =deltaz


for i in range(np.shape(rec_light_dist)[0]):  # from 0 to 158
    matrix_3D_corrected[i, :, :] = np.divide(
        rec_light_dist[i, :, :],
        lightcorrection_value[i],
        out=np.zeros_like(rec_light_dist[i, :, :]),
        where=lightcorrection_value[i] != 0,
    )


Tomograpic_viewer(
    matrix_3D_corrected / matrix_3D_corrected.max(), False, 1, s
)  # here you can enable the logaritmic scale
# matrix_3D_corrected_calib=matrix_3D_corrected*calib_value
# print(matrix_3D_corrected_calib.max())
# Tomograpic_viewer(matrix_3D_corrected_calib,False,0.834) # here you can enable the logaritmic scale


save_directory = "pictures/2021-09-01/"
np.save(save_directory + "letcorrectedrec_light_dist_shot29", matrix_3D_corrected)
