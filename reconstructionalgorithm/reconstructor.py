import numpy as np
import cv2
import scipy.ndimage
from tqdm import tqdm
from tomograpic_viewer import Tomograpic_viewer




class Reconstructor(object):
    """ Class for MLEM reconstruction
.

    Parameters
    ----------
    max_it : type
        Description of parameter `max_it`.
    cam_pic_front : type
        Description of parameter `cam_pic_front`.
    cam_pic_top : type
        Description of parameter `cam_pic_top`.
    cam_pic_120 : type
        Description of parameter `cam_pic_120`.
    cam_pic_240 : type
        Description of parameter `cam_pic_240`.
    correction_matrix : type
        Description of parameter `correction_matrix`.
    median_filter : type
        Description of parameter `median_filter`.

    Attributes
    ----------
    rec_light_dist : type
        Description of attribute `rec_light_dist`.
    max_it
    cam_pic_front
    cam_pic_top
    cam_pic_120
    cam_pic_240
    correction_matrix
    median_filter

    """

    # Initialization of class
    def __init__(
        self,
        max_it,
        cam_pic_front,
        cam_pic_top,
        cam_pic_120,
        cam_pic_240,
        correction_matrix,
        median_filter,
    ):

        # Number of iterations
        self.max_it = max_it

        # Camera pictures
        self.cam_pic_front = cam_pic_front.astype(float)
        self.cam_pic_top = cam_pic_top.astype(float)
        self.cam_pic_120 = cam_pic_120.astype(float)
        self.cam_pic_240 = cam_pic_240.astype(float)

        # Correction matrix
        self.correction_matrix = correction_matrix.astype(float)
        self.rec_light_dist = (np.zeros((np.shape(correction_matrix))) + 1).astype(
            float
        )

        # Median Filter
        self.median_filter = median_filter

    # Iterative MLEM algorithm for 3D reconstruction of the light distribution measured with SCIDOM detector
    def MLEM(self):

        # Multiply previous/starting light distribution with correction array
        # self.rec_light_dist=np.multiply(self.rec_light_dist,self.correction_matrix,out=np.zeros_like(self.rec_light_dist),where=np.logical_and(self.rec_light_dist!=0,self.correction_matrix!=0))

        # Rotate 3D light distributions for 120° and 240° camera view
        rec_light_dist_120 = scipy.ndimage.rotate(
            self.rec_light_dist, 120, axes=(1, 2), reshape=False, prefilter=True
        )
        rec_light_dist_240 = scipy.ndimage.rotate(
            self.rec_light_dist, 240, axes=(1, 2), reshape=False, prefilter=True
        )

        # Calculate forward projections
        forw_proj_front = self.rec_light_dist.sum(axis=0)
        forw_proj_top = self.rec_light_dist.sum(axis=1)
        forw_proj_120 = rec_light_dist_120.sum(axis=1)
        forw_proj_240 = rec_light_dist_240.sum(axis=1)

        # Multiply for the correction matrix

        # Divide cam pictures (projection_values) by forward projections
        quotient_front = np.divide(
            self.cam_pic_front,
            forw_proj_front,
            out=np.zeros_like(self.cam_pic_front),
            where=np.logical_and(self.cam_pic_front != 0, forw_proj_front != 0),
        )
        quotient_top = np.divide(
            self.cam_pic_top,
            forw_proj_top,
            out=np.zeros_like(self.cam_pic_top),
            where=np.logical_and(self.cam_pic_top != 0, forw_proj_top != 0),
        )
        quotient_120 = np.divide(
            self.cam_pic_120,
            forw_proj_120,
            out=np.zeros_like(self.cam_pic_120),
            where=np.logical_and(self.cam_pic_120 != 0, forw_proj_120 != 0),
        )
        quotient_240 = np.divide(
            self.cam_pic_240,
            forw_proj_240,
            out=np.zeros_like(self.cam_pic_240),
            where=np.logical_and(self.cam_pic_240 != 0, forw_proj_240 != 0),
        )

        # Back project qoutient into 3D volume
        rec_part_front = np.stack(
            [quotient_front] * np.shape(self.rec_light_dist)[0], axis=0
        )
        rec_part_top = np.stack(
            [quotient_top] * np.shape(self.rec_light_dist)[1], axis=1
        )
        rec_part_120 = np.stack(
            [quotient_120] * np.shape(self.rec_light_dist)[1], axis=1
        )
        rec_part_240 = np.stack(
            [quotient_240] * np.shape(self.rec_light_dist)[1], axis=1
        )

        # Rotate backprojected 120° and 240° light distributions
        rec_part_120 = scipy.ndimage.rotate(
            rec_part_120, 120, axes=(1, 2), reshape=False, prefilter=True
        )
        rec_part_240 = scipy.ndimage.rotate(
            rec_part_240, 240, axes=(1, 2), reshape=False, prefilter=True
        )

        # Add back projected light distributions
        back_proj_light_dist = (
            rec_part_front + rec_part_120 + rec_part_240 + rec_part_top
        )

        # Multiply back projected 3D light distributions with previous light distribution (already multiplied with correction matrix)
        self.rec_light_dist = np.multiply(
            self.rec_light_dist,
            back_proj_light_dist,
            out=np.zeros_like(self.rec_light_dist),
            where=np.logical_and(self.rec_light_dist != 0, back_proj_light_dist != 0),
        )
        self.rec_light_dist = np.divide(
            self.rec_light_dist,
            self.correction_matrix,
            out=np.zeros_like(self.rec_light_dist),
            where=np.logical_and(self.rec_light_dist != 0, self.correction_matrix != 0),
        )
        if self.median_filter == True:
            # Perform cubic median filter
            # filtered_rec_light_dist = scipy.ndimage.sobel(self.rec_light_dist)
            # Tomograpic_viewer(filtered_rec_light_dist,False,8)
            # self.rec_light_dist = self.rec_light_dist - np.abs(filtered_rec_light_dist)
            # self.rec_light_dist[self.rec_light_dist<0] = 0.0
            self.rec_light_dist = scipy.ndimage.median_filter(
                self.rec_light_dist, size=5
            )  # laser
            # self.rec_light_dist=scipy.ndimage.median_filter(self.rec_light_dist, size=5) #protons
            # self.rec_light_dist=scipy.ndimage.gaussian_filter(self.rec_light_dist,sigma=1) #oncoray protons
            # self.rec_light_dist=scipy.ndimage.gaussian_filter(self.rec_light_dist,sigma=1.5) #xrays

        # Normalization
        # if self.rec_light_dist.max()>0:
        #   norm=np.divide(255,self.rec_light_dist.max())
        #   self.rec_light_dist=np.multiply(self.rec_light_dist,norm,out=np.zeros_like(self.rec_light_dist),where=self.rec_light_dist!=0)

    # Perform iterative reconstruction
    def perform_MLEM(self):

        # Loop for iterative process
        for i in tqdm(range(self.max_it)):
            self.MLEM()

        # self.rec_light_dist=np.flip(self.rec_light_dist,axis=(0,1,2))
        self.rec_light_dist = np.flip(self.rec_light_dist, axis=(0, 1))
