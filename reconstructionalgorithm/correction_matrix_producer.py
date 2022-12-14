import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm


class Correction_matrix_producer(object):
    """#Class for calculating the response matrix in reconstruction volume.

    Parameters
    ----------
    shape_front : tuple-like
        Tuple that defines front projection dimension.
    shape_side : tuple-like
        Tuple that defines lateral projections dimension.
    shift_image_top : list-like
        List of 2 integr numbers that defines how much the top 2D array is going to be shifted.
    shift_image_120 : list-like
        Description of parameter `shift_image_120`.
    shift_image_240 :list-like
        Description of parameter `shift_image_240`.
    shift_image_front : list-like
        List of 2 elements.
    mask_border_front : list-like
        List of 4 elements.
    mask_border_top : list-like
        List of 4 elements.
    mask_border_120 : list-like
        List of 4 elements.
    mask_border_240 : list-like
        List of 4 elements. The first teo numbers are used to set the side values to zero,
        while the other two are used to set the front number to 0.

    Attributes
    ----------
    shape_front

    """


    def __init__(             self,
                       shape_front,
                        shape_side,
                   shift_image_top,
                   shift_image_120,
                   shift_image_240,
                 shift_image_front,
                 mask_border_front,
                 mask_border_top,
                 mask_border_120,
                 mask_border_240 ):


        self.shape_front = shape_front
        self.shape_side = shape_side

        self.shift_image_top = shift_image_top
        self.shift_image_120 = shift_image_120
        self.shift_image_240 = shift_image_240
        self.shift_image_front = shift_image_front

        self.mask_border_front = mask_border_front
        self.mask_border_top   = mask_border_top
        self.mask_border_120   = mask_border_120
        self.mask_border_240   = mask_border_240

        self.correction_matrix = self.calc_corr_matrix()




############################ Calculate correction matrix ###############################
                                                                                       #
    def calc_corr_matrix(self):
                                                                                       #
        side_picture_array = np.ones(self.shape_side)
                                                                                       #
        picture_array_top = np.copy(side_picture_array)
        picture_array_120 = np.copy(side_picture_array)
        picture_array_240 = np.copy(side_picture_array)
                                                                                       #
        picture_array_front = np.ones(self.shape_front)
                                                                                       #
        if self.shift_image_top[1]>0:
            picture_array_top[:self.shift_image_top[1],:]=0
        elif self.shift_image_top[1]<0:
            picture_array_top[self.shift_image_top[1]:,:]=0
                                                                                       #
        if self.shift_image_top[0]>0:
            picture_array_top[:,-self.shift_image_top[0]:]=0
        elif self.shift_image_top[0]<0:
            picture_array_top[:,:-self.shift_image_top[0]]=0
                                                                                       #
        if self.shift_image_120[1]>0:
            picture_array_120[:self.shift_image_120[1],:]=0
        elif self.shift_image_120[1]<0:
            picture_array_120[self.shift_image_120[1]:,:]=0
                                                                                       #
        if self.shift_image_120[0]>0:
            picture_array_120[:,-self.shift_image_120[0]:]=0
        elif self.shift_image_120[0]<0:
            picture_array_120[:,:-self.shift_image_120[0]]=0
                                                                                       #
        if self.shift_image_240[1]>0:
            picture_array_240[:self.shift_image_240[1],:]=0
        elif self.shift_image_240[1]<0:
            picture_array_240[self.shift_image_240[1]:,:]=0
                                                                                       #
        if self.shift_image_240[0]>0:
            picture_array_240[:,-self.shift_image_240[0]:]=0
        elif self.shift_image_240[0]<0:
            picture_array_240[:,:-self.shift_image_240[0]]=0
                                                                                       #
        if self.shift_image_front[1]>0:

            picture_array_front[-self.shift_image_front[1]:,:]=0
        elif self.shift_image_front[1]<0:
            picture_array_front[:-self.shift_image_front[1],:]=0

                                                                                       #
        if self.shift_image_front[0]>0:
            picture_array_front[:,-self.shift_image_front[0]:]=0
        elif self.shift_image_front[0]<0:
            picture_array_front[:,:-self.shift_image_front[0]]=0




        if self.mask_border_front!=None:
                    #set side values to zero
                    if self.mask_border_front[0]>0:
                        picture_array_front[:,:self.mask_border_front[0]]=0


                    if self.mask_border_front[1]>0:
                        picture_array_front[:,-self.mask_border_front[1]:]=0

                    #set top values to zero
                    if self.mask_border_front[2]>0:
                        picture_array_front[:self.mask_border_front[2],:]=0


                    if self.mask_border_front[3]>0:
                        picture_array_front[-self.mask_border_front[3]:,:]=0



        if self.mask_border_top !=None:
                        #set side values to zero
                        if self.mask_border_top [0]>0:
                            picture_array_top[:,:self.mask_border_top[0]]=0

                        if self.mask_border_top [1]>0:
                            picture_array_top[:,-self.mask_border_top[1]:]=0


                        #set top values to zero
                        if self.mask_border_top [2]>0:
                            picture_array_top[:self.mask_border_top[2],:]=0


                        if self.mask_border_top [3]>0:
                            picture_array_top[-self.mask_border_top[3]:,:]=0



        if self.mask_border_120!=None:
                    #set side values to zero
                    if self.mask_border_120[0]>0:
                        picture_array_120[:,:self.mask_border_120[0]]=0


                    if self.mask_border_120[1]>0:
                        picture_array_120[:,-self.mask_border_120[1]:]=0


                    #set top values to zero
                    if self.mask_border_120[2]>0:
                        picture_array_120[:self.mask_border_120[2],:]=0

                    if self.mask_border_120[3]>0:
                        picture_array_120[-self.mask_border_120[3]:,:]=0




        if self.mask_border_240!=None:
                    #set side values to zero
                    if self.mask_border_240[0]>0:
                        picture_array_240[:,:self.mask_border_240[0]]=0


                    if self.mask_border_240[1]>0:
                        picture_array_240[:,-self.mask_border_240[1]:]=0

                    #set top values to zero
                    if self.mask_border_240[2]>0:
                        picture_array_240[:self.mask_border_240[2],:]=0


                    if self.mask_border_240[3]>0:
                        picture_array_240[-self.mask_border_240[3]:,:]=0



                                                                                       #
        correction_matrix_front = np.stack([picture_array_front]*self.shape_side[0],
                                                                             axis=0 )
                                                                                       #
        correction_matrix_top = np.stack([picture_array_top]*self.shape_front[0],
                                                                          axis=1 )
                                                                                       #
        correction_matrix_120 = np.stack([picture_array_120]*self.shape_front[0],
                                                                          axis=1 )
                                                                                       #
        correction_matrix_240 = np.stack([picture_array_240]*self.shape_front[0],
                                                                          axis=1 )
                                                                                       #
        correction_matrix_120 = scipy.ndimage.rotate(correction_matrix_120,
                                                           120,axes=(1, 2),
                                                             reshape=False,
                                                            prefilter=True )
                                                                                       #
        correction_matrix_240 = scipy.ndimage.rotate(correction_matrix_240,
                                                           240,axes=(1, 2),
                                                             reshape=False,
                                                            prefilter=True )
                                                                                       #
        correction_matrix = correction_matrix_top + correction_matrix_120 +\
                            correction_matrix_240 + correction_matrix_front
                                                                                       #
        return correction_matrix
                                                                                       #
########################################################################################
