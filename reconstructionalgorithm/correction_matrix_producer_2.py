import numpy as np
from tomograpic_viewer import Tomograpic_viewer
import pandas as pd
import scipy.ndimage
from tqdm import tqdm



#Class for calculating the response matrix in reconstruction volume
class Correction_matrix_producer(object):


############################### Constructor of class ###################################
                                                                                       #
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
                                                                                       #
############################ Parameters #####################################
                                                                            #
        self.shape_front = shape_front
        self.shape_side = shape_side
                                                                            #
        self.shift_image_top = shift_image_top
        self.shift_image_120 = shift_image_120
        self.shift_image_240 = shift_image_240
        self.shift_image_front = shift_image_front

        self.mask_border_front = mask_border_front
        self.mask_border_top   = mask_border_top
        self.mask_border_120   = mask_border_120
        self.mask_border_240   = mask_border_240
                                                                            #
        self.correction_matrix = self.calc_corr_matrix()
                                                                            #
#############################################################################
                                                                                       #
########################################################################################


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


###################################### Test class ######################################
                                                                                       #
if __name__ == "__main__":
                                                                                       #
    shape_front = (186,152)
    shape_side = (161,152)
                                                                                       #
    shift_image_top = [0,0]
    shift_image_120 = [20,0]
    shift_image_240 = [-20,0]
    shift_image_front = [5,8]
                                                                                       #
    correction_matrix = Correction_matrix_producer(      shape_front,
                                                          shape_side,
                                                     shift_image_top,
                                                     shift_image_120,
                                                     shift_image_240,
                                                   shift_image_front ).correction_matrix
                                                                                       #
    Tomograpic_viewer(correction_matrix,False,4)
                                                                                       #
########################################################################################
