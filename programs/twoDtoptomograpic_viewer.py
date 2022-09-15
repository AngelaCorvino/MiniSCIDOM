import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
from PIL import Image
import numpy as np
import cv2
import matplotlib.image as mpimg
import skimage
from skimage.measure import profile_line
import pandas as pd
import matplotlib
from scipy.optimize import curve_fit
from tomograpic_viewer import Tomograpic_viewer
#Class for showing reconstruted 3D data slices


class twoDtopTomograpic_viewer(Tomograpic_viewer):

    #Initialization of class
    def __init__(self,X,c_log,max_val,miniscidom_pixel_size,wepl):
        """this class has all the attributes and method from tomograpic_viewer"""
        super().__init__(X,c_log,max_val,miniscidom_pixel_size)
        #3D data matrix
        self.wepl = wepl


        ########################################################################
        n=3
        self.ZXprofile=np.zeros((X.shape[0],X.shape[2]))
        self.ZXprofile=np.sum(X,axis=(1))
        image_2D=self.ZXprofile
        self.Xprofiletop=np.zeros(X.shape[2]+1)
        for i in range(-n,n):
            '''
            Return the intensity profile of a 2D image(ZX profile) measured along a scan line , x profile
            '''
            self.xprofiletop=skimage.measure.profile_line(image_2D, (X.shape[0]/2+i,0),
                                                     (X.shape[0]/2+i,X.shape[2]),
                                                                    linewidth=1,
                                                                     order=None,
                                                                      mode=None,
                                                                       cval=0.0)
            self.Xprofiletop += self.xprofiletop
        #self.xtop=np.average(self.ZXprofile, axis=1)
        self.xtop=self.Xprofiletop /2*n



        plt.figure(1)
        plt.title('RCF',fontsize=26)
        if self.c_log==True:
            plt.imshow(image_2D,cmap='jet',aspect="equal",origin='lower',norm=LogNorm())
        else:

            plt.imshow(image_2D,cmap='jet',aspect="equal",origin='lower')

        plt.xlabel('x in pixel',fontsize=24)
        plt.ylabel('z in pixel',fontsize=24)
        plt.tick_params(axis='x', which='major', labelsize=24)
        plt.tick_params(axis='y', which='major', labelsize=24)
        plt.axhline(y=image_2D.shape[0]/2+n, xmin=0, xmax=image_2D.shape[1], linewidth=1, color = 'black')
        plt.axhline(y=image_2D.shape[0]/2-n, xmin=0, xmax=image_2D.shape[1], linewidth=1, color = 'black')
        plt.fill_between(np.arange(0,X.shape[2]),
        np.ones(image_2D.shape[1])*(X.shape[0]/2+n),np.ones(image_2D.shape[1])*(image_2D.shape[0]/2-n),
                                                                           color='gray',
                                                                             alpha=0.7)
        #for i in range(0,X.shape[1]):
        #        self.zxprofile=X[:,i,:]
        #        self.ZXprofile+= self.zxprofile
        #self.ZXprofile=self.ZXprofile/X.shape[1]
        plt.show()
