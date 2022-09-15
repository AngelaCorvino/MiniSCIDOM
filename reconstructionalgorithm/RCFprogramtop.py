""" EVALUATE TOP PROJECTION OF MINISCIDOM AND RCF MATRIX"""

from pathlib import Path
import cv2
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from image_loader import Image_loader
from camera_picture_viewer import Camera_picture_viewer
from reconstructor import Reconstructor
from tomograpic_viewer import Tomograpic_viewer
from twoDtoptomograpic_viewer import twoDtopTomograpic_viewer
from rcf_rebinner import RCF_rebinner
from calibrationfactor import calibrationfactor
from scipy.optimize import curve_fit
import seaborn as sns
from rcf_matrix_producer import RCF_matrix_producer
import pandas as pd
from scipy.ndimage.interpolation import rotate
import skimage
from skimage.measure import profile_line
#Read_path


    #Read_path
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/21CY/cutfilm/'
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/85P3/'
#directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/85P3/'
filename_1='Doseof-01.tif'
filename_2='Doseof-02.tif'
filename_3='Doseof-03.tif'
filename_4='Doseof-04.tif'
filename_5='Doseof-05.tif'
filename_6='Doseof-06.tif'
filename_7='Doseof-07.tif'
filename_8='Doseof-08.tif'
filename_9='Doseof-09.tif'
filename_10='Doseof-10.tif'
filename_11='Doseof-11.tif'
filename_12='Doseof-12.tif'
filename_13='Doseof-13.tif'
filename_14='Doseof-14.tif'
filename_15='Doseof-15.tif'
filename_16='Doseof-16.tif'
filename_17='Doseof-17.tif'
filename_18='Doseof-18.tif'
filename_19='Doseof-19.tif'
filename_20='Doseof-20.tif'
filename_21='Doseof-21.tif'
filename_22='Doseof-22.tif'
filename_23='Doseof-23.tif'
filename_24='Doseof-24.tif'
filename_25='Doseof-25.tif'
filename_26='Doseof-26.tif'
filename_27='Doseof-27.tif'
filename_28='Doseof-28.tif'
filename_29='Doseof-29.tif'
filename_30='Doseof-30.tif'


file_path =[     directory+filename_1,
                 directory+filename_2,
                 directory+filename_3,
                 directory+filename_4,
                 directory+filename_5,
                 directory+filename_6,
                 directory+filename_7,
                 directory+filename_8,
                 directory+filename_9,
                directory+filename_10,
                directory+filename_11,
                directory+filename_12,
                directory+filename_13,
                directory+filename_14,
                directory+filename_15,
                directory+filename_16,
                directory+filename_17,
                directory+filename_18,
                directory+filename_19,
                directory+filename_20,
                directory+filename_21,
                directory+filename_22,
                directory+filename_23,
                directory+filename_24,
                directory+filename_25,
                directory+filename_26,
                directory+filename_27,
                directory+filename_28,
                directory+filename_29,
                directory+filename_30
                ]


rcf_matrix = RCF_matrix_producer(file_path).produce_stack()

#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
directory1='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/'
filename="cam_pic_top_shot137_7iter.npy"
dist_2D_top= np.load(directory1+filename)
dist_2D_top=np.flip(dist_2D_top,axis=0)
n=3
Xprofiletop=np.zeros(dist_2D_top.shape[1]+1)
for i in range(-n,n):
    '''
    Return the intensity profile of a 2D image(ZX profile) measured along a scan line , x profile
    '''
    xprofiletop=skimage.measure.profile_line(dist_2D_top, (dist_2D_top.shape[0]/2+i,0),
                                             (dist_2D_top.shape[0]/2+i,dist_2D_top.shape[1]),
                                                            linewidth=1,
                                                             order=None,
                                                              mode=None,
                                                               cval=0.0)
    Xprofiletop += xprofiletop
#self.xtop=np.average(self.ZXprofile, axis=1)
xtop_raw=Xprofiletop /2*n
xtop_raw=xtop_raw/xtop_raw.max()
plt.figure(1)
plt.title('CCD',fontsize=26)
plt.imshow(dist_2D_top,cmap='jet',aspect="equal",origin='lower')
plt.xlabel('x in pixel',fontsize=24)
plt.ylabel('z in pixel',fontsize=24)
plt.tick_params(axis='x', which='major', labelsize=24)
plt.tick_params(axis='y', which='major', labelsize=24)
plt.axhline(y=dist_2D_top.shape[0]/2+n, xmin=0, xmax=dist_2D_top.shape[1], linewidth=1, color = 'black')
plt.axhline(y=dist_2D_top.shape[0]/2-n, xmin=0, xmax=dist_2D_top.shape[1], linewidth=1, color = 'black')
plt.fill_between(np.arange(0,dist_2D_top.shape[1]),
np.ones(dist_2D_top.shape[1])*(dist_2D_top.shape[0]/2+n),np.ones(dist_2D_top.shape[1])*(dist_2D_top.shape[0]/2-n),
                                                                   color='gray',
                                                                     alpha=0.7)

plt.show()














directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/'
file_name_miniSCIDOM ='rec_light_dist_shot137.npy'
(rec,light,dist,shotnumber)= file_name_miniSCIDOM.split('_')
(shotnumber,extension)=shotnumber.split('.npy')
(shot,number)=shotnumber.split('t')
file_name_RCF='RCFmatrix85P3.npy'
(rcfname,extension)=file_name_RCF.split('.npy')
(RCF,rcfname)= rcfname.split('matrix')

path_miniSCIDOM=directory+file_name_miniSCIDOM
path_RCF=directory+file_name_RCF
miniscidom_matrix = np.load(path_miniSCIDOM)

miniscidom_pixel_size = 0.0508 #mm/px
m=miniscidom_pixel_size
rcf_pixel_size_lat = 0.0847 #mm/px
rcf_pixel_size_dep = 0.28*1.02 #mm/px
wepl=4.95 # mm



matrix = RCF_rebinner(  miniscidom_matrix[:,:,:],
                      miniscidom_pixel_size,
                        rcf_matrix[:,:,:], #3,10 #85o2rcf_matrix[:-2,:,:]85P3#rcf_matrix[:-2,3:,10:]85O2#85p2[:-2,:-4,8:]
                         rcf_pixel_size_lat,
                         rcf_pixel_size_dep ).zoom_matrix

matrix =scipy.ndimage.shift(matrix,[0,0,-10]) #[0,0,-8]RCF85P3
matrix[matrix<10**(-6)]=0
###############################################################################


#Tomograpic_viewer(matrix/matrix.max(),False,1,m)
#Tomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,m)
difference = np.abs(matrix/matrix.max()-miniscidom_matrix/miniscidom_matrix.max())
#Tomograpic_viewer(difference,False,1,m)
differencepercentage=np.divide(difference,matrix/matrix.max(),out=np.zeros_like(matrix),where=matrix!=0)*100
#Tomograpic_viewer(differencepercentage,False,20,m)



#X profile
miniprofile=twoDtopTomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,miniscidom_pixel_size,wepl)
mini_top=miniprofile.xtop
mini_top=mini_top/mini_top.max()


RCFprofile=twoDtopTomograpic_viewer(matrix/matrix.max(),False,1,miniscidom_pixel_size,wepl)
RCF_top=RCFprofile.xtop
RCF_top=RCF_top/RCF_top.max()

#Center of mass
#xcm_rcf=np.average(RCF_xmeanprofile,axis=1,weights=RCF_zmeanprofile) #calculated in pixel
xcm_rcf=np.average(np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size,weights=RCF_top)
xcm_sci=np.average(np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size,weights=mini_top)
#xcm_rcf=np.average(np.arange(0,matrix.shape[1]+1)*miniscidom_pixel_size,weights=RCF_zmeanprofile)
#xcm_sci=np.average(np.arange(0,miniscidom_matrix.shape[1]+1)*miniscidom_pixel_size,weights=mini_zmeanprofile)
shift=(xcm_sci-xcm_rcf)


#########################################################################################################FIT


#############################################################################################################
####PLOT

plt.figure(2)

plt.plot((np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size),
                                                                        RCF_top,
                                                                             '.',
                                                                    Markersize=11,
                                  label='RCF')


plt.plot((np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size),
                                                                        mini_top,
                                                                              '.',
                                                                    Markersize=11,
                     label='MS reconstruction')

plt.plot((np.arange(0,dist_2D_top.shape[1]+1)*miniscidom_pixel_size),
                                                                        xtop_raw,
                                                                              '.',
                                                                    Markersize=11,
                            label='MS raw data')



plt.ylim([-0.1,1.1])
plt.title('X-profile top projection ',fontsize=22)
plt.xlabel('x[mm]',fontsize=22)
plt.ylabel('Relative Intensity',fontsize=22)
plt.legend(title='',fontsize='20',markerscale=2,loc=3)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)

plt.show()
