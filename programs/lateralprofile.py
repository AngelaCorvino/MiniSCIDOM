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
from twoDtomograpic_viewer import twoDTomograpic_viewer
from rcf_rebinner import RCF_rebinner
from calibrationfactor import calibrationfactor
from scipy.optimize import curve_fit
import seaborn as sns
from rcf_matrix_producer import RCF_matrix_producer
import pandas as pd
from scipy.ndimage.interpolation import rotate
#Read_path


    #Read_path
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/21CY/cutfilm/'
#directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/85O2/'
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-08-13/85P3/'
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
wepl=4.914 # mm



matrix = RCF_rebinner(    miniscidom_matrix[:,:,:],
                      miniscidom_pixel_size,
                        rcf_matrix[:,:,:], #3,10 #85o2rcf_matrix[:-2,:,:]85P3#rcf_matrix[:-2,3:,10:]85O2#85p2[:-2,:-4,8:]
                         rcf_pixel_size_lat,
                         rcf_pixel_size_dep ).zoom_matrix
#print(matrix)

matrix =scipy.ndimage.shift(matrix,[0,0,-8]) #[0,-1,0]RCF85O2
matrix[matrix<10**(-6)]=0
###############################################################################


#Tomograpic_viewer(matrix/matrix.max(),False,1,m)
#Tomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,m)
difference = np.abs(matrix/matrix.max()-miniscidom_matrix/miniscidom_matrix.max())
#Tomograpic_viewer(difference,False,1,m)
differencepercentage=np.divide(difference,matrix/matrix.max(),out=np.zeros_like(matrix),where=matrix!=0)*100
#Tomograpic_viewer(differencepercentage,False,20,m)



#X profile
miniprofile=twoDTomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,miniscidom_pixel_size,wepl)


mini_xmeanprofile=miniprofile.xmeanprofile
#mini_xmeanprofile=mini_xmeanprofile-mini_xmeanprofile[0:len(mini_xmeanprofile)-2].min()
mini_ymeanprofile=miniprofile.ymeanprofile

RCFprofile=twoDTomograpic_viewer(matrix/matrix.max(),False,1,miniscidom_pixel_size,wepl)

RCF_xmeanprofile=RCFprofile.xmeanprofile
RCF_ymeanprofile=RCFprofile.ymeanprofile

x=RCF_xmeanprofile
y=RCF_ymeanprofile
RCF_xmeanprofile=RCF_xmeanprofile/RCF_xmeanprofile.max()
mini_xmeanprofile=mini_xmeanprofile/mini_xmeanprofile.max()
#y=np.flip(y) # Why flip this ?

#Center of mass
#xcm_rcf=np.average(RCF_xmeanprofile,axis=1,weights=RCF_ymeanprofile) #calculated in pixel
xcm_rcf=np.average(np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size,weights=RCF_xmeanprofile)
xcm_sci=np.average(np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size,weights=mini_xmeanprofile)

#xcm_rcf=np.average(np.arange(0,matrix.shape[1]+1)*miniscidom_pixel_size,weights=RCF_ymeanprofile)
#xcm_sci=np.average(np.arange(0,miniscidom_matrix.shape[1]+1)*miniscidom_pixel_size,weights=mini_ymeanprofile)



shift=(xcm_sci-xcm_rcf)
print(shift)


#########################################################################################################FIT
#we need to mask the array to fit just the first part of the two curves


x1=np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size

#mask1=np.logical_and(x1>0.5,x1<1.05)
mask1=np.logical_and(x1>4.9,x1<5.4)

x1fit=x1[mask1]#fit
y1fit=RCF_xmeanprofile[mask1]


k=RCF_xmeanprofile.max()/mini_xmeanprofile.max()

x2=np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size

#mask2=np.logical_and(x2>0.5,x2<1.05)
#mask2=np.logical_and(x2>4.9,x2<5.6)
mask2=np.logical_and(x1>4.7,x1<5.4)
x2fit=x2[mask2]#fit
y2fit=mini_xmeanprofile[mask2]


def linear(x,a,b):
    return a*x+b

param_bounds=([0,0],[np.inf,np.inf])
popt = (1,0) #initial values
popt1, pcov1 = curve_fit(linear, x1fit, y1fit)#, p0=popt,bounds=param_bounds)
a1,b1= popt1
da1,db1 = np.sqrt(np.diag(pcov1))
popt2, pcov2 = curve_fit(linear, x2fit, y2fit)#, p0=popt,bounds=param_bounds)
a2,b2= popt2
da2,db2 = np.sqrt(np.diag(pcov2))

#############################################################################################################
####PLOT
 #normalization
#plot TCF x profile
plt.figure(6)

plt.plot((np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size),
                                                                RCF_xmeanprofile,
                                                                             '.',
                                                                    Markersize=13,
                                            color=sns.color_palette(  "Paired")[1],
                                                 label='$D_{RCF}$',
                                                 zorder=2)



plt.scatter(x1fit,y1fit,marker='^',color=sns.color_palette(  "Paired")[2])
plt.plot(x1fit,x1fit*popt1[0]+popt1[1],
                                                                            '-',
                                                color=sns.color_palette(  "Paired")[0],
                                                                 linewidth=7,
                                                              label='$D_{RCF}$ linear fit',
                                                              zorder=1)

plt.plot(np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size,
                                                                mini_xmeanprofile,
                                                                             '.',
                                            color=sns.color_palette(  "Paired")[2],
                                                                   Markersize=13,
                                    label='$D_{MS}$',
                                                           zorder=2)


plt.scatter(x2fit,y2fit,
                                                                    marker='^',
                                        color=sns.color_palette(  "Paired")[3])

plt.plot(x2fit,(x2fit*popt2[0]+popt2[1]),
                                                                            '-',
                                        color=sns.color_palette(  "Paired")[3],
                                                                   linewidth=7,
                                                              label='$D_{MS}$ linear fit',
                                                              zorder=1)

plt.xlim([0.76,8])
plt.ylim([-0.1,1.1])
plt.title('X-profile',fontsize=24)
plt.text(6,1,'wepl={} mm '.format(wepl),fontsize=22)
plt.xlabel('x[mm]',fontsize=22)
plt.ylabel(' Relative Intensity',fontsize=22)

print(a1,da1,b1,db1)
print(a2,da2,b2,db2)
theta1=np.arctan(a1)
dtheta1=np.arctan(da1)
theta2=np.arctan(a2)
dtheta2=np.arctan(da2)
plt.rc('text', usetex=True)
plt.legend( title=r'Linear model estimation: $\theta_1$'f'= {theta1:.2f} $\pm$ {dtheta1:.3f}\n' r',$\theta_2 $ 'f'= {theta2:.2f} $\pm$ {dtheta2:.3f} ',
                                                               fontsize='18',
                                                               markerscale=3,
                                                               title_fontsize=22,
                                                               loc=3)
plt.rc('text', usetex=False)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)



plt.figure(7)
#plt.plot(x+shift,y,'.',markersize=11,label='RCF accumulated dose')
plt.plot((np.arange(0,matrix.shape[1]+1)*miniscidom_pixel_size),
                                        RCF_ymeanprofile/RCF_ymeanprofile.max(),
                                                                             '.',
                                                                    Markersize=11,
                                  label='RCF{} Accumulated dose'.format(rcfname))


plt.plot(np.arange(0,miniscidom_matrix.shape[1]+1)*miniscidom_pixel_size,
                                        mini_ymeanprofile/mini_ymeanprofile.max(),
                                                                             '.',
                                                                   Markersize=11,
                                  label='$D_{MS}$')

plt.ylim([-0.1,1.1])
plt.title('Y-profile' ,fontsize=24)
plt.text(7,1,'wepl={} mm '.format(wepl),fontsize=24)
plt.xlabel('x[mm]',fontsize=22)
plt.ylabel(' Relative',fontsize=22)
#plt.legend( title=f'Linear model estimation: a1 = {a1:.1f} $\pm$ {da1:.1f},b1 = {b1:.0f} $\pm$ {db1:.0f},a2 = {a2:.1f} $\pm$ {da2:.1f}, b2 = {b2:.0f} $\pm$ {db2:.0f}',
#                                                                fontsize='18')
plt.legend(title='',loc=1,fontsize='18')
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)

plt.show()
