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
#Read_path


######2 FILM from 2020-11-05

path1='pictures/2020-11-05/lowdose/2RCFlayers/xprofile/layer1.csv'
path2='pictures/2020-11-05/lowdose/2RCFlayers/xprofile/layer2.csv'

def read_data(path):
 #data = pd.read_excel (path)
    data=pd.read_csv(path,header=None,skiprows=1,delimiter=',')
    x=(data[0]) # pixel
    y=(data[1]) # intensity
    return x,y
depthfilm1,film1=read_data(path1)
depthfilm2,film2=read_data(path2)
film2=np.flip(film2, axis=0)
film1=np.flip(film1, axis=0)
film2=film2/film2.max()
film1=film1/film1.max()



mask=np.logical_and(depthfilm2>7.9,depthfilm2<8.74)
#mask=np.logical_and(depthfilm2>0.65,depthfilm2<1.3)
depthfilm2fit=depthfilm2[mask]#fit
film2fit=np.flip(film2[mask],axis=0)



def linear(x,a,b):
    return a*x+b



poptfilm2, pcov2 = curve_fit(linear, depthfilm2fit, film2fit)
afilm2,bfilm2= poptfilm2
dafilm2,dbfilm2 = np.sqrt(np.diag(pcov2))

print(afilm2,dafilm2)

print(bfilm2,dbfilm2)


directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/20KDKEProfile3/98x98/'
#directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-11-05/20EQ/'

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
#filename_25='Doseof-25.tif'
#filename_26='Doseof-26.tif'
#filename_27='Doseof-27.tif'
#filename_28='Doseof-28.tif'
#filename_29='Doseof-29.tif'
#filename_30='Doseof-30.tif'


file_path =[

                directory+filename_1,
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
                #directory+filename_25,
                #directory+filename_26,
                #directory+filename_27,
                #directory+filename_28,
                #directory+filename_29,
                #directory+filename_30
                ]


rcf_matrix = RCF_matrix_producer(file_path).produce_stack()
file_name_RCF='RCFmatrix20KDKE'
np.save(directory+file_name_RCF,rcf_matrix)
(RCF,rcfname)= file_name_RCF.split('matrix')

mini_directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/'
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-08-13/'
#directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-11-05/lowdose/'

file_name_miniSCIDOM ='rec_light_dist_shot93.npy'
(rec,light,dist,shotnumber)= file_name_miniSCIDOM.split('_')
(shotnumber,extension)=shotnumber.split('.npy')
(shot,number)=shotnumber.split('t')
path_miniSCIDOM=mini_directory+file_name_miniSCIDOM
miniscidom_matrix = np.load(path_miniSCIDOM)
#print(np.shape(miniscidom_matrix))

#rcf_matrix = np.load(path_RCF)
miniscidom_pixel_size =0.0634
m=miniscidom_pixel_size
rcf_pixel_size_lat = 0.0847 #mm/px
rcf_pixel_size_dep = 0.28*1.3 #mm/px
#wepl=4.914 # mm
wepl=0.3 # mm
rcf_matrix=np.flip(rcf_matrix,axis=(2))

matrix = RCF_rebinner(    miniscidom_matrix[:,:,:],
                      miniscidom_pixel_size,
                        rcf_matrix[:,:,:],
                         rcf_pixel_size_lat,
                         rcf_pixel_size_dep ).zoom_matrix

matrix =scipy.ndimage.shift(matrix,[0,0,0])
#matrix[matrix<10**(-6)]=0

np.save(directory+file_name_RCF+'_rebinned',matrix)


#print(matrix)

Tomograpic_viewer(matrix/matrix.max(),False,1,m)
Tomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,m)
difference = np.abs(matrix/matrix.max()-miniscidom_matrix/miniscidom_matrix.max())
#Tomograpic_viewer(difference,False,1,m)
differencepercentage=np.divide(difference,matrix/matrix.max(),out=np.zeros_like(matrix),where=matrix!=0)*100
#Tomograpic_viewer(differencepercentage,False,20,m)


twoDTomograpic_viewer(matrix/matrix.max(),False,1,miniscidom_pixel_size,wepl)

difference = np.abs(matrix/matrix.max()-miniscidom_matrix/miniscidom_matrix.max())
#Tomograpic_viewer(difference,False,1,m)


#X profile
miniprofile=twoDTomograpic_viewer(miniscidom_matrix/miniscidom_matrix.max(),False,1,miniscidom_pixel_size,wepl)


mini_xmeanprofile=miniprofile.xmeanprofile
mini_ymeanprofile=miniprofile.ymeanprofile
RCFprofile=twoDTomograpic_viewer(matrix/matrix.max(),False,1,miniscidom_pixel_size,wepl)
RCF_xmeanprofile=RCFprofile.xmeanprofile
RCF_ymeanprofile=RCFprofile.ymeanprofile

x=RCF_xmeanprofile
y=RCF_ymeanprofile
RCF_xmeanprofile=RCF_xmeanprofile/RCF_xmeanprofile.max()
mini_xmeanprofile=mini_xmeanprofile/mini_xmeanprofile.max()
RCF_ymeanprofile=RCF_ymeanprofile/RCF_ymeanprofile.max()
mini_ymeanprofile=mini_ymeanprofile/mini_ymeanprofile.max()




#########################################################################################################FIT
#we need to mask the array to fit just the first part of the two curves


x1=np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size

mask1=np.logical_and(x1>8,x1<8.5)
x1fit=x1[mask1]#fit
y1fit=RCF_xmeanprofile[mask1]


k=RCF_xmeanprofile.max()/mini_xmeanprofile.max()
x2=np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size
mask2=np.logical_and(x2>7.9,x2<8.8)
x2fit=x2[mask2]#fit
y2fit=mini_xmeanprofile[mask2]

param_bounds=([0,0],[np.inf,np.inf])
popt = (1,0) #initial values
popt1, pcov1 = curve_fit(linear, x1fit, y1fit)#, p0=popt,bounds=param_bounds)
a1,b1= popt1
da1,db1 = np.sqrt(np.diag(pcov1))
print(a1,da1)
print(b1,db1)
popt2, pcov2 = curve_fit(linear, x2fit, y2fit)#, p0=popt,bounds=param_bounds)
a2,b2= popt2
da2,db2 = np.sqrt(np.diag(pcov2))
print(a2,da2)
print(b2,db2)
#############################################################################################################
####PLOT
 #normalization

plt.figure(6)
plt.rc('text', usetex=True)
"""
plt.plot(depthfilm1+0.1,film1,'.',color='orange',markersize=13,label=r'$D_{Film1}$')
plt.plot(depthfilm1+0.1,depthfilm1+0.1*poptfilm1[0]+poptfilm1[1],
                                                                            '-',
                                                                  color='orange',
                                                                  linewidth=3.0,
                                                        label=r'$D_{Film1}$linear fit',
                                                                        zorder=2)
"""

#plt.scatter(depthfilm2fit+0.2,film2fit,marker='^',color='pink',
#                                                                        zorder=1)

plt.plot(depthfilm2,film2,
                                                                  '.',color='hotpink',
                                             markersize=13,label=r'$D_{RCF,P2}$')

plt.plot(depthfilm2+0.15,depthfilm2*poptfilm2[0]+poptfilm2[1],
                                                                            '-',
                                                                      color='lightpink',
                                                                  linewidth=5.0,
                                                  label=r'$D_{RCF,P2}$  linear fit',
                                                                       zorder=2)

plt.plot((np.arange(0,matrix.shape[2]+1)*miniscidom_pixel_size),
                                                                RCF_xmeanprofile,
                                                                             '.',
                                                                    Markersize=13,
                                            color=sns.color_palette(  "Paired")[1],
                                                             label=r' $D_{RCF,P1}$',
                                                                      zorder=10)



#plt.scatter(x1fit,y1fit,marker='^',color=sns.color_palette(  "Paired")[0],
#                                                                        zorder=1)
plt.plot(x1fit,x1fit*popt1[0]+popt1[1],
                                                                            '-',
                                            color=sns.color_palette(  "Paired")[0],
                                                                  linewidth=5.0,
                                                    label=r'$D_{RCF,P1}$  linear fit',
                                                                        zorder=2)
#plt.hlines(0.2, -0.1,10, colors='darkorange', linestyles='--', linewidth=3)
plt.hlines(0.1, -0.1,10, colors='darkorange', linestyles='--',linewidth=3,label='background level')

plt.plot(np.arange(0,miniscidom_matrix.shape[2]+1)*miniscidom_pixel_size,
                                                                mini_xmeanprofile,
                                                                             '.',
                                            color=sns.color_palette("Paired")[3],
                                                                   Markersize=13,
                                                               label=r'$D_{MS}$',
                                                                      zorder=10)


plt.scatter(x2fit,y2fit,
                                                                        alpha=1,
                                                                    marker='^',
                                          color=sns.color_palette(  "Paired")[2],
                                                                        zorder=1)
plt.plot(x2fit,(x2fit*popt2[0]+popt2[1]),
                                                                            '-',
                                          color=sns.color_palette(  "Paired")[2],
                                                                   linewidth=5.0,
                                                    label=r'$D_{MS}$ linear fit',
                                                                        zorder=1)
plt.rc('text', usetex=False)
#plt.xlim([0.765,9])
plt.xlim([-0.1,10])
plt.ylim([-0.1,1.1])
plt.title('Xprofile wepl={}mm '.format(wepl),fontsize=24)
plt.xlabel('x[mm]',fontsize=22)
plt.ylabel(' Relative Intensity',fontsize=22)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)

theta1=np.arctan(a1)
dtheta1=np.arctan(da1)
theta2=np.arctan(a2)
thetafilm2=np.arctan(afilm2)
dtheta2=np.arctan(da2)
dthetafilm2=np.arctan(dafilm2)
plt.rc('text', usetex=True)
plt.legend( title=r'Linear model estimation: $\theta_{RCF,P1}$'f'= {theta1:.2f} $\pm$ {dtheta1:.2f}'r' ,\\ $\theta_{MS} $ 'f'= {theta2:.2f} $\pm$ {dtheta2:.2f} 'r' , $\theta_{RCF,P2}$ 'f'= {thetafilm2:.2f} $\pm$ {dthetafilm2:.2f} ',
                                                                   fontsize='18',
                                                                   markerscale=3,
                                                               title_fontsize=22,
                                                                           loc=8)
plt.rc('text', usetex=False)


#############################################################################
"""
####Y PROFILE
popt1, pcov1=[0,0]
x1=np.arange(0,matrix.shape[1]+1)*miniscidom_pixel_size
#mask1=np.logical_and(x1>0.5,x1<1.05)
mask1=np.logical_and(x1>1.2,x1<2)
X1fit=x1[mask1]#fit
Y1fit=RCF_ymeanprofile[mask1]

k=RCF_ymeanprofile.max()/mini_ymeanprofile.max()
x2=np.arange(0,miniscidom_matrix.shape[1]+1)*miniscidom_pixel_size
#mask2=np.logical_and(x2>0.5,x2<1.05)
mask2=np.logical_and(x2>0.7,x2<2)
X2fit=x2[mask2]#fit
Y2fit=mini_ymeanprofile[mask2]


def linear(x,a,b):
    return a*x+b

param_bounds=([0,0],[np.inf,np.inf])
popt = (1,0) #initial values
popt1, pcov1 = curve_fit(linear, X1fit, Y1fit)#, p0=popt,bounds=param_bounds)
a1,b1= popt1
da1,db1 = np.sqrt(np.diag(pcov1))
popt2, pcov2 = curve_fit(linear, X2fit, Y2fit)#, p0=popt,bounds=param_bounds)
a2,b2= popt2
da2,db2 = np.sqrt(np.diag(pcov2))



plt.figure(7)

plt.plot((np.arange(0,matrix.shape[1]+1)*miniscidom_pixel_size),
                                                                RCF_ymeanprofile,
                                                                             '.',
                                                                    Markersize=11,
                                  label='RCF{} Accumulated dose'.format(rcfname))

plt.scatter(X1fit,Y1fit,marker='^',
                                          color=sns.color_palette(  "Paired")[0],
                                                                        zorder=1)

plt.plot(X1fit,X1fit*popt1[0]+popt1[1],
                                                                            '-',
                                            color=sns.color_palette(  "Paired")[0],
                                                                  linewidth=7.0,
                                                          label='RCF linear fit',
                                                                       zorder=2)


plt.plot(np.arange(0,miniscidom_matrix.shape[1]+1)*miniscidom_pixel_size,
                                        mini_ymeanprofile,
                                                                             '.',
                                            color=sns.color_palette(  "Paired")[3],
                                                                   Markersize=11,
                                  label='$D_{MS}')

plt.hlines(0.05, -0.1,10, colors='darkorange', linestyles='--',
                                            linewidth=3,label='background level')

plt.scatter(X2fit,Y2fit,
                                                                        alpha=1,
                                                                    marker='^',
                                        color=sns.color_palette(  "Paired")[2],

                                                                        zorder=1)
plt.plot(X2fit,(X2fit*popt2[0]+popt2[1]),
                                                                            '-',
                                        color=sns.color_palette(  "Paired")[2],
                                                                 linewidth=7.0,
                                                              label='Minisicdom linear fit',
                                                              zorder=1)

theta1=np.arctan(a1)
dtheta1=np.arctan(da1)
theta2=np.arctan(a2)
dtheta2=np.arctan(da2)
plt.rc('text', usetex=True)
plt.legend( title=r'Linear model estimation: $\theta_1$'f'= {theta1:.2f} $\pm$ {dtheta1:.2f}
                       'r',\\$\theta_2 $ 'f'= {theta2:.2f} $\pm$ {dtheta2:.2f} ',
                                                               fontsize='18',
                                                               markerscale=3,
                                                               title_fontsize=22,
                                                               loc=8)
plt.rc('text', usetex=False)
plt.xlim([-0.1,10])
plt.ylim([-0.1,1.1])
plt.title('Yprofile wepl={}mm '.format(wepl),fontsize=24)
plt.xlabel('mm',fontsize=18)
plt.ylabel(' Relative',fontsize=18)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)




"""
plt.show()
