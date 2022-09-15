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
from calibrationfactor import calibrationfactor
from correction_matrix_producer_2 import Correction_matrix_producer
#######################For-real-camera-pictures#######################################
#Directory measured data

directory='pictures/2020-11-05/lowdose/'
picture_name='42.png'
#directory='pictures/spatialresolution/'
#picture_name='60mGy.png'
#picture_name='obliquel100ms100mgplusother_2020-10-08.png'

#Define path for measured pictures
path_pic_front=directory+picture_name
path_pic_top=directory+picture_name
path_pic_120=directory+picture_name
path_pic_240=directory+picture_name



deltaz=159 #Draco measurements
s=0.0634
shape_front=(170,152) #you can not reconstruct somethin bigger than 161
shape_side=(159,152)

rot_angle=0


x_0=int(-20)
y_0=int(-58)

y1_front=y_0+461
y2_front=y_0+631
x1_front=x_0+453
x2_front=x_0+605

y1_top=y_0+697
y2_top=y_0+856
x1_top=x_0+451
x2_top=x_0+603

y1_120=y_0+694
y2_120=y_0+853
x1_120=x_0+210
x2_120=x_0+362

y1_240=y_0+694
y2_240=y_0+853
x1_240=x_0+693
x2_240=x_0+845
"""


x_0=int(-170)
y_0=int(-56)

y1_front=y_0+461
y2_front=y_0+631
x1_front=x_0+353
x2_front=x_0+505

y1_top=y_0+700
y2_top=y_0+859
x1_top=x_0+351
x2_top=x_0+503

y1_120=y_0+697
y2_120=y_0+856
x1_120=x_0+170
x2_120=x_0+322

y1_240=y_0+697
y2_240=y_0+856
x1_240=x_0+593
x2_240=x_0+745


shift_image_front =[0,0] #do not change: values different from 0 cause artifacts
shift_image_top = [0,0]
shift_image_120 = [0,0]
shift_image_240 = [0,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[0,0,0,0]
mask_border_240=[0,0,0,0]


"""
###shift
shift_image_front =[0,0] #do not change: values different from 0 cause artifacts
shift_image_top = [0,0]
shift_image_120 = [-4,0]
shift_image_240 = [4,0]




mask_border_front=[0,0,0,0]
mask_border_top=[0,0,0,0]
mask_border_120=[5,0,0,0]
mask_border_240=[0,5,0,0]





#Load images
color_channel='grey'
cam_pic_front=Image_loader(path_pic_front,color_channel,None,None,rot_angle,[y1_front,y2_front,x1_front,x2_front],None,None,shift_image_front,mask_border_front,False,10/9).image  #DELTAX,DELTAY SHIFTING
cam_pic_top=Image_loader(path_pic_top,color_channel,None,None,rot_angle,[y1_top,y2_top,x1_top,x2_top],None,None,shift_image_top,mask_border_top,False,1).image
cam_pic_120=Image_loader(path_pic_120,color_channel,None,1,rot_angle,[y1_120,y2_120,x1_120,x2_120],None,None,shift_image_120,mask_border_120,False,10/9).image
cam_pic_240=Image_loader(path_pic_240,color_channel,None,1,rot_angle,[y1_240,y2_240,x1_240,x2_240],None,None,shift_image_240,mask_border_240,False,10/9).image




Camera_picture_viewer(cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,False,14)

##########################################################Load correction_matrix
#correction_matrix=np.load('correction_matrix_Oncoray.npy')
#correction_matrix=np.load('correction_matrix.npy')
#correction_matrix=np.load('correction_matrix_XRay.npy')

#Tomograpic_viewer(correction_matrix/correction_matrix.max(),False,1) # here you can enable the logaritmic scale

                                                                                                                                                                          #
correction_matrix = Correction_matrix_producer(      shape_front,
                                                          shape_side,
                                                     shift_image_top,
                                                     shift_image_120,
                                                     shift_image_240,
                                                   shift_image_front,
                                                    mask_border_front,
                                                    mask_border_top,
                                                    mask_border_120,
                                                    mask_border_240).correction_matrix

#correction_matrix=scipy.ndimage.gaussian_filter(correction_matrix,sigma=1.5) #xrays
#correction_matrix = np.ones(np.shape(correction_matrix))
                                                                                    #
Tomograpic_viewer(correction_matrix,False,4,s)







#######################################directory for saving reconstruction file
save_directory='pictures/2020-11-05/lowdose/'
#####################################################################################



#######################################################Subtracting the backgroung


cam_pic_front[cam_pic_front<0]=0
cam_pic_top[cam_pic_top<0]=0
cam_pic_120[cam_pic_120<0]=0
cam_pic_240[cam_pic_240<0]=0




############################################################Show camera pictures
smin=Camera_picture_viewer(cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,False,16).sminval # here you can enable the logaritmic scale

cam_pic_front[cam_pic_front<smin]=0
cam_pic_top[cam_pic_top<smin]=0
cam_pic_120[cam_pic_120<smin]=0
cam_pic_240[cam_pic_240<smin]=0
#Camera_picture_viewer(cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,False,16)








#Camera_picture_viewer(cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,False,16)

############################################################number of iterations
max_it=5

#################################################################Reconstruction
reconstructor=Reconstructor(max_it,cam_pic_front,cam_pic_top,cam_pic_120,cam_pic_240,correction_matrix,True) #the last one is the median filter
reconstructor.perform_MLEM()
rec_light_dist=reconstructor.rec_light_dist
rec_light_dist=rec_light_dist.astype(int)
#save_directory='pictures/2020-09-25/centered-mousesetting/'
save_directory='pictures/2020-11-05/lowdose/'
np.save(save_directory+'rec_light_dist_shot60mGy',rec_light_dist)

print(rec_light_dist)
print(np.shape(rec_light_dist))
#Show reconstruction
Tomograpic_viewer(rec_light_dist/rec_light_dist.max(),False,1,s) # here you can enable the logaritmic scale



#rec_light_dist[rec_light_dist<90]=0
dist_3D=np.sum(rec_light_dist,axis=(1,2))

dist_3D_mask=np.zeros(np.shape(rec_light_dist))


"""MASK"""
def create_circular_mask(h, w, center=None, radius=None):

   if center is None: # use the middle of the image
      center = (int(w/2), int(h/2))
   if radius is None: # use the smallest distance between the center and image walls
      radius = min(center[0], center[1], w-center[0], h-center[1])

   YY,XX =np.ogrid[:h,:w]
   dist_from_center = np.sqrt((XX - center[0])**2 + (YY-center[1])**2)

   mask = dist_from_center <= radius
   return mask


mean_array=np.zeros((deltaz))
std=np.zeros((deltaz))
err=np.zeros((deltaz))
for i in range(len(mean_array)):
    img=rec_light_dist[i,:,:]
    h,w = img.shape[:2]


    innerradius=  40 #5mm inner circle
    #innerradius=55 #7mm
    mask = create_circular_mask(h,w,radius=innerradius) #create a boolean circle mask
    dist_3D_mask[i,mask] = img[mask]
    mean_array[i]=np.mean(img[mask],axis=0)
    std[i]=np.std(img[mask],axis=0)
    err[i]=std[i]/np.sqrt(len(img[mask]))




Tomograpic_viewer(dist_3D_mask/dist_3D_mask.max(),False,1,s) # here you can enable the logaritmic scale
"""

plt.fill_between(np.arange(0,len(mean_array),1)*0.0634,
                                                          mean_array-err,
                                                         mean_array + err,
                                                        color='gray', alpha=0.5)
plt.errorbar(  np.arange(0,len(mean_array),1)*0.0634,
                                                               mean_array,
                                                                      yerr=err,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    ecolor=None,
                                                                elinewidth=None)

plt.errorbar(  np.arange(0,len(mean_array),1)*0.0634,mean_array ,
                                                                       yerr=err,
                                                                      xerr=None,
                                                                         fmt='',
                                                                    ecolor=None,
                                                                elinewidth=None)
plt.title('depth dose distribution mean value mask image', fontdict=None,
                                                                   loc='center',
                                                           pad=None,fontsize=18)
plt.xlabel('Depth[mm]',fontsize=16)
plt.ylabel('Intensity',fontsize=16)

plt.show()
"""

#Append Tof information to mean array

#TOF=[434,114,3.82] #24/09 n.84
#TOF=[300,100,3] #11-05 n1385
TOF=[0,0,0]
mean_array=np.append(mean_array,TOF)

#directory for saving mean dose array

save_directory='pictures/2020-11-05/lowdose/notnormalized/'
#save_directory='pictures/spatialresolution/'
#####################################################################################
np.save(save_directory+'notnormalizedmean_array'+'60mGy',mean_array)
np.save(save_directory+'notnormalizederr'+'60mGy',err)
np.save(save_directory+'notnormalizedmean_array_notmasked'+'60mGy',dist_3D)

#############################################################################################

dist_1D_120=np.flip(np.sum(cam_pic_120,axis=(1)))

np.save(save_directory+'dist_1D_120_'+'60mGy',dist_1D_120)

plt.plot(np.arange(0,deltaz,1)*s,dist_1D_120/np.max(dist_1D_120),'.',label='120 ')
dist_1D_240=np.flip(np.sum(cam_pic_240,axis=(1)))
np.save(save_directory+'dist_1D_240_'+'60mGy',dist_1D_240)
plt.plot(np.arange(0,deltaz,1)*s,dist_1D_240/np.max(dist_1D_240),'.',label=' 240')
dist_1D_top=np.flip(np.sum(cam_pic_top,axis=(1)))
np.save(save_directory+'dist_1D_top_'+'60mGy',dist_1D_top)
plt.plot((np.arange(0,deltaz,1)*s),dist_1D_top/np.max(dist_1D_top),'.',label='top')
plt.plot(np.arange(0,deltaz,1)*s,dist_3D/np.max(dist_3D),'.',label='3D reconstruction')
mean_array=mean_array[:-3] #eliminate the TOF data
plt.errorbar(  np.arange(0,len(mean_array),1)*s,
                                             mean_array/np.max(mean_array),
                                                    yerr=err/np.max(mean_array),
                                                                      xerr=None,
                                                                         fmt='',
                                                                    ecolor=None,
                                                                elinewidth=None,
                                                        label='3D masked image')




plt.title('Depth dose distribution 1D  projection vs 3D reconstruction ',
                                                                  fontdict=None,
                                                                   loc='center',
                                                                    fontsize=22)
plt.legend(fontsize='large')
plt.xlabel('Depth[mm]',fontsize=18)
plt.ylabel('Intensity',fontsize=18)

plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.show()

save_directory='pictures/2020-11-05/lowdose/notnormalized/1Dtopprojection/'
#####################################################################################
#np.save(save_directory+'top1Dmean_array'+'4',dist_1D_top)



"""

"LET CORRECTION OF 3D DATA IN Depth"
LETdirectory='pictures/2020-09-25/centered-mousesetting/TOFinscintillator_4_ana.csv'
calib=calibrationfactor(mean_array,LETdirectory)
calib_value=calib.c_mean
lightcorrection_value=calib.lightcorrection
depth_sci=calib.depth
dose_sci=calib.dose_sci
matrix_3D_corrected=np.zeros(np.shape(rec_light_dist)) #((deltaz, 186, 152) z is the first =deltaz




for i in range(np.shape(rec_light_dist)[0]): #from 0 to 158
    matrix_3D_corrected[i,:,:]=rec_light_dist[i,:,:]/lightcorrection_value[i]

matrix_3D_corrected_calib=matrix_3D_corrected*calib_value
print(matrix_3D_corrected_calib.max())
Tomograpic_viewer(matrix_3D_corrected_calib,False,0.834) # here you can enable the logaritmic scale




#dist_3D=np.flip(np.mean(rec_light_dist,axis=(1,2)))
#dist_3D_corrected=np.flip(np.mean(matrix_3D_corrected,axis=(1,2)))
dist_3D_corrected_calib=(np.mean(matrix_3D_corrected_calib,axis=(1,2)))
plt.plot(np.arange(0,deltaz,1)*0.0634,dist_3D_corrected_calib/np.max(dist_3D_corrected_calib),'.',label='3D reconstruction')
#area_corrected_calib=np.trapz(dist_3D_corrected_calib, np.arange(0,len(dose),1)*0.0634)
#norm_corrected_calib=(area_sci/area_corrected_calib) #area normalization
plt.show()


#2D Projection

#Show reconstruction
#twoDTomograpic_viewer(rec_light_dist,False,16) # here you can enable the logaritmic scale
"""
