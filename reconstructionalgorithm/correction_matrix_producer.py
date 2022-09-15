import numpy as np
from camera_picture_viewer import Camera_picture_viewer
from image_loader import Image_loader
from tomograpic_viewer import Tomograpic_viewer
import scipy.ndimage
import cv2



shape_front=(186,152) #you can not reconstruct somethin bigger than 161
shape_top=(159,152)

rec_part_front=np.stack([np.ones(shape_front)]*shape_top[0],axis=0)
rec_part_top=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_120=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_240=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)

rec_part_120=scipy.ndimage.rotate(rec_part_120,120,axes=(1, 2),reshape=False,prefilter=True)
rec_part_240=scipy.ndimage.rotate(rec_part_240,240,axes=(1, 2),reshape=False,prefilter=True)

correction_matrix=rec_part_front+rec_part_top+rec_part_120+rec_part_240

#Perform cubic median filter
#correction_matrix=np.divide(1,np.round(correction_matrix,decimals=0),out=np.zeros_like(np.round(correction_matrix,decimals=0)),where=correction_matrix!=0)

#correction_matrix[np.logical_and(0<correction_matrix,correction_matrix<=0.5)]=0
#correction_matrix[np.logical_and(0.5<correction_matrix,correction_matrix<=1.5)]=1
#correction_matrix[np.logical_and(1.5<correction_matrix,correction_matrix<=2.5)]=2
#correction_matrix[np.logical_and(2.5<correction_matrix,correction_matrix<=3.5)]=3
#correction_matrix[np.logical_and(3.5<correction_matrix,correction_matrix<=4.5)]=4


#Perform cubic median filter
#for i in range(5):
#     correction_matrix=scipy.ndimage.median_filter(correction_matrix, size=3)


np.save('correction_matrix.npy',correction_matrix)

#Show reconstruction
#Tomograpic_viewer(correction_matrix,False,4)












shape_front=(210,180) #you can not reconstruct somethin bigger than 190
shape_top=(202,180)

rec_part_front=np.stack([np.ones(shape_front)]*shape_top[0],axis=0)
rec_part_top=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_120=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_240=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)

rec_part_120=scipy.ndimage.rotate(rec_part_120,120,axes=(1, 2),reshape=False,prefilter=True)
rec_part_240=scipy.ndimage.rotate(rec_part_240,240,axes=(1, 2),reshape=False,prefilter=True)

correction_matrix=rec_part_front+rec_part_top+rec_part_120+rec_part_240

#Perform cubic median filter
#correction_matrix=np.divide(1,np.round(correction_matrix,decimals=0),out=np.zeros_like(np.round(correction_matrix,decimals=0)),where=correction_matrix!=0)

#correction_matrix[np.logical_and(0<correction_matrix,correction_matrix<=0.5)]=0
#correction_matrix[np.logical_and(0.5<correction_matrix,correction_matrix<=1.5)]=1
#correction_matrix[np.logical_and(1.5<correction_matrix,correction_matrix<=2.5)]=2
#correction_matrix[np.logical_and(2.5<correction_matrix,correction_matrix<=3.5)]=3
#correction_matrix[np.logical_and(3.5<correction_matrix,correction_matrix<=4.5)]=4


#Perform cubic median filter
#for i in range(5):
#     correction_matrix=scipy.ndimage.median_filter(correction_matrix, size=3)


np.save('correction_matrix_XRay.npy',correction_matrix)

#Show reconstruction
#Tomograpic_viewer(correction_matrix,False,4)


















shape_front=(170,130) #you can not reconstruct somethin bigger than 161
shape_top=(140,130)

rec_part_front=np.stack([np.ones(shape_front)]*shape_top[0],axis=0)
rec_part_top=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_120=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)
rec_part_240=np.stack([np.ones(shape_top)]*shape_front[0],axis=1)

rec_part_120=scipy.ndimage.rotate(rec_part_120,120,axes=(1, 2),reshape=False,prefilter=True)
rec_part_240=scipy.ndimage.rotate(rec_part_240,240,axes=(1, 2),reshape=False,prefilter=True)

correction_matrix=rec_part_front+rec_part_top+rec_part_120+rec_part_240

#Perform cubic median filter
#correction_matrix=np.divide(1,np.round(correction_matrix,decimals=0),out=np.zeros_like(np.round(correction_matrix,decimals=0)),where=correction_matrix!=0)

#correction_matrix[np.logical_and(0<correction_matrix,correction_matrix<=0.5)]=0
#correction_matrix[np.logical_and(0.5<correction_matrix,correction_matrix<=1.5)]=1
#correction_matrix[np.logical_and(1.5<correction_matrix,correction_matrix<=2.5)]=2
#correction_matrix[np.logical_and(2.5<correction_matrix,correction_matrix<=3.5)]=3
#correction_matrix[np.logical_and(3.5<correction_matrix,correction_matrix<=4.5)]=4


#Perform cubic median filter
#for i in range(5):
#     correction_matrix=scipy.ndimage.median_filter(correction_matrix, size=3)


np.save('correction_matrix_Oncoray.npy',correction_matrix)

#Show reconstruction
Tomograpic_viewer(correction_matrix,False,4)
