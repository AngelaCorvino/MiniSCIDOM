import numpy as np
import cv2
import matplotlib.image as mpimg
import skimage
from skimage.measure import profile_line
import pandas as pd
import matplotlib
from scipy.optimize import curve_fit
from tomograpic_viewer import Tomograpic_viewer

directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
file_name_miniSCIDOM ='letcorrectedrec_light_dist_shot1.npy'

path_miniSCIDOM=directory+file_name_miniSCIDOM
rec_light_dist = np.load(path_miniSCIDOM)
miniscidom_pixel_size = 0.073825


s= 0.073825   #spatialresolution proton measurements at oncoray
deltaz=140# proton measurements

Tomograpic_viewer(rec_light_dist/rec_light_dist.max(),False,1,miniscidom_pixel_size) # here you can enable the logaritmic scale



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


    innerradius=  34 #5mm inner circle
    #innerradius=55 #7mm
    mask = create_circular_mask(h,w,radius=innerradius) #create a boolean circle mask
    dist_3D_mask[i,mask] = img[mask]
    mean_array[i]=np.mean(img[mask],axis=0)
    std[i]=np.std(img[mask],axis=0)
    err[i]=std[i]/np.sqrt(len(img[mask]))




Tomograpic_viewer(dist_3D_mask/dist_3D_mask.max(),False,1,miniscidom_pixel_size) # here you can enable the logaritmic scale
