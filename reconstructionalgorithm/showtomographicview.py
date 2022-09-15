import numpy as np
from tomograpic_viewer import Tomograpic_viewer
path='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/rec_light_dist_shot1.npy'
rec_light_dist = np.load(path)
s= 0.073825
deltaz=140
Tomograpic_viewer(rec_light_dist/rec_light_dist.max(),False,1,s)


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


    innerradius= 34  #5mm diamter inner circle
    mask = create_circular_mask(h,w,radius=innerradius) #create a boolean circle mask
    dist_3D_mask[i,mask] = img[mask]
    mean_array[i]=np.mean(img[mask],axis=0)
    std[i]=np.std(img[mask],axis=0)
    err[i]=std[i]/np.sqrt(len(img[mask]))


Tomograpic_viewer(dist_3D_mask/dist_3D_mask.max(),False,1,s) # here you can enable the logaritmic scale
