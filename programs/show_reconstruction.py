import numpy as np
from tomograpic_viewer import Tomograpic_viewer


#directory with reconstructed file
#directory='pictures/spatialresolution/'
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/'
#directory='/Volumes/ANGELA/miniscidom/pictures/2020-11-05/highdose/600micronhole/'

#load file
rec_light_dist=np.load(directory+'rec_light_dist_shot91.npy')

#show reconstruction
#rec_light_dist=np.flip(rec_light_dist,axis=0)
Tomograpic_viewer(rec_light_dist/rec_light_dist.max(),False,1,0.0634)
