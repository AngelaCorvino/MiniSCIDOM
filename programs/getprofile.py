import scipy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from scipy import interpolate
#the program is written thinking about dose values but it can be used for energy, let , whatever value we want to score

def get_x_profile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_profile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues




class Get_profile(object):

    #Initialization of class
    def __init__(self,data, z_binnumber,x_binnumber):

        #3D data matrix
        self.data = data
        self.z_binnumber= z_binnumber
        self.x_binnumber=x_binnumber

        self.xprofiles = np.zeros((self.z_binnumber, self.x_binnumber)) # create empty array to be filled
        self.zprofiles = np.zeros((self.x_binnumber, self.z_binnumber)) # create empty array to be filled
        #zprofiles_energy = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled






        # write each x- profile at depth z in row i in the array outside the loop

        for i in range(self.z_binnumber):
            self.xprofile_at_z  = get_x_profile_at_z(self.data, i)
            self.xprofiles[i,:] = self.xprofile_at_z
            i += 1

        self.xmeanprofile=np.sum(self.xprofiles,axis=0)/self.z_binnumber



        for i in range(self.x_binnumber):
            self.zprofile_at_x  = get_z_profile_at_x(self.data, i)
            self.zprofiles[i,:] = self.zprofile_at_x
            i += 1

        self.zmeanprofile=np.sum(self.zprofiles,axis=0)/self.x_binnumber
