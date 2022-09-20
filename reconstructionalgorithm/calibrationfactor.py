'''
COMPARISON BETWEEN TOF SIMULATED DATA IN SCINTILLATOR  AND SCINTILLATOR MEASUREMENTS
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
from readcsv import read_datarcf
from readcsv import read_data_let
from readcsv import read_data_let_scintillator
from readcsv import read_data_let_mini
from readcsv import read_data_scintillatorsimulateddose
from readcsv import read_data_scintillatorsimulateddose_it
from Birkmodel import lightcorrection
from Birkmodel import dosecorrection

from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
from PIL import Image
import numpy as np
import cv2



class calibrationfactor(object):
    """Class for  correcting 2D slices.

    Parameters
    ----------
    mean_array : type
        Description of parameter `mean_array`.
    directory : type
        Description of parameter `directory`.

    Attributes
    ----------
    mean_array
    directory

    """

    #Initialization of class
    def __init__(self,mean_array,directory):

        #3D data matrix
        self.mean_array = mean_array
        self.directory=directory

        depth,depth_sci,rcfdose,rcferr,dose_sci,dose_sci_upper,dose_sci_lower,ys_ana_mini,ys_ana_upper_mini,ys_ana_lower_mini=read_data_scintillatorsimulateddose(self.directory)

        "LET correction simulations in scintillator"

        dS=0 #theoretical values
        dscintillator= 1.023 #[g/cm^3] scintillator density
        dactivelayer=1.2 #[g/cm^3]
        k=207/10000 #[g/MeV cm^2]
        a=0.9 #scintillator efficiency
        dx= 65 #µm scintillator spatial resolution
        ddx=1 #spatial resolution error
        k=k/dscintillator*10#[micrometers/kev]

        #SIMULATED LET IN Scintillator starting from TOF of a scidom shot

        S_a_mini=np.interp(np.arange(0,len(self.mean_array),1)*0.0634,depth_sci,ys_ana_mini)
        S_a_low_mini=np.interp(np.arange(0,len(self.mean_array),1)*0.0634,depth_sci,ys_ana_lower_mini)
        S_a_up_mini=np.interp(np.arange(0,len(self.mean_array),1)*0.0634,depth_sci,ys_ana_upper_mini)

        P_a_mini=dosecorrection(self.mean_array,S_a_mini,a,k,dx) #pixel mean value corrected with analytical tof simulation

        "CALIBRATION FACTOR "
    #Step 1 : interpolate dosce_sci in order to divide it by P_a_mini
        dose_sci_interp=np.interp(np.arange(0,len(self.mean_array),1)*0.0634,depth_sci,dose_sci)
    #Step 2: divide
        c=dose_sci_interp/P_a_mini #this is an arry (simulate dose in scintillator/ mean pixel value corrected withLET value )

    #Step 3 : average the vector

        self.c_mean=np.mean(c)

        self.lightcorrection=lightcorrection(S_a_mini,a,k,dx)
        self.depth= depth_sci
        self.dose_sci=dose_sci
