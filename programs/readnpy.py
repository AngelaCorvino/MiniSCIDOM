import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





def read_dose(directory,filename,s):
 #data = pd.read_excel (path)
    ''' read reconstruction  data from npy document'''
    data= np.load(directory+filename)

    dose=data[0:len(data)-3]  #this is usually the dose
    depth=np.arange(0,len(dose),1)*s  #m = [mm/pixe]it depends on the camera
    tof=data[len(data)-3:len(data)]
    (dist,shotnumber)= filename.split('_')
    (shotnumber,extension)=shotnumber.split('.npy')
    (array,number)=shotnumber.split('y')
    return dose,depth,tof,number

def read_doserr(directory,filename):
 #data = pd.read_excel (path)
    ''' read reconstruction  data from npy document'''
    data= np.load(directory+filename)
    err=data[0:len(data)]  #this is usually the dose

    return err
