""" TOMOGRAPHIC VIEWER OF RCF"""
from pathlib import Path
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import pytesseract
from RCF_snippets1 import *
import re
from tomograpic_viewer import Tomograpic_viewer
import scipy.misc
from scipy import ndimage

#datestring = '2020_08_16'
datestring ='2021-08-13'

foldername = '85O1'
(year, month, day) = datestring.split('-')
filenamepattern = '*.tif' # Set here specific filename filter

databasedirectory=Path('/home/corvin22/Desktop/miniscidom/pictures')
datadirectory = databasedirectory/datestring/foldername
filelist = list(datadirectory.glob(filenamepattern))
number_of_files = len(filelist)


if number_of_files==0:
    print('No files found in '+datadirectory.name)
    quit() # if no files are present, stop the script
else:
    print(number_of_files, 'images can be analysed in ' + str(datadirectory.resolve())) # display number of relevant images




Labels=[]
StackColumns=['Label','Image']
Stack = pd.DataFrame(columns=StackColumns)
for file in filelist:
    (dose,label)=file.name.split('-')# make ROI a parameter when geometry data is available
    #(dose,label)=file.name.split('Dose of 85P2film')

    Stack=Stack.append(pd.DataFrame(\
                                   [[label,file]],columns=StackColumns)\
                      ,ignore_index=True)
Stack.sort_values(by=['Label'],inplace=True,ignore_index=True)



for i in Stack.index:

    img=np.flip(cv2.imread((str(Stack.Image[i].resolve())),cv2.IMREAD_ANYDEPTH),axis=0)


    if i==0:
        Matrix = np.zeros((len(Stack.index),np.shape(img)[0],np.shape(img)[1]))

    Matrix[i,:,:]=img

print(Matrix)
print(np.shape(Matrix))

save_directory='/home/corvin22/Desktop/miniscidom/pictures/2021-08-13/'

np.save(save_directory+'RCFmatrix85O1',Matrix)



Tomograpic_viewer(Matrix/Matrix.max(),False,1)

plt.fig, (plt.ax1,plt.ax2,plt.ax3) = plt.subplots(1, 3,figsize=(12,3))
plt.ax1.set_title('x profile')
plt.ax1.set_ylabel('Layer')
plt.ax1.imshow( ndimage.rotate(Matrix[:,45,:], 0),cmap='jet',aspect="equal")


plt.ax2.set_title('y profile')
plt.ax2.set_ylabel('Layer')
plt.ax2.imshow(ndimage.rotate(Matrix[45,:,:],0),cmap='jet',aspect="equal")



plt.ax3.set_title('z profile Layer 12')
plt.ax3.set_xlabel('x in pixel')
plt.ax3.set_ylabel('y in pixel')
im_z=plt.ax3.imshow(Matrix[12,:,:],cmap='jet',aspect="equal")
#im_z=plt.ax3.colorbar(cmap='jet',label='Dose[Gy]')
plt.colorbar(im_z,label='Dose[Gy]')
plt.show()
 #plt.close()



"""
#StackColumns=['Layer','Image']
#Matrix = pd.DataFrame(columns=StackColumns)
Matrix=[]
for i in Stack.index:

    img=cv2.imread(str(Stack.Image[i].resolve()))

     #90 x 90x3
    #centralcolumn=img[:,45,:]  #90 X3
    #centralrow=img[45,:,:]

    #Matrix=Matrix.append(pd.DataFrame(\
     #                              [[i,img[:,45,:]]],columns=StackColumns)\
     #                 ,ignore_index=True)
    Matrix=Matrix.append(img[:,45,:])
    Matrix=np.array(Matrix)
    #plt.rcParams.update({'figure.max_open_warning': 0})
    #plt.figure(figsize=(10,8))
    #plt.title(Stack.Label[i])
    #plt.imshow(img)
    #plt.show()


#print(Matrix.Image.shape)
print(Matrix)
#imgmatrix=cv2.imread(str(Matrix.Image))

plt.show()
 #plt.close()
"""
