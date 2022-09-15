import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import skimage
from skimage.morphology import disk
from skimage.morphology import (erosion, dilation, opening, closing,
                                white_tophat)
#Class for loading and manipulate camera pictures


class Image_loader(object):

    #Initialization of class
    def __init__(self,path,color,rescale,flip,rot,crop,radius_cut_edge,radius_circle,shift,mask_borders,numpy,factor):

        #3D data matrix
        self.path = path
        self.color = color
        self.rescale = rescale
        self.flip = flip
        self.rot = rot
        self.crop = crop
        self.radius_cut_edge = radius_cut_edge
        self.radius_circle = radius_circle
        self.shift = shift
        self.mask_borders = mask_borders
        self.numpy = numpy
        self.factor = factor

        self.width=None
        self.height=None
        self.image = self.image()



    #Function load and manipulated image
    def image(self):

        #read image
        if self.numpy==True:
            im = np.load(self.path,allow_pickle=True)
            im = im.astype('float32') # Cast a float64 image to float32
        else:
            #im = cv2.imread(self.path)
            im = cv2.imread(self.path,-1)
        (self.height, self.width) = im.shape[:2]


        #choose color channel
        if self.color=='red':
           im=im[:,:,2].T
        elif self.color=='green':
           im=im[:,:,1].T
        elif self.color=='blue':
           im=im[:,:,0].T
        elif self.color=='grey':
           im=im.T


        #rescale pixel number
        if self.rescale!=None:
           self.height=int(self.height*self.rescale)
           self.width=int(self.width*self.rescale)
           if self.rescale>1:
               im = cv2.resize(im, (self.height, self.width), interpolation = cv2.INTER_CUBIC)
           if self.rescale<1:
               im = cv2.resize(im, (self.height, self.width), interpolation = cv2.INTER_AREA)


        #rotate image
        if self.rot!=None:
           M = cv2.getRotationMatrix2D((self.height / 2, self.width / 2), self.rot, 1)
           im = cv2.warpAffine(im, M, (self.height, self.width))
        #plt.imshow(im/im.max(),vmin=0,vmax=0.3)
        plt.imshow(im,vmin=0,vmax=0.5*2**16)
        #plt.imshow(im,vmin=0,vmax=2**8) #oncoray
        #plt.imshow(im,vmin=0,vmax=0.3*2**14) #oncoray
        plt.plot([self.crop[2],self.crop[3],self.crop[3],self.crop[2],self.crop[2]],[self.crop[0],self.crop[0],self.crop[1],self.crop[1],self.crop[0]])
        #plt.show()

       #Crop image
        if self.crop!=None:
           height_min=self.crop[0]
           height_max=self.crop[1]
           width_min=self.crop[2]
           width_max=self.crop[3]
           im=im[height_min:height_max,width_min:width_max]
           (self.height, self.width) = im.shape[:2]
        #plt.show()

        #flip image
        if self.flip!=None:
           # flip =  0: Vetical flip
           # flip =  1: Horizontal flip
           # flip = -1: Horizontal and Vetical flip
           if self.flip in [-1,0,1]:
              im = cv2.flip(im, self.flip)







        #Cut edge
        if self.radius_cut_edge!=None:

           for i in range(self.radius_cut_edge):
                for j in range(self.radius_cut_edge):
                    if self.radius_cut_edge**2<(self.radius_cut_edge-j)**2+(self.radius_cut_edge-i)**2:
                        if i<=self.radius_cut_edge and j<=self.radius_cut_edge:
                            #Edge top left
                            im[:i,j-1]=0
                            #Edge top right
                            im[:i,self.width-j-1]=0
                            #Edge lower left
                            im[-1-i:,j]=0
                            #Edge lower right
                            im[-1-i:,self.width-j-1]=0



        #Cut circle
        if self.radius_circle!=None:
           for i in range(self.height):
                for j in range(self.width):
                    if self.radius_circle**2<(int(self.width/2-j))**2+(int(self.height/2-i))**2:
                        im[i,j]=0


        #Shift image
        if self.shift!=None:
           # Create translation matrix.
           # If the shift is (x, y) then matrix would be
           # M = [1 0 x]
           #     [0 1 y]

           if self.shift[0]<0:
              #print('oK1')

              im=np.pad(im,((0,0),(abs(self.shift[0]),0)),mode='constant',constant_values=(0))
              (self.height, self.width) = im.shape[:2]
              im=im[:,:-abs(self.shift[0])]
           elif self.shift[0]>0:
              #print('oK2')

              im=np.pad(im,((0,0),(0,abs(self.shift[0]))),mode='constant',constant_values=(0))
              im=im[:,abs(self.shift[0]):]
              (self.height, self.width) = im.shape[:2]
           if self.shift[1]<0:
              #print('oK3')
              im=np.pad(im,((abs(self.shift[1]),0),(0,0)),mode='constant',constant_values=(0))
              #im=im[abs(self.shift[1]):,]
              im=im[:-abs(self.shift[1]),:]

              (self.height, self.width) = im.shape[:2]
           elif self.shift[1]>0:
              #print('oK4')
              im=np.pad(im,((0,abs(self.shift[1])),(0,0)),mode='constant',constant_values=(0))
              im=im[abs(self.shift[1]):,]

              #im=im[:-abs(self.shift[1]),:]
              #(self.height, self.width) = im.shape[:2]
        #plt.show()

        if self.mask_borders!=None:
            #set side values to zero
            if self.mask_borders[0]>0:
                im[:,:self.mask_borders[0]]=0
#            else:
#                print('0 or negative value in the mask array, please check!')

            if self.mask_borders[1]>0:
                im[:,-self.mask_borders[1]:]=0
#            else:
#                print('0 or negative value in the mask array, please check!')

            #set top values to zero
            if self.mask_borders[2]>0:
                im[:self.mask_borders[2],:]=0
#            else:
#                print('0 or negative value in the mask array, please check!')

            if self.mask_borders[3]>0:
                im[-self.mask_borders[3]:,:]=0
#            else:
#                print('0 or negative value in the mask array, please check!')


        if self.numpy==False:
            for i in range(1):

                op=1#cambia per decidere quanto grossi sono gli hotspot da togliere, ma se lo aumenti troppo l'immagine si sfoca.
                selem=disk(op)
                im=opening(im, selem)


                #im=cv2.medianBlur(im,1)  #If the input type is not np.uint8, the only allowed ksize values for cv2.medianBlur are 3 and 5
                #print(type(im))
                #im=scipy.signal.medfilt2d(im,1) #allows for float64 with larger kernel sizes
                #im=cv2.GaussianBlur(im,(5,5),5,5) #Gaussian Kernel Size 5x5



        im=im*self.factor

        return im
        plt.show()
