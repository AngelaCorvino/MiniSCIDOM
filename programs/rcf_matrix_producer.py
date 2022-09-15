from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
from tomograpic_viewer import Tomograpic_viewer
import cv2
from scipy import signal as spsig

#Class for loading and manipulate camera pictures


class RCF_matrix_producer(object):

    #Initialization of class
    def __init__(self,path_array_images ):

        self.path_array_images= path_array_images



    def LoadScannerFile(self,filepath):
        return np.flip(cv2.imread(filepath,cv2.IMREAD_ANYDEPTH),axis=0)


    #Function load and manipulated image
    def produce_stack(self):



      for i in range(len(self.path_array_images)):

            img = self.LoadScannerFile(self.path_array_images[i])


            if i==0:
                  RCF_matrix = np.zeros((len(self.path_array_images),
                                                    np.shape(img)[0],
                                                    np.shape(img)[1] ))

            RCF_matrix[i,:,:]=img

      shape_RCF_matrix = np.shape(RCF_matrix)


      process_RCF_matrix = np.zeros((                           shape_RCF_matrix[0],
                                     int(shape_RCF_matrix[1]+shape_RCF_matrix[1]),
                                     int(shape_RCF_matrix[2]+shape_RCF_matrix[2]) ))
      #print(np.shape(process_RCF_matrix))

      for i in range(shape_RCF_matrix[0]):

         cross_corr = spsig.correlate(          RCF_matrix[0]/RCF_matrix[0].max(),
                                                RCF_matrix[i]/RCF_matrix[i].max() )


         indices = np.where(cross_corr==cross_corr.max())
         index_max_1 = indices[0][0]
         index_max_2 = indices[1][0]


         x_min = int(index_max_1+1-shape_RCF_matrix[1]/2)
         x_max = int(index_max_1+1+shape_RCF_matrix[1]/2)
         y_min = int(index_max_2+1-shape_RCF_matrix[2]/2)
         y_max = int(index_max_2+1+shape_RCF_matrix[1]/2)



         process_RCF_matrix[          i,
                            x_min:x_max,
                            y_min:y_max ] = RCF_matrix[i,:,:]


         aligned_rcf_matrix = process_RCF_matrix[                                  :shape_RCF_matrix[0],
                                                 int(shape_RCF_matrix[1]/2):-int(shape_RCF_matrix[1]/2),
                                                 int(shape_RCF_matrix[2]/2):-int(shape_RCF_matrix[2]/2) ]


      return aligned_rcf_matrix


if __name__ == "__main__":

    #Read_path
    directory='/home/corvin22/Desktop/miniscidom/pictures/2021-08-13/85P3/'
    filename_1='Doseof-01.tif'
    filename_2='Doseof-02.tif'
    filename_3='Doseof-03.tif'
    filename_4='Doseof-04.tif'
    filename_5='Doseof-05.tif'
    filename_6='Doseof-06.tif'
    filename_7='Doseof-07.tif'
    filename_8='Doseof-08.tif'
    filename_9='Doseof-09.tif'
    filename_10='Doseof-10.tif'
    filename_11='Doseof-11.tif'
    filename_12='Doseof-12.tif'
    filename_13='Doseof-13.tif'
    filename_14='Doseof-14.tif'
    filename_15='Doseof-15.tif'
    filename_16='Doseof-16.tif'
    filename_17='Doseof-17.tif'
    filename_18='Doseof-18.tif'
    filename_19='Doseof-19.tif'
    filename_20='Doseof-20.tif'
    filename_21='Doseof-21.tif'
    filename_22='Doseof-22.tif'
    filename_23='Doseof-23.tif'
    filename_24='Doseof-24.tif'
    filename_25='Doseof-25.tif'
    filename_26='Doseof-26.tif'
    filename_27='Doseof-27.tif'
    filename_28='Doseof-28.tif'
    filename_29='Doseof-29.tif'
    filename_30='Doseof-30.tif'



    file_path =[ directory+filename_1,
                 directory+filename_2,
                 directory+filename_3,
                 directory+filename_4,
                 directory+filename_5,
                 directory+filename_6,
                 directory+filename_7,
                 directory+filename_8,
                 directory+filename_9,
                directory+filename_10,
                directory+filename_11,
                directory+filename_12,
                directory+filename_13,
                directory+filename_14,
                directory+filename_15,
                directory+filename_16,
                directory+filename_17,
                directory+filename_18,
                directory+filename_19,
                directory+filename_20,
                directory+filename_21,
                directory+filename_22,
                directory+filename_23,
                directory+filename_24,
                directory+filename_25,
                directory+filename_26,
                directory+filename_27,
                directory+filename_28,
                directory+filename_29,
                directory+filename_30 ]

    rcf_matrix = RCF_matrix_producer(file_path).produce_stack()
    print(np.shape(rcf_matrix))


    Tomograpic_viewer(rcf_matrix,False,6)
