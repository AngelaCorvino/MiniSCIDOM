"""
Created on Sun Apr 26  2020
@author: hs5955
Indentation = 4 spaces

Class for RCF scanner image segmenation procedures

Overall key facts
* Output should be "stack" of images: List of all images of a specific stack, possibly with some meta-data
** This should be an object! ==> make as class
* It should be possible that several scan files are the source
** This list should be composed at the end
* Metadata could be obtained from Wiki
** with valid stack name, parameters can be obtained
** necessary here: configuration in general -> number of layers (!) & depth information if possible; Tile layout -> expected images size in pixel (with scan dpi), ROI of text; List of tiles at lasercutting
*** Todo at Lasercutter: upload TileLayout and List of LabelText (how to discriminate not full stacks?)


Status
* Open a single file (image file => np.array)
* Displays single file split into channels for visual inspection (np.array => visual output and decision to continue)
* Detect tiles in scan by edge detection (np.array => list of contours, use len(contours) for decision to continue)
* Simplify these edges to rectangles ()
* Split image into rectangles
* Read DPI from tiff file header (not clear for other file types)


TODO
* Rotation/Flip if scan was not correct
* Compose
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import pytesseract

# for reading the header
from PIL import Image
from PIL.TiffTags import TAGS

def getDPI(filepath):
    with Image.open(filepath) as img: # test what path or string is required
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
        DPI=meta_dict['XResolution'][0][0]
        return DPI
# maybe test ExifRead ??

def LoadScannerFile(filepath): # loads a single file
    # filepath is string, noting fancy like path
    # OpenCV uses wavelength-sorted color channels per default, hence Blue-Green-Red, what must be corrected here for matplotlib.pyplot
    return cv2.cvtColor(cv2.imread(filepath),cv2.COLOR_BGR2RGB)



def PreviewByChannels(img): # generates preview into color channels
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    black=np.zeros(img.shape[:-1]).astype('int8')
    plt.figure(figsize=(20,8))
    plt.subplot(2,4,1,title='RGB color')
    plt.imshow(img) # raw image as RGB, was 3x8 bit per pixel
    plt.subplot(2,4,2,title='Red channel')
    plt.imshow(np.stack([img[:,:,0],black,black],axis=2 ))
    plt.subplot(2,4,3,title='Green channel')
    plt.imshow(np.stack([black,img[:,:,1],black],axis=2 ))
    plt.subplot(2,4,4,title='Blue channel')
    plt.imshow(np.stack([black,black,img[:,:,2]],axis=2 ))
    plt.subplot(2,4,5,title='inverted grayscale in pseudocolor')
    plt.imshow(-gray,cmap='inferno') # raw image as RGB, was 3x8 bit per pixel
    plt.subplot(2,4,6,title='inverted red as pseudocolor')
    plt.imshow(-img[:,:,0],cmap='inferno')
    plt.subplot(2,4,7,title='inverted green as pseudocolor')
    plt.imshow(-img[:,:,1],cmap='inferno')
    plt.subplot(2,4,8,title='inverted blue pseudocolor')
    plt.imshow(-img[:,:,2],cmap='inferno')
    plt.show()
    return # nothing; can be improved to return graph object

def DetectTiles(img, blur=5, threshold=245): # detect tiles by edge recognition
    gray = cv2.medianBlur(\
                          cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)\
                          ,blur)
    _, thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV) # mask all white regions > thresh, transmit all regions less than 254 counts
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # returns only outer edges
    return contours
# future: number of tiles is known before and it can be checked

def Tiles2Rects(contours): # create pd.DataFrame of tiles as rectangles
    rectangle_columns=['PosX','PosY','W','H','angle'] # column names
    rectangles = pd.DataFrame(columns=rectangle_columns)
    for contour in contours:
        rect = cv2.minAreaRect(contour);
        # due to strange angle calculation of cv2.minAreaRect() and defining w and h before; actually not documented how minAreaRect works!
        if rect[2] < -45:
            angle = 90+rect[2]
            w=rect[1][1]
            h=rect[1][0]
        else:
            angle = rect[2]
            w=rect[1][0]
            h=rect[1][1]
        rectangles = rectangles.append(pd.DataFrame([[ rect[0][0], rect[0][1], w, h, angle ]], columns=rectangle_columns ),ignore_index=True)
    return rectangles

def SplitImage2Rects(image,rectangles,rectangle): # create list of sub-images as np.arrays
    TileList=[];
    for index, row in rectangles.iterrows():
        rotMatrix = cv2.getRotationMatrix2D((row['PosX'],row['PosY']),row['angle'],1)
        rotatedImage = cv2.warpAffine(image,rotMatrix,(image.shape[1],image.shape[0])) ## ATTENTION: Tuple order reversed!
        croppedImage = cv2.getRectSubPix(rotatedImage, rectangle, (row['PosX'],row['PosY']))
        TileList.append(croppedImage) # as list of np.arrays
    return TileList

def getRectangle(rectangles,mode,**kwargs):
    # if mode in {mean,min,max}: use rectangles.W/H.<mode>
    # if mode == manual: check kwargs for widht  and height and use them
    if mode in {'mean','max','min'}:
        return(\
              (int(eval('rectangles.W.'+mode+'()'))),int(eval('rectangles.H.'+mode+'()'))\
              )
    elif mode in {'manual'}:
        if 'width' in kwargs and 'height' in kwargs:
            return (int(kwargs['width']), int(kwargs['height']))
        else:
            raise Exception('Wrong keywords used. Use widht= and height=')
    else:
        raise Exception('Wrong mode chosen. Allowed are manual, mean, min, max.')


#def OCR2Label(image,config,ROI,channel):
#    label=pytesseract.image_to_string(image[ROI,channel], config=config)
#    return label

# even here, the ROI position should be read from the lasercutter data
# how to parameterize the first 2 and third dimension of slice independently?