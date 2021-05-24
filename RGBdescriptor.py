import numpy as np 
import cv2

# In this Histogram X-axis represents the bins and Y-axis is used for values


class HistogramBGR:
    def __init__(self,bins):
        # Histogram bins 
        self.bins = bins

    def featurize(self,image):
        # image,    channels, mask, histSize,   rangesOfEachChannel
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        # normalizing in terms of pixel count say 20% insted of 120 pixel
        hist = cv2.normalize(hist, hist)   

        #finally Flatten bin*bin*bin
        return hist.flatten()