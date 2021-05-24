from RGBdescriptor import HistogramBGR
from query_engine import SearchEngine
import numpy as np 
import cv2
import pickle
########################################
# Make Sure Dataset is indexed         #
# beforehand. For this run             #
# Index_my_dataset.py  file            #
########################################                  


queryImagePath = raw_input("Enter Path For Query Image : ").strip()
queryimage = cv2.imread(queryImagePath)
cv2.imshow('Query',queryimage)

# extracting features from qury image
desc = HistogramBGR([8, 8, 8])
queryFeatures = desc.featurize(queryimage)

#loading Indexed Pickle file  
pickle_in = open('histogram.pickle', 'rb')

index = pickle.load(pickle_in)
#print index
search = SearchEngine(index)

results = search.engine(queryFeatures)

# initialize the two montages to display our results --
# we have a total of 25 images in the index, but let's only
# display the top 10 results; 5 images per montage, with
# images that are 400x166 pixels
montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")

for i in xrange(0,10):
    (score, imageName) =  results[i]

    # dataset folder is taken as images as default
    path = './images/%s' % imageName
    print path
    result  = cv2.imread(path)
    print "\t%d. %s : %.3f" % (i + 1, imageName, score)
 
    # check to see if the first montage should be used
    if i < 5:
        montageA[i * 166:(i + 1) * 166, :] = result
 
    # otherwise, the second montage should be used
    else:
        montageB[(i - 5) * 166:((i - 5) + 1) * 166, :] = result

# show the results
cv2.imshow("Results 1-5", montageA)
cv2.imshow("Results 6-10", montageB)
cv2.waitKey(0)   