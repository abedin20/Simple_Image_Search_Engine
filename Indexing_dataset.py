from RGBdescriptor import HistogramBGR
import Pickle
import glob
import cv2

# initializing the index dictionary 

index = {}

desc = HistogramBGR([8, 8, 8])

path = raw_input("Enter Dataset Folder Path: ").strip()

if len(path) == 0:
    # folder for dataset is hardcoded to be 'images' if not provided
    path = './images'

for imgpath in glob.glob(path + "/*.png"): #keeping default format of images to be .png
    filename = imgpath[imgpath.rfind("/") + 1:]
    image = cv2.imread(imgpath)
    features = desc.featurize(image)
    index[filename] = features
    print features
    cv2.imshow(image)
    cv2.waitKey(1)

with open('histogram.pickle','wb') as f:
        pickle.dump(index,f)