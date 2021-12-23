import os
import numpy as np
from PIL import Image

class Image_process:

    #labels of the images
    labels = []

    #path of images
    path = []

    #path and label tuple list
    path_label = []

    #label and image matrix
    label_image = []

    def __init__(self):
        self.get_path_label()
    
    def get_path_label(self):
        dir = os.path.dirname(".")
        images = os.path.join(dir, "data") 

        #iterating through all the files in the data directory
        for root, dirs, files in os.walk(images):
            for file in files:

                #if file is an image then get the directory name as label
                #and the path of the image for further manipulation
                if file.endswith("png") or file.endswith("jpg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path)). replace(" ", "-").lower()

                    self.labels.append(label)
                    self.path.append(path)
                    self.path_label.append((path, label))
                    self.label_image.append((label,np.array(Image.open(path).convert("L"), "uint8")))
