import os
import sys
import json
import numpy as np
import time
ROOT_DIR = ''
# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
from assets.config import UConfig
import random
from imutils import paths
from assets.mask import to_binary_mask
ROOT_DIR=""
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
import cv2
DATASET_PATH = os.path.abspath("isic/")
IMAGES_PATH = os.path.sep.join([DATASET_PATH,  "image"])
MASKS_PATH = os.path.sep.join([DATASET_PATH,  "mask"])

IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))


class isicConfig(Config):
    NAME="isic"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    VALIDATION_STEPS = 1
    STEPS_PER_EPOCH = 2000
    NUM_CLASSES = 2

class InterfaceISIC(isicConfig):
    DETECTION_MIN_CONFIDENCE = 0.9

config=isicConfig()

class isicDataset(utils.Dataset):

    def load_image(self, image_id):
        path=self.image_info[image_id]
        file=path["path"]
        img=cv2.imread(file)
        return img

    def load_data(self):
        self.add_class("lesion", 1, "lesion")
        onlyfiles = [f for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
        i=0;
        for paths in IMAGE_PATHS:
            file_id=onlyfiles[i]
            self.add_image("lesion",
                           path=paths,
                           image_id=file_id)
            i=i+1

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        filename = info["id"].split(".")[0]+"_segmentation.png"
        mask=to_binary_mask(cv2.imread(filename))
        return mask, 1

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

dataset_train=isicDataset()
dataset_train.load_data()
dataset_train.prepare()
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

start_train = time.time()

end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)