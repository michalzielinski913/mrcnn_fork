import os
from os import listdir
from xml.etree import ElementTree
import sys
import json
import numpy as np
import time
from assets.mask import to_binary_mask
ROOT_DIR = ''
import cv2
# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.datasets.cocolike import CocoLikeDataset
from assets.config import UConfig, InferenceConfig
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DATASET_DIR="kangaroo-master"

config=UConfig()
COCO_MODEL_PATH="mask_rcnn_coco.h5"
class CangarooDataset(utils.Dataset):

    def load_data(self, dataset_dir, is_train=True):

        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "kangaroo")

        # define data locations for images and annotations
        images_dir = DATASET_DIR + '\\images\\'
        annotations_dir = DATASET_DIR + '\\annots\\'

        # Iterate through all files in the folder to
        # add class, images and annotaions
        for filename in os.listdir(images_dir):

            # extract image id
            image_id = filename[:-4]

            # skip bad images
            if image_id in ['00090']:
                continue
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:
                continue

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'

            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):

        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        i=0
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            i=i+1
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return boxes, width, height, i

    def chunks(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path=info["annotation"]
        boxes, w, h, i = self.extract_boxes(path)
        instance_masks = []
        class_ids = []
        mask = np.ones((w, h), dtype=np.uint8)
        x=0
        while x!=i:
            points = self.chunks(boxes[x], 2)
            x=x+1
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, np.array([points]), 255)
            instance_masks.append(to_binary_mask(mask))
            class_ids.append(1)
        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


data_train=CangarooDataset()
data_train.load_data(DATASET_DIR, is_train=True)
data_train.prepare()
data_val=CangarooDataset()
data_val.load_data(DATASET_DIR,is_train=False)
data_val.prepare()

# model = modellib.MaskRCNN(mode="training", config=config,
#                           model_dir=MODEL_DIR)
#
# model.load_weights(COCO_MODEL_PATH, by_name=True,
#                     exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                             "mrcnn_bbox", "mrcnn_mask"])
#
# start_train = time.time()
# model.train(data_train, data_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=9,
#             layers='heads')
# end_train = time.time()
# minutes = round((end_train - start_train) / 60, 2)


import skimage
inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
model.load_weights('kangaroo.h5', by_name=True)
real_test_dir = 'kangaroo-master/images'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

#for image_path in image_paths:
img = skimage.io.imread("boa-dusiciel.jpg")
img_arr = np.array(img)
results = model.detect([img_arr], verbose=0)
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            data_val.class_names, r['scores'], figsize=(5,5))



