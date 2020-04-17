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
from mrcnn.datasets.cocolike import CocoLikeDataset
from assets.config import UConfig, InferenceConfig
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
import skimage

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cig_butts_0004.h5")
dataset_train = CocoLikeDataset()
dataset_train.load_data('datasets/kangaroo/kangaroo.json', 'datasets/kangaroo')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('datasets/kangaroo/kangaroo.json', 'datasets/kangaroo')
dataset_val.prepare()




inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)
model.load_weights('test.h5', by_name=True)
real_test_dir = 'datasets/kangaroo'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

# for image_path in image_paths:
img = skimage.io.imread("dog.jpg")
img_arr = np.array(img)
results = model.detect([img_arr], verbose=0)
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], figsize=(5,5))
