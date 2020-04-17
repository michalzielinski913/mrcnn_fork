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
from assets.config import UConfig
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


config=UConfig()

ROOT_DIR=""
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
dataset_train = CocoLikeDataset()
dataset_train.load_data('datasets/kangaroo/train/kangaroo.json', 'datasets/train/kangaroo')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('datasets/kangaroo/val/kangaroo.json', 'datasets/val/kangaroo')
dataset_val.prepare()

# dataset = dataset_train
# image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

start_train = time.time()
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=9,
            layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
