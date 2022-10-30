#!/usr/bin/env python3
import torch, torchvision
import detectron2
import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
import torch
torch.cuda.empty_cache()
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from detectron2.utils.logger import setup_logger
setup_logger()

# register_coco_instances("dropbox_train_dataset", {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/DATASET_SHIPPING_BOX/final_dataset/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/DATASET_SHIPPING_BOX/final_dataset/train/original_images")
# register_coco_instances("dropbox_val_dataset",   {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/DATASET_SHIPPING_BOX/final_dataset/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/DATASET_SHIPPING_BOX/final_dataset/train/original_images")
# register_coco_instances("dropbox_test_dataset",  {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images")

# my_dataset_train_metadata = MetadataCatalog.get("dropbox_train_dataset")
# dataset_dicts = DatasetCatalog.get("dropbox_train_dataset")

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("dropbox_train_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = [] #(1000, 1500)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=False)

print("starting to train")
# trainer.train()

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = "/home/hello-robot/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/inference/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85

cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)
# evaluator = COCOEvaluator("dropbox_train_dataset", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "dropbox_train_dataset")
# inference_on_dataset(trainer.model, val_loader, evaluator)

cfg.MODEL.WEIGHTS = "/home/hello-robot/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/inference/output/model_final.pth"
# cfg.DATASETS.TEST = ("dropbox_train_dataset", )
# cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("dropbox_train_dataset")

from detectron2.utils.visualizer import ColorMode
import glob

itr = 0
# output_video = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, (512, 384))

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    itr = 0
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        itr += 1

        im = color_image
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=0.8
                        )
        print("Outputs : ",outputs)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image = out.get_image()[:, :, ::-1]
        # image = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
        # cv2.imwrite(f"image_{itr}.jpg",image)
        # itr += 1
        # cv2.imshow("image",image)
        cv2.waitKey(1)
        # cv2.imwrite(f'image{itr}.jpg', images)
        cv2.imshow('RealSense', image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

# import time
# for imageName in sorted(glob.glob('/home/hello-robot/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/dataset/drop_box_dataset/train/original_images/*jpg')):
#   st = time.time()

#   im = cv2.imread(imageName)
#   outputs = predictor(im)
#   v = Visualizer(im[:, :, ::-1],
#                 metadata=test_metadata, 
#                 scale=0.8
#                  )
#   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#   image = out.get_image()[:, :, ::-1]
#   height, width, layers = image.shape
#   size = (width,height)
#   print("Size : ",size)
#   # image = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
#   # cv2.imwrite(f"image_{itr}.jpg",image)
#   itr += 1
#   print(time.time() - st)
#   # output_video.write(image)
#   cv2.imshow("image",image)
#   cv2.waitKey(1)

# output_video.release()
