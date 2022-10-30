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
# from detectron2.evaluation import COCOEvaluator
import torch
torch.cuda.empty_cache()
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from detectron2.utils.logger import setup_logger
setup_logger()

register_coco_instances("dropbox_train_dataset", {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images")
register_coco_instances("dropbox_val_dataset",   {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images")
# register_coco_instances("dropbox_test_dataset",  {}, "/home/mrsd-cmptr-4/Desktop/object_recognition/labels_coco.json", "/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images")

my_dataset_train_metadata = MetadataCatalog.get("dropbox_train_dataset")
dataset_dicts = DatasetCatalog.get("dropbox_train_dataset")

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
cfg.MODEL.DEVICE = 'cpu'

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
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

print("starting to train")
trainer.train()

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
print("Output directory",cfg.OUTPUT_DIR)

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("dropbox_train_dataset", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "dropbox_train_dataset")
# inference_on_dataset(trainer.model, val_loader, evaluator)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("dropbox_train_dataset", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("dropbox_train_dataset")

from detectron2.utils.visualizer import ColorMode
import glob

itr = 0
for imageName in glob.glob('/home/mrsd-cmptr-4/Desktop/object_recognition/final_dataset/train/original_images/*jpg'):
  im = cv2.imread(imageName)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8
                 )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  image = out.get_image()[:, :, ::-1]
  # image = cv2.rotate(src, cv2.cv2.ROTATE_90_CLOCKWISE)
  # cv2.imwrite(f"image_{itr}.jpg",image)
  itr += 1
  cv2.imshow("image",image)
  cv2.waitKey(1)


