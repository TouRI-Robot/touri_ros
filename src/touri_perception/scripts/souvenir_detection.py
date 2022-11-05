#!/usr/bin/env python3
"""
TouRI Robot Base Code
"""
__author__    = "Shivani Sivakumar"
__mail__      = "ssivaku3@andrew.cmu.edu"
__copyright__ = "NONE"

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import torch, torchvision
import detectron2
import numpy as np
import cv2
import open3d as o3d
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
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32
from touri_perception.srv import transform_service,transform_serviceResponse
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from std_srvs.srv import Trigger, TriggerRequest
import sys
from std_msgs.msg import String, Bool
from touri_perception.srv import perception,perceptionResponse
from touri_perception.srv import picking_centroid_calc

# -----------------------------------------------------------------------------

bridge = CvBridge()
setup_logger()

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

#TODO: ADD OBJECTS DATASET
cfg.DATASETS.TRAIN = ("picking_train_dataset",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = [] #(1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.OUTPUT_DIR = "/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/output_picking_dataset"

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=False)

print("starting to train")
# trainer.train()

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

model_path = "/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/output_picking_dataset/model_final.pth"
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

# cfg.MODEL.WEIGHTS = "/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/picking_objects/picking_objects/output_picking_objects/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("picking_train_dataset")

from detectron2.utils.visualizer import ColorMode
import glob

print("==========================================================")

from pdb import set_trace as bp

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, \
    (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

class final:
    def __init__(self):
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        cloud_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)
        ts = message_filters.TimeSynchronizer([depth_sub, image_sub, cloud_sub], 10)

        ts.registerCallback(self.callback)

        self.image_content = None
        self.depth_content = None
        self.pointcloud_content = None
        
        self.image_server = rospy.Service('perception_service', perception, self.service_callback)
        rospy.wait_for_service('centroid_calc')
        self.centroid_client = rospy.ServiceProxy('centroid_calc', centroid_calc)

    def callback(self, depth_image, image, pointcloud):
        # print(" callabck")

        # self.image_content = image
        # self.depth_content = depth_image
        self.image_content = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.depth_content = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.height, depth_image.width, -1)
        self.pointcloud_content = pointcloud

    
    def service_callback(self, req):

        print("service callback")

        image = self.image_content
        depth_image = self.depth_content
        pointcloud = self.pointcloud_content
        print("Image : ",type(image))
        # Detectron2
        # image = bridge.imgmsg_to_cv2(image, "bgr8")
        # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
        # bp()
        # image = cv2.imread("/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/picking_objects/picking_objects/picking_objects/train/labelled_images/raw_image_0.jpg")
        outputs = predictor(image)
        v = Visualizer(image[:, :, ::-1],
                        metadata=test_metadata, 
                        scale=0.8
                        )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # bp()
        cv2.imwrite("image_written124.jpg",image)
        # cv2.waitKey(1)
        im = out.get_image()[:, :, ::-1]
        # if outputs['instances'].get_fields()['pred_boxes'][0]<1:
        #     detected = False

        try:
            # bp()
            x_start = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][0])
            y_start = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][1])
            x_end = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][2])
            y_end = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][3])
             
            resp_centroid = self.centroid_client(x_start, y_start,x_end,y_end)
            x_3d = resp_centroid.x_reply
            y_3d = resp_centroid.y_reply
            z_3d = resp_centroid.z_reply
            width = resp_centroid.width 
             # 3D
            
            print("starting 3D pipeline")
            # start_point1 = (619,186)
            # end_point1 = (960,630)
            start_point1 = (x_start,y_start)
            end_point1 = (x_end,y_end)

            w = end_point1[0] - start_point1 [0]
            h = end_point1[1] - start_point1 [1]
            centroidx = int(w/2 + start_point1[0])
            centroidy = int(h/2 + start_point1[1])

            print("scaling")

            # scale bb
            scaling_factor = 1.2
            scaled_w = scaling_factor * w
            scaled_h = scaling_factor * h
            start_point2 = (int(centroidx-scaled_w/2),int(centroidy-scaled_h/2))
            end_point2 = (int(centroidx+scaled_w/2),int(centroidy+scaled_h/2))
            color1 = (255, 0, 0)
            color2 = (0, 255, 0)
            thickness = 2
            print("changing image")
            image = cv2.rectangle(image, start_point1, end_point1, color1, thickness)
            image = cv2.rectangle(image, start_point2, end_point2, color2, thickness)
            cv2.imwrite("sectangpleimg.jpg", image)
            centroid_2d = (centroidx,centroidy)
            X = centroid_2d[0]
            Y = centroid_2d[1]
            default_z_3d = -1

            souvenir_position=[]

            # intrinsic matrix
            camera_matrix = np.array([[910.7079467773438, 0.0, 634.5316772460938], [0.0, 910.6213989257812, 355.40097045898], [0.0, 0.0, 1.0]])

            f_x = camera_matrix[0,0]
            c_x = camera_matrix[0,2]
            f_y = camera_matrix[1,1]
            c_y = camera_matrix[1,2]

            depth_data = self.depth_content
            # bp()
            Z = depth_data[Y-10:Y+10, X-10:X+10]
            z = np.mean(Z[np.nonzero(Z)])

            if z > 0:
                z_3d = z / 1000.0
                x_3d = ((X - c_x) / f_x) * z_3d
                y_3d = ((Y - c_y) / f_y) * z_3d

                print(x_3d, y_3d, z_3d)

                rospy.wait_for_service('transform_to_base')
                try:
                    transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
                except rospy.ServiceException as e:
                        print("Service call failed: %s"%e)
                        return
                resp1 = transform_to_base_client(x_3d, y_3d, z_3d)
                x_base = resp1.x_reply
                y_base = resp1.y_reply
                z_base = resp1.z_reply

                # self.image_pub.publish(self.br.cv2_to_imgmsg(image))
                # self.depth_pub.publish(self.br.cv2_to_imgmsg(depth_data))
                detected = True
                print(x_base, y_base, z_base)

                souvenir_position.append((x_base, y_base, z_base))
                return perceptionResponse(detected,x_base,y_base,z_base)

                
            else:
                z_3d = default_z_3d
        except IndexError:
            print("not detected")
            detected = False
            return perceptionResponse(detected,-1,-1,-1)
            
        

       
        # except Exception as e:
        #     print(e)

def main(args):
    rospy.init_node('souvenir_pose_estimator', anonymous=True)
    f = final()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
        

