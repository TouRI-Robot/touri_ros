#!/usr/bin/env python3

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

cfg.MODEL.WEIGHTS = "/home/hello-robot/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/inference/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("dropbox_train_dataset")

from detectron2.utils.visualizer import ColorMode
import glob

# itr = 0

#         im = color_image
#         cv2.imshow('RealSense', image)
#         cv2.waitKey(1)

# finally:

#     # Stop streaming
#     pipeline.stop()

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
from pdb import set_trace as bp

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, \
    (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)


def callback(depth_image, image, pointcloud):
    print("Image : ",type(image))
    # Detectron2
    image = bridge.imgmsg_to_cv2(image, "bgr8")
    # bp()
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8
                    )
    print("Outputs : ",outputs)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im = out.get_image()[:, :, ::-1]
    # bp()
    # cv2.imshow('RealSense', im)
    # cv2.waitKey(1)

    # print("depth_image : ",type(depth_image))

    x_start = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][0])
    y_start = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][1])
    x_end = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][2])
    y_end = int(outputs['instances'].get_fields()['pred_boxes'].tensor[0][3])

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

    # intrinsic matrix
    camera_matrix = np.array([[910.7079467773438, 0.0, 634.5316772460938], [0.0, 910.6213989257812, 355.40097045898], [0.0, 0.0, 1.0]])

    f_x = camera_matrix[0,0]
    c_x = camera_matrix[0,2]
    f_y = camera_matrix[1,1]
    c_y = camera_matrix[1,2]

    # 3D point from 2D point
    start_homo = np.array([[start_point2[0]],[start_point2[1]],[1.0]])
    start_3d = np.linalg.inv(camera_matrix) @ start_homo
    end_homo = np.array([[end_point2[0]],[end_point2[1]],[1.0]])
    end_3d = np.linalg.inv(camera_matrix) @ end_homo

    z = 0.75
    start_x_3d = ((start_point2[0] - c_x) / f_x) * z
    start_y_3d = ((start_point2[1] - c_y) / f_y) * z
    end_x_3d = ((end_point2[0] - c_x) / f_x) * z
    end_y_3d = ((end_point2[1] - c_y) / f_y) * z

    # visualize 3D pcl

    rospy.loginfo("-------Received ROS PointCloud2 message-------")
    
    # Get cloud data from pointcloud
    field_names=[field.name for field in pointcloud.fields]
    cloud_data = list(pc2.read_points(pointcloud, skip_nans=True, \
                                      field_names = field_names))

    # Check if pointcloud is empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data)==0:
        rospy.loginfo("Converting an empty cloud")
        return None

    rospy.loginfo("Converting ROS PointCloud2 ->  open3d format")
    
    # Set open3d_cloud
    if "rgb" in field_names:
        rospy.loginfo("Found an rgb point cloud")
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb        
        # Get xyzopen3d_cloudopen3d_cloudopen3d_cloud
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)
        # Get rgb Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        # points = np.asarray(open3d_cloud.points)
        # colors = np.asarray(open3d_cloud.colors)
    

    # open3d_cloud = o3d.io.read_point_cloud("/home/shivani0812/Desktop/scripts/1664987632821457.pcd", format='pcd')
    # o3d.visualization.draw_geometries([open3d_cloud])

    # remove points from pointcloud
    points = np.asarray(open3d_cloud.points)
    colors = np.asarray(open3d_cloud.colors)
    open3d_cloud = open3d_cloud.select_by_index(np.where((start_x_3d<= points[:,0]) & (points[:,0] <= end_x_3d) & (start_y_3d <= points[:,1]) & (points[:,1] <= end_y_3d))[0])

    print("Poincloud  : ",type(open3d_cloud))
    # centroid = np.mean(points,axis=0).reshape(1,-1)
    # print(centroid)
    # plane_center = open3d_cloud.get_center()
    # plane_center = np.reshape(plane_center.T, (1,3))
    # points = np.append(points, plane_center, axis=0)
    # colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)
    # o3d.visualization.draw_geometries([open3d_cloud])

    # plane segmentation
    print("Downsampling point cloud")
    open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size = 0.003)
    new_cloud = open3d_cloud
    outlier_cloud = open3d_cloud
    centroids = []
    # normals = []

    # for i in range(5):
    #     plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
    #                                         ransac_n=3,
    #                                         num_iterations=500)
        
    #     [a, b, c, d] = plane_model
    #     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    #     print("Getting the inliers")

    #     inlier_cloud = outlier_cloud.select_by_index(inliers)
    #     outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    #     inlier_points = np.asarray(inlier_cloud.points)
    #     centroid = np.mean(inlier_points,axis=0).reshape(1,-1)
    #     centroids.append(centroid)
    #     print(f"Centroid: {centroid}")
    #     plane_center = inlier_cloud.get_center()    
    #     plane_center = np.reshape(plane_center.T,(1,3))
    #     print(f"Plane center: {plane_center}")

    #     points = np.append(points, plane_center, axis=0)
    #     colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

    #     pcl = o3d.geometry.PointCloud()
    #     pcl.points = o3d.utility.Vector3dVector(points)
    #     pcl.colors = o3d.utility.Vector3dVector(colors)
    #     # print(outlier_cloud.has_normals())
        
    #     # bb on segmented plane
    #     # aabb = plane_cloud.get_axis_aligned_bounding_box()
    #     # aabb.color = (1,0,0)
    #     obb = inlier_cloud.get_oriented_bounding_box()
    #     obb.color = (0,1,0)
    #     box_coords = np.asarray(obb.get_box_points())
    #     obb_center = obb.get_center()
    #     print(f"OBB center: {obb_center}")
    #     #find normal to plane
    #     # normal = np.array([a,b,c])
    #     # normal = normal / np.linalg.norm(normal)
    #     # print(f"Normal: {normal}")
    #     # normals.append(normal)
    #     # normal = np.reshape(normal.T,(1,3))
    #     # points = np.append(points, normal, axis=0)
    #     colors = np.append(colors, np.array([[0,1,0]]), axis=0)
        



    #     # print(box_coords)
    #     # print(obb.has_normals())

    #     x,y,z = centroid.flatten()
    #     for i in range(box_coords.shape[0]):
    #         x,y,z = box_coords[i].flatten()

    #     x,y,z = centroid.flatten()



    #     # o3d.visualization.draw_geometries([open3d_cloud, pcl, obb])
    #     # n = pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     # print("n", n)


    #     # o3d.geometry.estimate_normals(
    #     # inlier_cloud,
    #     # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
    #     #                                                   max_nn=30))
    #     # o3d.visualization.draw_geometries([inlier_cloud])
  
    #     o3d.visualization.draw_geometries([outlier_cloud, pcl, obb])
    normals = []
    for i in range(4):
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
                                            ransac_n=3,
                                            num_iterations=500)
        
        [a, b, c, d] = plane_model

        plane_cloud = outlier_cloud.select_by_index(inliers)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

        points = np.asarray(plane_cloud.points)
        colors = np.asarray(plane_cloud.colors)
        plane_center = plane_cloud.get_center()
        plane_center = np.reshape(plane_center.T,(1,3))
        centroids.append(plane_center)
        points = np.append(points, plane_center, axis=0)
        colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)

        #find normal to plane
        normal = np.array([a,b,c])
        normal = normal / np.linalg.norm(normal)
        print(f"Normal: {normal}")
        # normal = np.reshape(normal.T,(1,3))
        normals.append(normal)
        normals_arr = np.array(normals)
        # np.append(normals,normal, axis = 0)
        # print("normal:", normal)

        # bb on segmented plane
        # aabb = plane_cloud.get_axis_aligned_bounding_box()
        # aabb.color = (1,0,0)
        obb = plane_cloud.get_oriented_bounding_box()
        obb.color = (0,1,0)
        # o3d.visualization.draw_geometries([open3d_cloud, pcl, obb])
    
    # for n in normals:
    #     print("normals are:", n)
    print(normals_arr)
    a = 0
    b = 0
    print("calculating dot product")
    for i in range(normals_arr.shape[0]):
        print("i:", i)
        for j in range(normals_arr.shape[0]):
             print("j:", i)
             if(i!=j):
                dot_product = np.dot(normals_arr[i],normals_arr[j])
                if abs(dot_product)<=1 and abs(dot_product)>=0.96:
                    a = i
                    b = j
                    break
    print("a", a)
    print("b", b)

    # for c in centroids:
    #     print(c)
    print(centroids[a])
    print(centroids[b])

    diff = abs(centroids[a] - centroids[b])
    print("difference:", diff)
    max = 0
    index = 0
    # for i in range(diff.shape[1]):
    #     if diff[0][i] >= max:
    #         max = diff[0][i]
    #         index = i
    # centroid3d = diff[index] / 2

    # centroid3d = [0,0,0]
    # centroid3d[index] = diff[0][index] / 2
    cent1 = np.array(centroids[a])
    cent2 = np.array(centroids[b])
    print(cent1.shape)
    print(cent2.shape)
    # for i in range(3):
    #     if(i!=index):
    centroid3d= (cent1+ cent2)/2     
            
    print("final centroid:", centroid3d)
    final_centroid = np.array(centroid3d).reshape((1,3))
    print("reshaped final centroid")



    # points = np.asarray(new_cloud.points)
    # colors = np.asarray(new_cloud.colors)
    # points = np.append(points, np.array(centroid3d).reshape((1,3)), axis=0)
    # colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)
    # new_cloud.points = o3d.utility.Vector3dVector(points)
    # new_cloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([new_cloud])

    x_3d = final_centroid[0][0]
    y_3d = final_centroid[0][1]
    z_3d = final_centroid[0][2]

    print(x_3d, y_3d, z_3d)

    

    print("waiting for service")
    rospy.wait_for_service('transform_to_base')
    print("service found")


    try:
        transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        return
    resp1 = transform_to_base_client(x_3d, y_3d, z_3d)
    x_base = resp1.x_reply
    y_base = resp1.y_reply
    z_base = resp1.z_reply
    # detected = true
    print("base:")
    print(x_base, y_base, z_base)
    point  = PointStamped()
    point.header.stamp = rospy.Time.now()
    point.header.frame_id = '/map'
    point.point.x = resp1.x_reply
    point.point.y = resp1.y_reply
    point.point.z = resp1.z_reply
    place_pub.publish(point)

    # import os
    # time.sleep(1)
    # cmd = "rosservice call /grasp_object/trigger_grasp_object"
    # os.system("gnome-terminal -- {}".format(cmd))

    # print(detected,x_base,y_base,z_base)
    print("called the service")
    return resp1


    # return perceptionResponse(detected,x_base,y_base,z_base)



    
    







    

    
# print('callbak1')
if __name__ == '__main__':
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    cloud_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)
    place_pub  = rospy.Publisher("/goal_to_place",PointStamped, queue_size=1)
    rospy.init_node('Final')
    ts = message_filters.TimeSynchronizer([depth_sub, image_sub, cloud_sub], 10)
    ts.registerCallback(callback)
    print('spinning')
    rospy.spin()