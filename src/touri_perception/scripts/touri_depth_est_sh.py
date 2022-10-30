#!/usr/bin/env python3

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt

# def roi_cropping (x,y,w,h):

    # points to be converted to 3D
    
    # intrinsic matrix
    # projecting 2D points to the 3D points in camera frame
    # depth point cloud


    
if __name__ == '__main__':
    # drawing bb for the centroid and w,h
    image = cv2.imread("/home/shivani0812/Desktop/scripts/camera_image.jpeg")
    window_name = 'Image'
    start_point1 = (619,186)
    end_point1 = (960,630)
    w = end_point1[0] - start_point1 [0]
    h = end_point1[1] - start_point1 [1]
    centroidx = int(w/2 + start_point1[0])
    centroidy = int(h/2 + start_point1[1])

    # scale bb
    scaling_factor = 1.5
    scaled_w = scaling_factor * w
    scaled_h = scaling_factor * h
    start_point2 = (int(centroidx-scaled_w/2),int(centroidy-scaled_h/2))
    end_point2 = (int(centroidx+scaled_w/2),int(centroidy+scaled_h/2))
    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point1, end_point1, color1, thickness)
    image = cv2.rectangle(image, start_point2, end_point2, color2, thickness)

    # Displaying the image 
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

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
    open3d_cloud = o3d.io.read_point_cloud("/home/shivani0812/Desktop/scripts/1664987632821457.pcd", format='pcd')
    # o3d.visualization.draw_geometries([pcd])

    # remove points from pointcloud
    points = np.asarray(open3d_cloud.points)
    colors = np.asarray(open3d_cloud.colors)
    # print(points.shape)
    # mask = start_3d[0] <= points[:,0] <= end_3d[0] and start_3d[1] <= points[:,1] <= end_3d[1]
    # mask = start_3d[0] <= points[:,0]
    # pcd.points = o3d.utility.Vector3dVector(points[mask])
    # pcd = pcd.select_by_index(np.where(-0.1 <= points[:,0] <= end_3d[0] and start_3d[1] <= 0.2 <= end_3d[1])[0])
    open3d_cloud = open3d_cloud.select_by_index(np.where((start_x_3d<= points[:,0]) & (points[:,0] <= end_x_3d) & (start_y_3d <= points[:,1]) & (points[:,1] <= end_y_3d))[0])
    # & (start_3d[1] <= points[:,1]) & (points[:,1] <= end_3d[1])
    # (start_3d[0] <= points[:,0]) & (points[:,0] <= end_3d[0])

    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)

    # check axes
    # for i in range(50):
    #     # step = np.random.choice([-1, 0, 1], size=(1,2))
    #     # print(steps)
    #     points = np.append(points, np.array([[0,i*0.1,0.75]]), axis=0)
    #     colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

    
    # print(points.shape)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # visualize 3D pcl
    # o3d.visualization.draw_geometries([pcd])

    # plane segmentation
    print("Downsampling point cloud")
    open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size = 0.003)
    outlier_cloud = open3d_cloud
    centroids = []
    normals = []

    for i in range(6):
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
                                            ransac_n=3,
                                            num_iterations=500)
        
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        print("Getting the inliers")

        inlier_cloud = outlier_cloud.select_by_index(inliers)
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

        inlier_points = np.asarray(inlier_cloud.points)
        centroid = np.mean(inlier_points,axis=0).reshape(1,-1)
        centroids.append(centroid)
        print(f"Centroid: {centroid}")
        plane_center = inlier_cloud.get_center()    
        plane_center = np.reshape(plane_center.T,(1,3))
        print(f"Plane center: {plane_center}")

        points = np.append(points, plane_center, axis=0)
        colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        # print(outlier_cloud.has_normals())
        
        # bb on segmented plane
        # aabb = plane_cloud.get_axis_aligned_bounding_box()
        # aabb.color = (1,0,0)
        obb = inlier_cloud.get_oriented_bounding_box()
        obb.color = (0,1,0)
        box_coords = np.asarray(obb.get_box_points())
        obb_center = obb.get_center()
        print(f"OBB center: {obb_center}")
        #find normal to plane
        normal = np.array([a,b,c])
        normal = normal / np.linalg.norm(normal)
        print(f"Normal: {normal}")
        normals.append(normal)
        normal = np.reshape(normal.T,(1,3))
        points = np.append(points, normal, axis=0)
        colors = np.append(colors, np.array([[0,1,0]]), axis=0)
        



        # print(box_coords)
        # print(obb.has_normals())

        x,y,z = centroid.flatten()
        for i in range(box_coords.shape[0]):
            x,y,z = box_coords[i].flatten()

        x,y,z = centroid.flatten()



        # o3d.visualization.draw_geometries([open3d_cloud, pcl, obb])
        # n = pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # print("n", n)


        # o3d.geometry.estimate_normals(
        # inlier_cloud,
        # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
        #                                                   max_nn=30))
        # o3d.visualization.draw_geometries([inlier_cloud])
  
        o3d.visualization.draw_geometries([outlier_cloud, pcl, obb])
    
    
  
    





    