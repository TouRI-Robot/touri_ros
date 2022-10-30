import open3d as o3d
import numpy as np


open3d_cloud = o3d.io.read_point_cloud("/home/shivani0812/Desktop/scripts/1664987632821457.pcd", format='pcd')
# o3d.visualization.draw_geometries([open3d_cloud])
print("Found an open3d point cloud")
# print(open3d_cloud)
# print(np.asarray(open3d_cloud.points))
# open3d_cloud.colors = o3d.utility.Vector3dVector(np.zeros((np.asarray(open3d_cloud.points).shape[0], 3)))
xyz = [(x,y,z) for x,y,z in zip(np.asarray(open3d_cloud.points)[:,0], np.asarray(open3d_cloud.points)[:,1], np.asarray(open3d_cloud.points)[:,2])]
# print(xyz)

print("Downsampling point cloud")
open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size = 0.003)
outlier_cloud = open3d_cloud
# o3d.visualization.draw_geometries([outlier_cloud])
displayed_pointcloud = False
centroids = []
for i in range(5):
    print("Finding a 2d plane")
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02, 
                                        ransac_n=3, 
                                        num_iterations=200)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    print("Getting the inliers")
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # o3d.visualization.draw_geometries([inlier_cloud])
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    inlier_points = np.asarray(inlier_cloud.points)
    centroid = np.mean(inlier_points,axis=0).reshape(1,-1)
    centroids.append(centroid)
    print("centroid", centroid)
    plane_center = inlier_cloud.get_center()
    plane_center = np.reshape(plane_center.T,(1,3))
    # o3d.visualization.draw_geometries([inlier_cloud, plane_center], point_show_normal=True)
    
    obb = inlier_cloud.get_oriented_bounding_box()
    obb.color = (1, 0, 0)
    box_coords = np.asarray(obb.get_box_points())
    # print("box_coords", box_coords)

    x,y,z = centroid.flatten()
    for i in range(box_coords.shape[0]):
        x,y,z = box_coords[i].flatten()

    x,y,z = centroid.flatten()

    if not displayed_pointcloud:
        o3d.visualization.draw_geometries([inlier_cloud, open3d_cloud, obb])
        displayed_pointcloud = True



