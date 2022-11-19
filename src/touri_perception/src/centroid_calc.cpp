// """
// TouRI Robot Base Code
// Centroid_calc.cpp
// This file implements the C++ pipeline for estimating the 3D centroid for shipping box.
// """
// __author__    = "Jigar Patel, Shivani Shivkumar"
// __mail__      = "jkpatel@andrew.cmu.edu"
// __copyright__ = "NONE"
// -----------------------------------------------------------------------------

// SYSTEM
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <ros/console.h>

// PCL specific includes
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <visualization_msgs/Marker.h>
#include <cmath>

// CROPBOX
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>

// RANSAC
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>

// TOURI SPECIFIC INCLUDES
#include "touri_perception/centroid_calc.h"
#include "touri_perception/transform_service.h"

// PCL Visualizer
#include <iostream>
#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

using namespace std::chrono_literals;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

// Publish centroids and normals
ros::Publisher marker_pub;

// Publish planes 
sensor_msgs::PointCloud2 plane_data;
ros::Publisher planes_pub;

// Markers to draw normals and centroids
visualization_msgs::Marker vis_centroids, vis_normals;

// image pointers - updated in callback
ImageConstPtr depth_=nullptr;
ImageConstPtr image_=nullptr;
PointCloud2ConstPtr cloud_=nullptr;

// Calculate dot product
double dot_product(const std::vector<std::vector<double>> &e, const std::vector<double> &p)
{
  double result = 0;
  for (auto& v : e) //range based loop
      result += std::inner_product(v.begin(), v.end(), p.begin(), 0.0);
  return result;
}

// Callback to update image pointers
void callback(const ImageConstPtr& depth_sub, const ImageConstPtr& image_sub, const PointCloud2ConstPtr& cloud_sub)
{
  depth_ = depth_sub;
  image_ = image_sub;
  cloud_ = cloud_sub;
  // Publish the data.
  // pub.publish(output);
  geometry_msgs::Point p;
  p.x = 5.0f;
  p.y = 3.0f;
  p.z = 2.0f;

  vis_centroids.points.push_back(p);
  // std::cout << "Publishing \n";
  marker_pub.publish(vis_centroids);
  // marker_pub.publish(vis_normals);
  // planes_pub.publish(plane_data)
  // ROS_INFO("Updating image pointers");
}

pcl::visualization::PCLVisualizer::Ptr rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  // viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();
  return (viewer);
}

void updateVis(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud){
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->removePointCloud();
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb);
  viewer->spinOnce();
}

bool service_callback(touri_perception::centroid_calc::Request  &req,
                        touri_perception::centroid_calc::Response &res)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "base_link";
  marker.header.stamp = ros::Time();
  marker.ns = "my_namespace";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::ARROW;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 1.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1.0;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  
  cv_bridge::CvImagePtr cv_ptr;
  const sensor_msgs::ImageConstPtr& source = image_;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(source, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Cv_bridge exception: %s", e.what());
    return  false;
  }
  // ======================== Extracting bounds ========================

  ROS_INFO("Starting 3d pipeline!");
  int start_point1_x = req.x_start;
  int start_point1_y = req.y_start;
  int end_point1_x = req.x_end;
  int end_point1_y = req.y_end;
  
  cv::Point start_point1(start_point1_x, start_point1_y);
  cv::Point end_point1(end_point1_x, end_point1_y);

  double w = end_point1_x - start_point1_x;
  double h = end_point1_y - start_point1_y;
  double centroidx = (w/2) + start_point1_x;
  double centroidy = (h/2) + start_point1_y;

  // ======================== scaling the bounding box ======================== 
  double scaling_factor = 1.2;
  double scaled_w = scaling_factor * w;
  double scaled_h = scaling_factor * h;

  double start_point2_x = centroidx-scaled_w/2;
  double start_point2_y = centroidy-scaled_h/2;
  double end_point2_x = centroidx+scaled_w/2;
  double end_point2_y = centroidy+scaled_h/2;

  cv::Point start_point2(start_point2_x, start_point2_y);
  cv::Point end_point2(end_point2_x, end_point2_y);
  double thickness = 2;

  ROS_INFO("changing the image node current");
  cv::rectangle( cv_ptr->image, start_point1, end_point1, CV_RGB(255,0,0), thickness);
  cv::rectangle( cv_ptr->image, start_point2, end_point2, CV_RGB(0,255,0), thickness);
  std::cout << "Savning oimg\n";
  cv::namedWindow("image_window", cv::WINDOW_AUTOSIZE);
  cv::imshow("image_window", cv_ptr->image);
  cv::waitKey(1);
  // cv::imwrite("/home/hello-robot/overwritten_image.jpg", cv_ptr->image);

  // ======================== Estimate centroid in 3D ========================       
  vector<vector<double>>camera_matrix{
                      {910.7079467773438, 0.0, 634.5316772460938},
                      {0.0, 910.6213989257812, 355.40097045898},
                      {0.0, 0.0, 1.0}
                      };

  double f_x = camera_matrix[0][0];
  double c_x = camera_matrix[0][2];
  double f_y = camera_matrix[1][1];
  double c_y = camera_matrix[1][2];

  vector<vector<double>> camera_matrix_inverse{
                      {0.0010980468585331087935, 0.0, -0.69674551483981780681},
                      {0.0, 0.0010981512197930497631, -0.39028400922516253192},
                      {0.0, 0.0, 1.0}
                      };

  // 3d point from 2d point
  vector<double> start_homo = {{start_point2_x, start_point2_y, 1.0}};
  double start_3d = dot_product(camera_matrix_inverse, start_homo);

  vector<double> end_homo = {{end_point2_x, end_point2_y, 1.0}};
  double end_3d = dot_product(camera_matrix_inverse, end_homo);

  ROS_INFO("Calulated start and end 3d");

  cv_bridge::CvImagePtr cv_ptr_depth;
  const sensor_msgs::ImageConstPtr& source_depth = depth_;
  try
  {
    cv_ptr_depth = cv_bridge::toCvCopy(source_depth, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return  false;
  }

  double z_3d1 = -1;
  cv::Mat depth_image = cv_ptr_depth->image;
  int window_size = 10;
  // std::cout << "centroid y : " << centroidy << "\n";
  // std::cout << "centroid x : " << centroidx << "\n";
  
  cv::Mat Z = depth_image(cv::Range((centroidy - window_size), (centroidy + window_size)), cv::Range((centroidx - window_size), (centroidx + window_size)));
  
  ROS_INFO("Calulated cv::Mat Z");
  
  int num_elements = 0;
  double sum = 0;
  for (int i = 0; i < Z.rows; i++)
  {
      for(int j = 0; j < Z.cols; j++)
      {
          if (Z.at<uint16_t>(j, i) > 0) {
            sum += Z.at<uint16_t>(j, i);
            num_elements++;  
          }
      }
  }
  
  z_3d1 = sum / (1000 * num_elements); // convert to meters
  
  double start_x_3d = ((start_point1_x - c_x) / f_x) * z_3d1;
  double start_y_3d = ((start_point1_y - c_y) / f_y) * z_3d1;
  double end_x_3d = ((end_point2_x - c_x) / f_x) * z_3d1;
  double end_y_3d = ((end_point2_y - c_y) / f_y) * z_3d1;

  // ======================== Crop the pointcloud ========================      
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  // This will convert the message into a pcl::PointCloud
  pcl::fromROSMsg(*cloud_, *pcl_cloud);
  size_t num_points = pcl_cloud->size(); 
  // std::cout << "cloud.height : " << cloud_->height << "cloud.width : " << \
        cloud_->width << "Number of points : " << num_points << "\n";

  pcl::CropBox<pcl::PointXYZRGB> boxFilter;
  boxFilter.setMin(Eigen::Vector4f(start_x_3d, start_y_3d, -1.0, 1.0));
  boxFilter.setMax(Eigen::Vector4f(end_x_3d, end_y_3d, 1000.0, 1.0));
  boxFilter.setInputCloud(pcl_cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  boxFilter.filter(*cropped_cloud);
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  // ======================== Find opposite plane centroids and normals ========================       
  std::vector<std::vector<float>> plane_normals;
  std::vector<std::vector<float>> plane_centroids;
  
  // auto viewer = rgbVis(cropped_cloud);
  ROS_INFO("Calculating planes");

  int num_planes = 4;
  for(int i = 0; i < num_planes; i++)
  {
    std::cout << "I : " << i << "\n";
    // std::cout << "iteration : "<< i << "\n";
    std::vector<int> inliers;
    std::vector<int> outliers;

    // RandomSampleConsensus object and compute the appropriated model
    std::cout << "Ransac\n";
    num_points = cropped_cloud->size();  
    pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cropped_cloud));
    pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);
    ransac.setDistanceThreshold (.02);
    ransac.computeModel();
    pcl::Indices model;
    ransac.getModel(model);
    ransac.getInliers(inliers);

    pcl::PointIndices::Ptr inlier_indices (new pcl::PointIndices);
    inlier_indices->indices = inliers;

    // Centroid computation
    std::cout << "Centroid computation\n";
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cropped_cloud, *inlier_indices, centroid);
    std::vector<float> plane_centroid(&centroid[0], centroid.data()+centroid.cols()*centroid.rows());
    plane_centroids.push_back(plane_centroid);

    // Extracting model
    std::cout << "Extracting model\n";
    Eigen::VectorXf model_coefficients;
    ransac.getModelCoefficients(model_coefficients);
    std::vector<float> plane_normal(&model_coefficients[0], model_coefficients.data()+model_coefficients.cols()*model_coefficients.rows());
    plane_normals.push_back(plane_normal);

    // Creating outliers
    std::cout << "Creating outliers\n";
    int inlier_iter = 0;
    for(int i=0;i < num_points;++i){
      if(inliers[inlier_iter] == i)
        inlier_iter++;
      else
        outliers.push_back(i);
    }
  
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outlier_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*cropped_cloud, outliers, *outlier_cloud);
    cropped_cloud = outlier_cloud;
    // updateVis(viewer, cropped_cloud); 
  }
 
  ROS_INFO("Calulated planes");
  
  int a = 0;
  int b = 0;

  bool foundPlanes = false;
  for(int i=0; i < num_planes; ++i){
    for(int j=0; j < num_planes; ++j){
      if(i!=j){
        double dot_product_value = 0;
        for (int k = 0; k < 3; ++k){
          std::cout << "i : "<< " k " << plane_normals[i][k] << "\n";
          std::cout << "j : "<< " k " << plane_normals[i][k] << "\n";
          
          dot_product_value += (plane_normals[i][k] * plane_normals[j][k]);
        }
          
        if(abs(dot_product_value)<=1 && abs(dot_product_value)>=0.96){
          a = i;
          b = j;
          foundPlanes = true;
          break;
        }
      }     
    }
    if(foundPlanes){
      break;
    }
  }

  // if(!foundPlanes){
  //   std::cout << "planes not found\n";
  //   return false;
  // }

  std::cout << "a : " << a << "\n";
  std::cout << "b : " << b << "\n";

  // ======================== Calculate the Box Centroid ========================

  vector<float> centroid_1 = plane_centroids[a];
  vector<float> centroid_2 = plane_centroids[b];
  vector<float> centroid3d;
  for(int i=0;i<centroid_2.size();++i){
    centroid3d.push_back((centroid_1[i] + centroid_2[i])/2);
  }

  // for(int i=0;i<centroid3d.size();++i){
  //   std::cout << "centroid3d : " << centroid3d[i] << " ";
  // }std::cout << "\n";

  double x_3d = centroid3d[0];
  double y_3d = centroid3d[1];
  double z_3d = centroid3d[2];

  geometry_msgs::Point p;
  p.x = x_3d;
  p.y = y_3d;
  p.z = z_3d;

  vis_centroids.points.push_back(p);
  // vis_normals.points.push_back(p);
  
  // std::cout << "Publishing \n";
  marker_pub.publish(vis_centroids);
  // marker_pub.publish(vis_normals);

  res.x_reply = x_3d;
  res.y_reply = y_3d;
  res.z_reply = z_3d;
  res.width = -1;
  std::cout << "Returning true from service\n";
  return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "shipping_vision_node");
  ROS_INFO("Intialized node");  
  ros::NodeHandle nh;
  ros::ServiceServer service = nh.advertiseService("centroid_calc", service_callback);

  // ========================== CENTROID AND NORMALS DEFINITION ======================
  // Common for points and normals
  vis_centroids.header.frame_id = vis_normals.header.frame_id = "camera_link";
  vis_centroids.header.stamp = vis_normals.header.stamp = ros::Time::now();
  vis_centroids.ns = vis_normals.ns = "points_and_lines";
  vis_centroids.action = vis_normals.action = visualization_msgs::Marker::ADD;
  vis_centroids.pose.orientation.w = vis_normals.pose.orientation.w = 1.0;

  vis_centroids.id = 0;
  vis_normals.id = 1;

  vis_centroids.type = visualization_msgs::Marker::SPHERE;
  // visualization_msgs::Marker::POINTS;
  vis_normals.type = visualization_msgs::Marker::ARROW;

  // POINTS markers use x and y scale for width/height respectively
  vis_centroids.scale.x = 0.2;
  vis_centroids.scale.y = 0.2;
  vis_centroids.scale.z = 0.2;

  // vis_normals markers use only the x component of scale, for the line width
  vis_normals.scale.x = 0.1;
  vis_normals.scale.x = 0.1;

  // Points are green
  vis_centroids.color.g = 1.0f;
  vis_centroids.color.a = 1.0;

  // Line strip is blue
  vis_normals.color.b = 1.0;
  vis_normals.color.a = 1.0;
  // ========================== CENTROID AND NORMALS DEFINITION ======================


  // Publish
  marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 10);
  planes_pub = nh.advertise<sensor_msgs::PointCloud2> ("planes_topic", 1);

  // ros::NodeHandle nh;
  message_filters::Subscriber<Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1000);
  message_filters::Subscriber<Image> image_sub(nh, "/camera/color/image_raw", 1000);
  message_filters::Subscriber<PointCloud2> cloud_sub(nh, "/camera/depth/color/points", 1000);
  
  TimeSynchronizer<Image, Image, PointCloud2> sync(depth_sub, image_sub, cloud_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::spin();

  return 0;
}