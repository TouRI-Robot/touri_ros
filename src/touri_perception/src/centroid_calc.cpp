#include <iostream>
#include <vector>
#include <cmath>
// #include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

// PCL specific includes
#include <ros/console.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include <pcl/point_types.h>
// #include <pcl/PCLPointCloud2.h>
// #include <pcl/conversions.h>
// #include <pcl/sample_consensus/ransac.h>
// #include <pcl/filters/crop_box.h>
// #include <pcl_ros/transforms.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
// CROPBOX
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>

// RANSAC
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>


#include "touri_perception/centroid_calc.h"
#include "touri_perception/transform_service.h"



// using namespace cv;
using namespace std;
// using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

ImageConstPtr depth_=nullptr;
ImageConstPtr image_=nullptr;
PointCloud2ConstPtr cloud_=nullptr;

double dot_product(const std::vector<std::vector<double>> &e, const std::vector<double> &p)
{
    double result = 0;
    for (auto& v : e) //range based loop
        result += std::inner_product(v.begin(), v.end(), p.begin(), 0.0);
    return result;
}

void callback(const ImageConstPtr& depth_sub, const ImageConstPtr& image_sub, const PointCloud2ConstPtr& cloud_sub)
{
    depth_ = depth_sub;
    image_ = image_sub;
    cloud_ = cloud_sub;
    // ROS_INFO("getting data");
}

bool service_callback(touri_perception::centroid_calc::Request  &req,
             touri_perception::centroid_calc::Response &res)
{
  cv_bridge::CvImagePtr cv_ptr;
  const sensor_msgs::ImageConstPtr& source = image_;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(source, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return  false;
  }
  
  // cv_ptr = cv_bridge::toCvCopy(const ImageConstPtr& source, const std::string& encoding = std::string());
  // cv_ptr = cv_bridge::toCvCopy(source, sensor_msgs::image_encodings::BGR8);
  // const std::string &encoding=std::string()
  cout << "starting 3d pipeline!" << endl;
  int start_point1_x = req.x_start;
  int start_point1_y = req.y_start;
  int end_point1_x = req.x_end;
  int end_point1_y = req.y_end;
  
//   // auto start_point1 = std::make_tuple(start_point1_x, start_point1s_y);
  cv::Point start_point1(start_point1_x, start_point1_y);
//   // auto end_point1 = std::make_tuple(end_point1_x, end_point1_y);
  cv::Point end_point1(end_point1_x, end_point1_y);

  double w = end_point1_x - start_point1_x;
  double h = end_point1_y - start_point1_y;

  double centroidx = (w/2) + start_point1_x;
  double centroidy = (h/2) + start_point1_y;

cout << "Scaling" << endl;

// // // scaling the bounding box

  double scaling_factor = 1.2;
  double scaled_w = scaling_factor * w;
  double scaled_h = scaling_factor * h;

  double start_point2_x = centroidx-scaled_w/2;
  double start_point2_y = centroidy-scaled_h/2;
  double end_point2_x = centroidx+scaled_w/2;
  double end_point2_y = centroidy+scaled_h/2;

  cv::Point start_point2(start_point2_x, start_point2_y);
  cv::Point end_point2(end_point2_x, end_point2_y);



//   auto start_point2 = std::make_tuple(start_point2_x, start_point2_y);
//   auto end_point2 = std::make_tuple(end_point2_x, end_point2_y);

//   tuple<int,int,int> color1 {255,0,0};
//   tuple<int,int,int> color2 {0,255,0};
  double thickness = 2;

  cout << "changing the image" << endl;
  cv::rectangle( cv_ptr->image, start_point1, end_point1, CV_RGB(255,0,0), thickness);
  cv::rectangle( cv_ptr->image, start_point2, end_point2, CV_RGB(0,255,0), thickness);
//   cv_ptr->image = cv::rectangle( cv_ptr->image, start_point2, end_point2, CV_RGB(0,255,0), thickness);


// // Mat image = cv2.rectangle(image, start_point1, end_point1, color1, thickness);
//  // intrinsic matrix
vector<vector<double>>camera_matrix{
                    {910.7079467773438, 0.0, 634.5316772460938},
                    {0.0, 910.6213989257812, 355.40097045898},
                    {0.0, 0.0, 1.0}
                    };

double f_x = camera_matrix[0][0];
double c_x = camera_matrix[0][2];
double f_y = camera_matrix[1][1];
double c_y = camera_matrix[1][2];

vector<vector<double>>camera_matrix_inverse{
                    {0.0010980468585331087935, 0.0, -0.69674551483981780681},
                    {0.0, 0.0010981512197930497631, -0.39028400922516253192},
                    {0.0, 0.0, 1.0}
                    };

// 3d point from 2d point

vector<double> start_homo = {{start_point2_x, start_point2_y, 1.0}};
// // Eigen::Map<Matrix<double,3,1,RowMajor> > mat(start_homo[0]);
// // mat = mat.inverse();

  double start_3d = dot_product(camera_matrix_inverse, start_homo);

  vector<double> end_homo = {{end_point2_x, end_point2_y, 1.0}};
  double end_3d = dot_product(camera_matrix_inverse, end_homo);

  double z = 0.75; 
  double start_x_3d = ((start_point2_x - c_x) / f_x) * z;
  double start_y_3d = ((start_point2_y - c_y) / f_y) * z;
  double end_x_3d = ((end_point2_x - c_x) / f_x) * z;
  double end_y_3d = ((end_point2_y - c_y) / f_y) * z;

  ROS_INFO("Converting ROS PointCloud2 ->  open3d format");

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  // This will convert the message into a pcl::PointCloud
  pcl::fromROSMsg(*cloud_, *pcl_cloud);
  // pcl::PCLPointCloud2* pclCloud = new pcl::PCLPointCloud2; 
  // pcl::PCLPointCloud2ConstPtr cloudPtr(pclCloud);
  // pcl_conversions::toPCL(*cloud_, *pclCloud);
  std::cout << "cloud.height : " << cloud_->height << "\n";
  std::cout << "cloud.width : " << cloud_->width   << "\n";
  size_t num_points = pcl_cloud->size(); 
  std::cout << "Number of points : " << num_points << "\n";

  pcl::CropBox<pcl::PointXYZRGB> boxFilter;
  
  std::cout << "start_x_3d : " << start_x_3d << "\n";
  std::cout << "start_y_3d : " << start_y_3d << "\n";
  std::cout << "end_x_3d : " << end_x_3d << "\n";
  std::cout << "end_y_3d : " << end_y_3d << "\n";

  boxFilter.setMin(Eigen::Vector4f(start_x_3d, start_y_3d, -1.0, 1.0));
  boxFilter.setMax(Eigen::Vector4f(end_x_3d, end_y_3d, 1000.0, 1.0));
  boxFilter.setInputCloud(pcl_cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  boxFilter.filter(*cropped_cloud);
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  num_points = cropped_cloud->size();
  std::cout << "Cropped points size : " << num_points << "\n";

  std::vector<std::vector<float>> plane_normals;
  std::vector<std::vector<float>> plane_centroids;
  
  int num_planes = 4;
  for(int i = 0; i < num_planes; i++)
  {
    std::cout << "iteration : "<< i << "\n";
    std::vector<int> inliers;
    std::vector<int> outliers;
    // RandomSampleConsensus object and compute the appropriated model
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
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cropped_cloud, *inlier_indices, centroid);
    std::vector<float> plane_centroid(&centroid[0], centroid.data()+centroid.cols()*centroid.rows());
    plane_centroids.push_back(plane_centroid);

    std::cout << "Centroid : " << centroid << "\n"; 

    // Extracting model
    Eigen::VectorXf model_coefficients;
    ransac.getModelCoefficients(model_coefficients);
    std::vector<float> plane_normal(&model_coefficients[0], model_coefficients.data()+model_coefficients.cols()*model_coefficients.rows());
    plane_normals.push_back(plane_normal);

    // Creating outliers
    std::cout << "Inliers size : " << inliers.size() << "\n"; 
    int inlier_iter = 0;
    for(int i=0;i < num_points;++i){
      if(inliers[inlier_iter] == i)
        inlier_iter++;
      else
        outliers.push_back(i);
    }
    std::cout << "outliers size : " << outliers.size() << "\n";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outlier_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud (*cropped_cloud, outliers, *outlier_cloud);
    cropped_cloud = outlier_cloud;
    
    std::cout << "Iteration " << i << " - Cropped points size : " << cropped_cloud->size() << "\n";
  }
 
  int a = 0;
  int b = 0;

  bool foundPlanes = false;
  for(int i=0;i < num_planes; ++i){
    for(int j=0;j < num_planes; ++j){
      if(i!=j){
        double dot_product_value = 0;
        for (int k = 0; k < plane_normals[i].size(); ++k)
          dot_product_value += (plane_normals[i][k] * plane_normals[j][k]);
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

  std::cout << "a : " << a << "\n";
  std::cout << "b : " << b << "\n";

  vector<float> centroid_1 = plane_centroids[a];
  vector<float> centroid_2 = plane_centroids[b];
  vector<float> centroid3d;
  for(int i=0;i<centroid_2.size();++i){
    centroid3d.push_back((centroid_1[i] + centroid_2[i])/2);
  }
  for(int i=0;i<centroid3d.size();++i){
    std::cout << "centroid3d : " << centroid3d[i] << " ";
  }std::cout << "\n";

  double x_3d = centroid3d[0];
  double y_3d = centroid3d[1];
  double z_3d = centroid3d[2];

  res.x_reply = x_3d;
  res.y_reply = y_3d;
  res.z_reply = z_3d;
  res.width = -1;
  



  // convert point to point in base
  // ros::NodeHandle n;
  // ros::ServiceClient transform_client = n.serviceClient<touri_perception::transform_service>("transform_to_base");
  // touri_perception::transform_service point_3d;
  // point_3d.request.x_request = x_3d;
  // point_3d.request.y_request = y_3d;
  // point_3d.request.z_request = z_3d;
  // // double x_base = point_base.x_reply;
  // // double x_base = point_base.x_reply;
  // if (transform_client.call(point_3d))
  // {
  //   double x_base = point_3d.response.x_reply;
  //   double y_base = point_3d.response.y_reply;
  //   double z_base = point_3d.response.z_reply;
  //   cout << "x_base: " << x_base << endl;
  //   cout << "y_base: " << y_base << endl;
  //   cout << "z_base: " << z_base << endl;

  

  

  return true;
}

// bool add(beginner_tutorials::AddTwoInts::Request  &req,
//          beginner_tutorials::AddTwoInts::Response &res)

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_node");
  ROS_INFO("Intialized node");  
  ros::NodeHandle nh;
  ros::ServiceServer service = nh.advertiseService("centroid_calc", service_callback);


//   ros::NodeHandle nh;
  message_filters::Subscriber<Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1000);
  message_filters::Subscriber<Image> image_sub(nh, "/camera/color/image_raw", 1000);

  message_filters::Subscriber<PointCloud2> cloud_sub(nh, "/camera/depth/color/points", 1000);
//   message_filters::Subscriber<CameraInfo> info_sub(nh, "camera_info", 1);
  TimeSynchronizer<Image, Image, PointCloud2> sync(depth_sub, image_sub, cloud_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::spin();

  return 0;
}


