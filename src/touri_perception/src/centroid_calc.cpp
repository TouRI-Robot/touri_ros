#include <iostream>
#include <vector>
#include <cmath>
// #include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cmath>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <ros/console.h>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;

namespace cv_bridge {

class CvImage
{
public:
  std_msgs::Header header;
  std::string encoding;
  cv::Mat image;
};

typedef boost::shared_ptr<CvImage> CvImagePtr;
typedef boost::shared_ptr<CvImage const> CvImageConstPtr;
}
ImageConstPtr depth_=nullptr;
ImageConstPtr image_=nullptr;
PointCloud2ConstPtr cloud_=nullptr;

void callback(const ImageConstPtr& depth_sub, const ImageConstPtr& image_sub, const PointCloud2ConstPtr& cloud_sub)
{
    depth_ = depth_sub;
    image_ = image_sub;
    cloud_ = cloud_sub;
    ROS_INFO("getting data");
}

void service_callback()
{
  cv_bridge::CvImagePtr toCvCopy(const ImageConstPtr& source, const std::string& encoding = std::string());

}

// bool add(beginner_tutorials::AddTwoInts::Request  &req,
//          beginner_tutorials::AddTwoInts::Response &res)

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vision_node");
  ROS_INFO("Intialized node");  
  ros::NodeHandle nh;

//   ros::NodeHandle nh;
  message_filters::Subscriber<Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1000);
  messagie_filters::Subscriber<Image> image_sub(nh, "/camera/color/image_raw", 1000);

  message_filters::Subscriber<PointCloud2> cloud_sub(nh, "/camera/depth/color/points", 1000);
//   message_filters::Subscriber<CameraInfo> info_sub(nh, "camera_info", 1);
  TimeSynchronizer<Image, Image, PointCloud2> sync(depth_sub, image_sub, cloud_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::spin();

  return 0;
}
// class CentroidCalc{
//     private:
//     public:
//         CentroidCalc(ros::NodeHandle& nh){
            
//             this->image_content = nullptr;
//             this->depth_content = nullptr;
//             this->pointcloud_content = nullptr;
//         }

//         void callback(const ImageConstPtr& depth, \
//                       const ImageConstPtr& image,
//                       const PointCloud2ConstPtr& cloud){
//             std::cout << "Callback called\n";
//         }
// };
// void callback(const ImageConstPtr& depth, \
//                       const ImageConstPtr& image,
//                       const PointCloud2ConstPtr& cloud){
//             std::cout << "Callback called\n";
//         }
// // void callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info)
// // {
// //   // Solve all of perception here...
// // }

// int main(int argc, char** argv)
// {
//   ros::init(argc, argv, "vision_node");

//   ros::NodeHandle nh;
//   message_filters::Subscriber<Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1000);
//   message_filters::Subscriber<Image> image_sub(nh, "/camera/color/image_raw", 1000);
//   message_filters::Subscriber<PointCloud2> cloud_sub(nh, "/camera/depth/color/points", 1000);
//   TimeSynchronizer<Image, Image, PointCloud2> ts(depth_sub, image_sub, cloud_sub, 10);
//   sync.registerCallback(boost::bind(&callback, _1, _2, _3));

//   ros::spin();

//   return 0;
// }

// double dot_product(const std::vector<std::vector<double>> &e, const std::vector<double> &p)
// {
//     double result = 0;
//     for (auto& v : e) //range based loop
//         result += std::inner_product(v.begin(), v.end(), p.begin(), 0.0);
//     return result;
// }

// int main(int argc, char **argv)
// {
//   ros::init(argc, argv, "centroid_client");
//   ros::NodeHandle n;
//   ros::ServiceClient client = n.serviceClient<beginner_tutorials::AddTwoInts>("add_two_ints");
//   beginner_tutorials::AddTwoInts srv;
//   srv.request.a = atoll(argv[1]);
//   srv.request.b = atoll(argv[2]);
//   if (client.call(srv))
//   {
//     ROS_INFO("Sum: %ld", (long int)srv.response.sum);
//   }
//   else
//   {
//     ROS_ERROR("Failed to call service add_two_ints");
//     return 1;
//   }

//   return 0;
// }

// void service_callback()
// {

// cout << "starting 3d pipeline!" << endl;
// int start_point1_x = x_start;
// int start_point1_y = y_start;
// int end_point1_x = x_end;
// int end_point1_y = y_end;

// int w = end_point1_x - start_point1_x;
// int h = end_point1_y - start_point1_y;

// int centroidx = (w/2) + start_point1_x;
// int centroidy = (h/2) + start_point1_y;

// cout << "Scaling" << endl;

// // scaling the bounding box

// double scaling_factor = 1.2;
// double scaled_w = scaling_factor * w;
// double scaled_h = scaling_factor * h;

// int start_point2_x = centroidx-scaled_w/2;
// int start_point2_y = centroidy-scaled_h/2;
// int end_point2_x = centroidx+scaled_w/2;
// int end_point2_y = centroidy+scaled_h/2;

// tuple<int,int,int> color1 {255,0,0};
// tuple<int,int,int> color2 {0,255,0};
// int thickness = 2;

// cout << "changing the image" << endl;

// Mat image = cv2.rectangle(image, start_point1, end_point1, color1, thickness);
//  // intrinsic matrix
// vector<vector<double>>camera_matrix{
//                     {910.7079467773438, 0.0, 634.5316772460938},
//                     {0.0, 910.6213989257812, 355.40097045898},
//                     {0.0, 0.0, 1.0}
//                     }

// double f_x = camera_matrix[0][0];
// double c_x = camera_matrix[0][2];
// double f_y = camera_matrix[1][1];
// double c_y = camera_matrix[1][2];

// vector<vector<double>>camera_matrix_inverse{
//                     {0.0010980468585331087935, 0.0, -0.69674551483981780681},
//                     {0.0, 0.0010981512197930497631, -0.39028400922516253192},
//                     {0.0, 0.0, 1.0}
//                     }

// // 3d point from 2d point

// vector<double> start_homo = {{start_point2_x, start_point2_y, 1.0}};
// // Eigen::Map<Matrix<double,3,1,RowMajor> > mat(start_homo[0]);
// // mat = mat.inverse();

// double start_3d = dot_product(camera_matrix_inverse, start_homo);

// vector<double> end_homo = {{end_point2_x, end_point2_y, 1.0}};
// double end_3d = dot_product(camera_matrix_inverse, end_homo);

// double z = 0.75;

// double start_x_3d = ((start_point2_x - c_x) / f_x) * z;
// double start_y_3d = ((start_point2_y - c_y) / f_y) * z;
// double end_x_3d = ((end_point2_x - c_x) / f_x) * z;
// double end_y_3d = ((end+point2_y - c_y) / f_y) * z;


// // visualizing 3D pcl
// cout << "-------Received ROS PointCloud2 message-------" << endl;

//  // get cloud data from point cloud 
 
// // convert open3d point cloud to cpp 
// // starts here 

// {
//   // Create a container for the data.
//   sensor_msgs::PointCloud2 output;

//   // Do data processing here...
//   output = *input;

//   // Publish the data.
//   pub.publish (output);



//   // Container for original & filtered data
//   pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
//   pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
//   pcl::PCLPointCloud2 cloud_filtered;

//   // Convert to PCL data type
//   pcl_conversions::toPCL(*cloud_msg, *cloud);

//   // Perform the actual filtering
//   pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
//   sor.setInputCloud (cloudPtr);
//   sor.setLeafSize (0.1, 0.1, 0.1);
//   sor.filter (cloud_filtered);

//   // Convert to ROS data type
//   sensor_msgs::PointCloud2 output;
//   pcl_conversions::fromPCL(cloud_filtered, output);

//   // Publish the data
//   pub.publish (output);
// }

// // ends here 












// }


