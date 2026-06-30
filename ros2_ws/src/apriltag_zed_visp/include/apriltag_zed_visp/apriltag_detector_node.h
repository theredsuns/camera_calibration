#ifndef APRILTAG_DETECTOR_NODE_H
#define APRILTAG_DETECTOR_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.hpp>

#include "apriltag_zed_visp/apriltag_36h11_detector.h"
#include "apriltag_zed_visp/pose_estimator.h"
#include "apriltag_zed_visp/error_compensation.h"

#include <memory>
#include <string>
#include <mutex>

namespace apriltag_zed_visp {

class AprilTagDetectorNode : public rclcpp::Node {
public:
    AprilTagDetectorNode();
    ~AprilTagDetectorNode();

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void leftImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void rightImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    
    void processStereoImages();
    
    void publishDetection(const AprilTagDetection& detection,
                          const TagPose& pose,
                          const std_msgs::msg::Header& header);
    
    void publishMarker(const TagPose& pose,
                       const std_msgs::msg::Header& header);
    
    void publishTransform(const TagPose& pose,
                          const std_msgs::msg::Header& header);
    
    void printPosePrecise(const TagPose& pose, double confidence);
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    std::unique_ptr<AprilTag36h11Detector> detector_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    std::unique_ptr<ErrorCompensation> error_compensation_;
    
    sensor_msgs::msg::Image::ConstSharedPtr last_left_image_;
    sensor_msgs::msg::Image::ConstSharedPtr last_right_image_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr last_camera_info_;
    
    std::mutex image_mutex_;
    
    std::string camera_frame_;
    std::string tag_frame_;
    int target_tag_id_;
    double tag_size_;
    bool use_stereo_;
    bool publish_tf_;
    bool publish_marker_;
    bool enable_compensation_;
    bool print_precise_pose_;
    
    bool camera_info_received_;
    bool camera_params_set_;
    
    void declareParameters();
    void loadBasicParameters();
    void loadAdvancedParameters();
    
    void cameraCaptureLoop();
    
    bool use_direct_camera_;
    int camera_device_id_;
    std::thread camera_thread_;
    std::atomic<bool> running_;
};

} 

#endif 
