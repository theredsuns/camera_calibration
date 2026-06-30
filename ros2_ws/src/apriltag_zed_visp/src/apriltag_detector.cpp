#include "apriltag_zed_visp/apriltag_detector_node.h"
#include <iomanip>
#include <sstream>
#include <functional>
#include <thread>

using namespace std::placeholders;

namespace apriltag_zed_visp {

AprilTagDetectorNode::AprilTagDetectorNode()
    : Node("apriltag_detector")
    , camera_info_received_(false)
    , camera_params_set_(false)
    , running_(true)
    , use_direct_camera_(false)
    , camera_device_id_(0)
    , target_tag_id_(2)
    , tag_size_(0.06)
    , use_stereo_(false)
    , publish_tf_(true)
    , publish_marker_(true)
    , enable_compensation_(true)
    , print_precise_pose_(true)
{
    RCLCPP_INFO(this->get_logger(), "Initializing AprilTag detector node...");
    
    RCLCPP_INFO(this->get_logger(), "Declaring parameters...");
    declareParameters();
    RCLCPP_INFO(this->get_logger(), "Parameters declared");
    
    RCLCPP_INFO(this->get_logger(), "Loading basic parameters...");
    loadBasicParameters();
    RCLCPP_INFO(this->get_logger(), "Basic parameters loaded");
    
    try {
        detector_ = std::make_unique<AprilTag36h11Detector>(target_tag_id_, 1.0, 0.0, true);
        RCLCPP_INFO(this->get_logger(), "Detector created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create detector: %s", e.what());
    }
    
    try {
        pose_estimator_ = std::make_unique<PoseEstimator>(tag_size_);
        RCLCPP_INFO(this->get_logger(), "Pose estimator created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create pose estimator: %s", e.what());
    }
    
    try {
        error_compensation_ = std::make_unique<ErrorCompensation>();
        RCLCPP_INFO(this->get_logger(), "Error compensation created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create error compensation: %s", e.what());
    }
    
    RCLCPP_INFO(this->get_logger(), "Loading advanced parameters...");
    loadAdvancedParameters();
    RCLCPP_INFO(this->get_logger(), "Advanced parameters loaded");
    
    try {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        RCLCPP_INFO(this->get_logger(), "TF broadcaster created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create TF broadcaster: %s", e.what());
    }
    
    try {
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "tag_pose", 10);
        RCLCPP_INFO(this->get_logger(), "Pose publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create pose publisher: %s", e.what());
    }
    
    try {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "tag_marker", 10);
        RCLCPP_INFO(this->get_logger(), "Marker publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create marker publisher: %s", e.what());
    }
    
    try {
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "debug_image", 10);
        RCLCPP_INFO(this->get_logger(), "Debug image publisher created");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create debug image publisher: %s", e.what());
    }
    
    if (use_direct_camera_) {
        RCLCPP_INFO(this->get_logger(), "Using direct camera capture mode");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        camera_thread_ = std::thread(&AprilTagDetectorNode::cameraCaptureLoop, this);
    } else if (use_stereo_) {
        left_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "zed/left/image_rect_color", 10,
            std::bind(&AprilTagDetectorNode::leftImageCallback, this, _1));
        right_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "zed/right/image_rect_color", 10,
            std::bind(&AprilTagDetectorNode::rightImageCallback, this, _1));
        left_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "zed/left/camera_info", 10,
            std::bind(&AprilTagDetectorNode::cameraInfoCallback, this, _1));
    } else {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_rect", 10,
            std::bind(&AprilTagDetectorNode::imageCallback, this, _1));
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", 10,
            std::bind(&AprilTagDetectorNode::cameraInfoCallback, this, _1));
    }
    
    RCLCPP_INFO(this->get_logger(), "AprilTag ZED ViSP Detector initialized");
    RCLCPP_INFO(this->get_logger(), "Target Tag ID: %d", target_tag_id_);
    RCLCPP_INFO(this->get_logger(), "Tag Size: %.3f meters (6cm)", tag_size_);
    if (use_direct_camera_) {
        RCLCPP_INFO(this->get_logger(), "Camera Device ID: %d", camera_device_id_);
    }
}

AprilTagDetectorNode::~AprilTagDetectorNode() {
    running_ = false;
    if (camera_thread_.joinable()) {
        camera_thread_.join();
    }
}

void AprilTagDetectorNode::declareParameters() {
    this->declare_parameter("camera_frame", "camera_frame");
    this->declare_parameter("tag_frame", "tag_36h11_id2");
    this->declare_parameter("target_tag_id", 2);
    this->declare_parameter("tag_size", 0.06);
    this->declare_parameter("use_stereo", false);
    this->declare_parameter("use_direct_camera", true);
    this->declare_parameter("camera_device_id", 0);
    this->declare_parameter("camera_fx", 500.0);
    this->declare_parameter("camera_fy", 500.0);
    this->declare_parameter("camera_cx", 320.0);
    this->declare_parameter("camera_cy", 240.0);
    this->declare_parameter("image_width", 640);
    this->declare_parameter("image_height", 480);
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("publish_marker", true);
    this->declare_parameter("enable_compensation", true);
    this->declare_parameter("print_precise_pose", true);
}

void AprilTagDetectorNode::loadBasicParameters() {
    camera_frame_ = this->get_parameter("camera_frame").as_string();
    RCLCPP_INFO(this->get_logger(), "camera_frame: %s", camera_frame_.c_str());
    
    tag_frame_ = this->get_parameter("tag_frame").as_string();
    
    target_tag_id_ = this->get_parameter("target_tag_id").as_int();
    RCLCPP_INFO(this->get_logger(), "target_tag_id: %d", target_tag_id_);
    
    tag_size_ = this->get_parameter("tag_size").as_double();
    RCLCPP_INFO(this->get_logger(), "tag_size: %.3f", tag_size_);
    
    use_stereo_ = this->get_parameter("use_stereo").as_bool();
    RCLCPP_INFO(this->get_logger(), "use_stereo: %s", use_stereo_ ? "true" : "false");
    
    use_direct_camera_ = this->get_parameter("use_direct_camera").as_bool();
    RCLCPP_INFO(this->get_logger(), "use_direct_camera: %s", use_direct_camera_ ? "true" : "false");
    
    camera_device_id_ = this->get_parameter("camera_device_id").as_int();
    RCLCPP_INFO(this->get_logger(), "camera_device_id: %d", camera_device_id_);
}

void AprilTagDetectorNode::loadAdvancedParameters() {
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    publish_marker_ = this->get_parameter("publish_marker").as_bool();
    enable_compensation_ = this->get_parameter("enable_compensation").as_bool();
    print_precise_pose_ = this->get_parameter("print_precise_pose").as_bool();
    
    if (use_direct_camera_) {
        CameraParameters params;
        double fx = this->get_parameter("camera_fx").as_double();
        double fy = this->get_parameter("camera_fy").as_double();
        double cx = this->get_parameter("camera_cx").as_double();
        double cy = this->get_parameter("camera_cy").as_double();
        int width = this->get_parameter("image_width").as_int();
        int height = this->get_parameter("image_height").as_int();
        
        params.camera_matrix = cv::Matx33d(
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
        );
        params.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
        params.image_width = width;
        params.image_height = height;
        
        if (pose_estimator_) {
            pose_estimator_->setCameraParameters(params);
        }
        if (error_compensation_) {
            error_compensation_->setImageDimensions(width, height);
        }
        camera_params_set_ = true;
        
        RCLCPP_INFO(this->get_logger(), 
            "Direct camera params set: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, %dx%d",
            fx, fy, cx, cy, width, height);
    }
}

void AprilTagDetectorNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    
    if (!camera_params_set_) {
        CameraParameters params;
        params.camera_matrix = cv::Matx33d(
            msg->k[0], msg->k[1], msg->k[2],
            msg->k[3], msg->k[4], msg->k[5],
            msg->k[6], msg->k[7], msg->k[8]
        );
        
        params.dist_coeffs = cv::Mat(msg->d.size(), 1, CV_64F);
        for (size_t i = 0; i < msg->d.size(); ++i) {
            params.dist_coeffs.at<double>(i) = msg->d[i];
        }
        
        params.image_width = msg->width;
        params.image_height = msg->height;
        
        pose_estimator_->setCameraParameters(params);
        error_compensation_->setImageDimensions(msg->width, msg->height);
        
        camera_params_set_ = true;
        
        RCLCPP_INFO(this->get_logger(), 
            "Camera parameters set: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
            params.fx(), params.fy(), params.cx(), params.cy());
    }
    
    camera_frame_ = msg->header.frame_id;
    camera_info_received_ = true;
}

void AprilTagDetectorNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    
    if (!camera_params_set_) {
        RCLCPP_WARN_ONCE(this->get_logger(), 
            "Waiting for camera info...");
        return;
    }
    
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            msg, sensor_msgs::image_encodings::BGR8);
        
        cv::Mat image = cv_ptr->image;
        
        auto detections = detector_->detect(image);
        
        for (const auto& det : detections) {
            if (det.id != target_tag_id_) continue;
            
            TagPose pose = pose_estimator_->estimatePoseIterative(det.corners);
            
            if (!pose.valid) continue;
            
            if (enable_compensation_) {
                TagQualityMetrics metrics = error_compensation_->computeQualityMetrics(
                    det.corners, 
                    cv::Mat(pose_estimator_->getCameraParameters().camera_matrix));
                
                pose.translation = error_compensation_->compensateAll(
                    pose.translation, pose.rotation, det.corners,
                    cv::Mat(pose_estimator_->getCameraParameters().camera_matrix));
                
                if (print_precise_pose_) {
                    printPosePrecise(pose, metrics.confidence);
                }
            } else if (print_precise_pose_) {
                printPosePrecise(pose, 1.0);
            }
            
            publishDetection(det, pose, msg->header);
            
            if (publish_marker_) {
                publishMarker(pose, msg->header);
            }
            
            if (publish_tf_) {
                publishTransform(pose, msg->header);
            }
        }
        
        if (debug_image_pub_->get_subscription_count() > 0) {
            for (const auto& det : detections) {
                cv::polylines(image, std::vector<std::vector<cv::Point>>{
                    std::vector<cv::Point>(det.corners.begin(), det.corners.end())},
                    true, cv::Scalar(0, 255, 0), 2);
                
                for (const auto& corner : det.corners) {
                    cv::circle(image, corner, 3, cv::Scalar(0, 0, 255), -1);
                }
                
                std::stringstream ss;
                ss << "ID: " << det.id;
                cv::putText(image, ss.str(), det.center, 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
            
            sensor_msgs::msg::Image::SharedPtr debug_msg = 
                cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
            debug_image_pub_->publish(*debug_msg);
        }
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}

void AprilTagDetectorNode::leftImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    last_left_image_ = msg;
    processStereoImages();
}

void AprilTagDetectorNode::rightImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    last_right_image_ = msg;
}

void AprilTagDetectorNode::processStereoImages() {
    if (!last_left_image_) return;
    
    imageCallback(last_left_image_);
}

void AprilTagDetectorNode::printPosePrecise(const TagPose& pose, double confidence) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n========================================" << std::endl;
    ss << "AprilTag ID " << target_tag_id_ << " Detection Results:" << std::endl;
    ss << "========================================" << std::endl;
    ss << "Position (meters):" << std::endl;
    ss << "  X: " << pose.translation[0] * 1000.0 << " mm" << std::endl;
    ss << "  Y: " << pose.translation[1] * 1000.0 << " mm" << std::endl;
    ss << "  Z: " << pose.translation[2] * 1000.0 << " mm" << std::endl;
    ss << "Rotation (radians):" << std::endl;
    ss << "  Rx: " << pose.rotation[0] << std::endl;
    ss << "  Ry: " << pose.rotation[1] << std::endl;
    ss << "  Rz: " << pose.rotation[2] << std::endl;
    ss << "Reprojection Error: " << std::setprecision(6) 
       << pose.reprojection_error << " pixels" << std::endl;
    ss << "Confidence: " << std::setprecision(3) 
       << confidence * 100.0 << "%" << std::endl;
    ss << "========================================" << std::endl;
    
    RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
}

void AprilTagDetectorNode::publishDetection(
    const AprilTagDetection& detection,
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    auto msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
    msg->header = header;
    msg->header.frame_id = camera_frame_;
    
    msg->pose.position.x = pose.translation[0];
    msg->pose.position.y = pose.translation[1];
    msg->pose.position.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        msg->pose.orientation.x = q.x();
        msg->pose.orientation.y = q.y();
        msg->pose.orientation.z = q.z();
        msg->pose.orientation.w = q.w();
    } else {
        msg->pose.orientation.w = 1.0;
    }
    
    pose_pub_->publish(std::move(msg));
}

void AprilTagDetectorNode::publishMarker(
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    auto marker = std::make_unique<visualization_msgs::msg::Marker>();
    marker->header = header;
    marker->header.frame_id = camera_frame_;
    marker->ns = "apriltag";
    marker->id = target_tag_id_;
    marker->type = visualization_msgs::msg::Marker::CUBE;
    marker->action = visualization_msgs::msg::Marker::ADD;
    
    marker->pose.position.x = pose.translation[0];
    marker->pose.position.y = pose.translation[1];
    marker->pose.position.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        marker->pose.orientation.x = q.x();
        marker->pose.orientation.y = q.y();
        marker->pose.orientation.z = q.z();
        marker->pose.orientation.w = q.w();
    } else {
        marker->pose.orientation.w = 1.0;
    }
    
    marker->scale.x = tag_size_;
    marker->scale.y = tag_size_;
    marker->scale.z = 0.01;
    
    marker->color.r = 0.0;
    marker->color.g = 1.0;
    marker->color.b = 0.0;
    marker->color.a = 0.8;
    
    marker_pub_->publish(std::move(marker));
}

void AprilTagDetectorNode::publishTransform(
    const TagPose& pose,
    const std_msgs::msg::Header& header) {
    
    geometry_msgs::msg::TransformStamped transform;
    
    transform.header = header;
    transform.header.frame_id = camera_frame_;
    transform.child_frame_id = tag_frame_;
    
    transform.transform.translation.x = pose.translation[0];
    transform.transform.translation.y = pose.translation[1];
    transform.transform.translation.z = pose.translation[2];
    
    double angle = cv::norm(pose.rotation);
    if (angle > 1e-6) {
        cv::Vec3d axis = pose.rotation / angle;
        tf2::Quaternion q(
            axis[0] * sin(angle / 2),
            axis[1] * sin(angle / 2),
            axis[2] * sin(angle / 2),
            cos(angle / 2)
        );
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
    } else {
        transform.transform.rotation.w = 1.0;
    }
    
    tf_broadcaster_->sendTransform(transform);
}

void AprilTagDetectorNode::cameraCaptureLoop() {
    RCLCPP_INFO(this->get_logger(), "Attempting to open camera device %d...", camera_device_id_);
    
    cv::VideoCapture cap;
    
    try {
        if (!cap.open(camera_device_id_, cv::CAP_V4L2)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d with V4L2", camera_device_id_);
            if (!cap.open(camera_device_id_)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d", camera_device_id_);
                RCLCPP_ERROR(this->get_logger(), "Please check if the camera is connected");
                RCLCPP_ERROR(this->get_logger(), "Check permissions: ls -la /dev/video*");
                RCLCPP_ERROR(this->get_logger(), "Try: sudo chmod 666 /dev/video%d", camera_device_id_);
                return;
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception opening camera: %s", e.what());
        return;
    }
    
    RCLCPP_INFO(this->get_logger(), "Camera opened, setting parameters...");
    
    int width = this->get_parameter("image_width").as_int();
    int height = this->get_parameter("image_height").as_int();
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width));
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));
    cap.set(cv::CAP_PROP_FPS, 30.0);
    
    int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    
    RCLCPP_INFO(this->get_logger(), "Camera initialized: %dx%d @ %.1ffps (requested: %dx%d @ 30fps)", 
                actual_width, actual_height, actual_fps, width, height);
    
    cv::Mat frame;
    auto last_time = std::chrono::steady_clock::now();
    
    while (running_ && rclcpp::ok()) {
        cap >> frame;
        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame received");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_time).count();
        
        if (elapsed < 33) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_time = current_time;
        
        auto detections = detector_->detect(frame);
        
        for (const auto& det : detections) {
            if (det.id != target_tag_id_) continue;
            
            TagPose pose = pose_estimator_->estimatePoseIterative(det.corners);
            
            if (!pose.valid) continue;
            
            if (enable_compensation_) {
                TagQualityMetrics metrics = error_compensation_->computeQualityMetrics(
                    det.corners, 
                    cv::Mat(pose_estimator_->getCameraParameters().camera_matrix));
                
                pose.translation = error_compensation_->compensateAll(
                    pose.translation, pose.rotation, det.corners,
                    cv::Mat(pose_estimator_->getCameraParameters().camera_matrix));
                
                if (print_precise_pose_) {
                    printPosePrecise(pose, metrics.confidence);
                }
            } else if (print_precise_pose_) {
                printPosePrecise(pose, 1.0);
            }
            
            auto header = std_msgs::msg::Header();
            header.stamp = this->now();
            header.frame_id = camera_frame_;
            
            publishDetection(det, pose, header);
            
            if (publish_marker_) {
                publishMarker(pose, header);
            }
            
            if (publish_tf_) {
                publishTransform(pose, header);
            }
        }
        
        if (debug_image_pub_->get_subscription_count() > 0) {
            cv::Mat debug_frame = frame.clone();
            for (const auto& det : detections) {
                std::vector<cv::Point> corners_int;
                for (const auto& c : det.corners) {
                    corners_int.emplace_back(cv::Point(static_cast<int>(c.x), static_cast<int>(c.y)));
                }
                cv::polylines(debug_frame, std::vector<std::vector<cv::Point>>{corners_int},
                             true, cv::Scalar(0, 255, 0), 2);
                
                for (const auto& corner : det.corners) {
                    cv::circle(debug_frame, corner, 3, cv::Scalar(0, 0, 255), -1);
                }
                
                std::stringstream ss;
                ss << "ID: " << det.id;
                cv::putText(debug_frame, ss.str(), det.center, 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
            
            auto header = std_msgs::msg::Header();
            header.stamp = this->now();
            header.frame_id = camera_frame_;
            
            sensor_msgs::msg::Image::SharedPtr debug_msg = 
                cv_bridge::CvImage(header, "bgr8", debug_frame).toImageMsg();
            debug_image_pub_->publish(*debug_msg);
        }
    }
    
    cap.release();
    RCLCPP_INFO(this->get_logger(), "Camera capture loop stopped");
}

} 

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<apriltag_zed_visp::AprilTagDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
