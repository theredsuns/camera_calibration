#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <memory>
#include <chrono>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <sl/Camera.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2/LinearMath/Matrix3x3.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono_literals;

const double TAG_SIZE = 0.06;
const int BASE_TAG_ID_0 = 0;
const int BASE_TAG_ID_1 = 1;
const int TARGET_TAG_ID = 2;
const int ROS_DOMAIN_ID = 36;
const string TOPIC_NAME = "Trace5_zed_relative";

const double ID0_TO_ID1_X = 0.1587;
const double ID0_TO_ID1_Y = 0.000;
const double ID0_TO_ID1_Z = 0.000;

// Global display strings (populated at startup)
string g_intrinsics_source = "???";
string g_intrinsics_values = "";
string g_distortion_info = "";

class AdvancedFilter {
private:
    deque<double> history;
    deque<double> history_x;
    deque<double> history_y;
    deque<double> history_z;
    deque<double> history_rx;
    deque<double> history_ry;
    deque<double> history_rz;
    int max_size;
    double alpha;
    double last_filtered;
    double last_x, last_y, last_z;
    double last_rx, last_ry, last_rz;
    bool initialized;
    mutex mtx;

public:
    AdvancedFilter(int size = 60, double smooth_factor = 0.1) 
        : max_size(size), alpha(smooth_factor), initialized(false) {
        last_filtered = last_x = last_y = last_z = 0;
        last_rx = last_ry = last_rz = 0;
    }

    void add(double x, double y, double z, double rx, double ry, double rz, double distance) {
        lock_guard<mutex> lock(mtx);

        if (!initialized) {
            last_filtered = distance;
            last_x = x; last_y = y; last_z = z;
            last_rx = rx; last_ry = ry; last_rz = rz;
            initialized = true;
            history.push_back(distance);
            history_x.push_back(x);
            history_y.push_back(y);
            history_z.push_back(z);
            history_rx.push_back(rx);
            history_ry.push_back(ry);
            history_rz.push_back(rz);
            return;
        }

        // Outlier rejection: skip if jump from last filtered is > 3 sigma
        double jump_dist = sqrt(pow(x - last_x, 2) + pow(y - last_y, 2) + pow(z - last_z, 2)) * 1000.0;
        double jump_rot = sqrt(pow(rx - last_rx, 2) + pow(ry - last_ry, 2) + pow(rz - last_rz, 2)) * 180.0 / M_PI;

        if (history_x.size() >= 5) {
            double std_dist = getStdDev(history) * 1000.0; // mm
            double std_rot = getStdDev(history_rx) * 180.0 / M_PI; // deg
            if (std_dist > 0.01 && jump_dist > std_dist * 6.0) return; // outlier, skip
            if (std_rot > 0.001 && jump_rot > std_rot * 6.0) return;
        }

        history.push_back(distance);
        history_x.push_back(x);
        history_y.push_back(y);
        history_z.push_back(z);
        history_rx.push_back(rx);
        history_ry.push_back(ry);
        history_rz.push_back(rz);

        if ((int)history.size() > max_size) {
            history.pop_front();
            history_x.pop_front();
            history_y.pop_front();
            history_z.pop_front();
            history_rx.pop_front();
            history_ry.pop_front();
            history_rz.pop_front();
        }
    }

    double getStdDev(const deque<double>& data) {
        if (data.size() < 2) return 0.0;
        double mean = 0.0;
        for (double v : data) mean += v;
        mean /= data.size();
        double var = 0.0;
        for (double v : data) var += (v - mean) * (v - mean);
        return sqrt(var / (data.size() - 1));
    }

    double getTrimmedMean(const deque<double>& data, double trim_percent = 0.25) {
        if (data.empty()) return 0;
        deque<double> sorted = data;
        sort(sorted.begin(), sorted.end());
        
        int trim_count = (int)(sorted.size() * trim_percent / 2);
        double sum = 0;
        int count = 0;
        
        for (int i = trim_count; i < (int)sorted.size() - trim_count; i++) {
            sum += sorted[i];
            count++;
        }
        
        if (count > 0) return sum / count;
        return sorted[sorted.size() / 2];
    }

    double lowPass(double new_val, double last_val, double a) {
        return a * new_val + (1 - a) * last_val;
    }

    void getSmoothed(double& x, double& y, double& z, 
                    double& rx, double& ry, double& rz, double& distance) {
        lock_guard<mutex> lock(mtx);
        
        if (history.empty()) {
            x = y = z = rx = ry = rz = distance = 0;
            return;
        }

        distance = lowPass(getTrimmedMean(history, 0.25), last_filtered, alpha);
        last_filtered = distance;

        x = lowPass(getTrimmedMean(history_x, 0.25), last_x, alpha);
        last_x = x;
        y = lowPass(getTrimmedMean(history_y, 0.25), last_y, alpha);
        last_y = y;
        z = lowPass(getTrimmedMean(history_z, 0.25), last_z, alpha);
        last_z = z;

        rx = lowPass(getTrimmedMean(history_rx, 0.25), last_rx, alpha);
        last_rx = rx;
        ry = lowPass(getTrimmedMean(history_ry, 0.25), last_ry, alpha);
        last_ry = ry;
        rz = lowPass(getTrimmedMean(history_rz, 0.25), last_rz, alpha);
        last_rz = rz;
    }

    void clear() {
        lock_guard<mutex> lock(mtx);
        history.clear();
        history_x.clear();
        history_y.clear();
        history_z.clear();
        history_rx.clear();
        history_ry.clear();
        history_rz.clear();
        initialized = false;
    }

    bool isStable() {
        lock_guard<mutex> lock(mtx);
        return history.size() >= max_size * 0.6;
    }
};

Mat rvecToMatrix(const Vec3d& rvec) {
    Mat R;
    Rodrigues(rvec, R);
    return R;
}

Vec3d matrixToRvec(const Mat& R) {
    Vec3d rvec;
    Rodrigues(R, rvec);
    return rvec;
}

Mat inverseTransform(const Mat& R, const Mat& t) {
    Mat T = Mat::eye(4, 4, CV_64F);
    R.copyTo(T(Rect(0, 0, 3, 3)));
    t.copyTo(T(Rect(3, 0, 1, 3)));
    return T.inv();
}

void transformPoint(const Mat& T, double x, double y, double z, 
                   double& ox, double& oy, double& oz) {
    Mat p = (Mat_<double>(4, 1) << x, y, z, 1.0);
    Mat tp = T * p;
    ox = tp.at<double>(0, 0);
    oy = tp.at<double>(1, 0);
    oz = tp.at<double>(2, 0);
}

Mat multiplyTransforms(const Mat& T1, const Mat& T2) {
    return T1 * T2;
}

void extractTransform(const Mat& T, Mat& R, Mat& t) {
    R = T(Rect(0, 0, 3, 3)).clone();
    t = T(Rect(3, 0, 1, 3)).clone();
}

class ThreeTagSystemNode : public rclcpp::Node {
public:
    ThreeTagSystemNode() : Node("three_tag_system") {
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(TOPIC_NAME, 10);
        RCLCPP_INFO(this->get_logger(), "三标签基准系统已启动");
        RCLCPP_INFO(this->get_logger(), "DOMAIN ID: %d", ROS_DOMAIN_ID);
        RCLCPP_INFO(this->get_logger(), "话题名: %s", TOPIC_NAME.c_str());
        RCLCPP_INFO(this->get_logger(), "基准标签: ID0 (主), ID1 (辅助)");
        RCLCPP_INFO(this->get_logger(), "目标标签: ID2");
    }

    void publishRelative(double x, double y, double z, 
                        double rx, double ry, double rz,
                        double distance, bool id0_found, bool id1_found, bool id2_found) {
        auto message = std_msgs::msg::Float64MultiArray();
        message.data = {
            x, y, z, distance,
            rx, ry, rz,
            id0_found ? 1.0 : 0.0,
            id1_found ? 1.0 : 0.0,
            id2_found ? 1.0 : 0.0
        };
        publisher_->publish(message);
    }

private:
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
};

void draw3DAxis(Mat& image, const Vec3d& rvec, const Vec3d& tvec, 
                const Mat& camera_matrix, const Mat& dist_coeffs, float length) {
    vector<Point3f> axis_points = {
        Point3f(0, 0, 0),
        Point3f(length, 0, 0),
        Point3f(0, length, 0),
        Point3f(0, 0, length)
    };

    vector<Point2f> image_points;
    projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);

    if (image_points.size() >= 4) {
        arrowedLine(image, image_points[0], image_points[1], Scalar(0, 0, 255), 2);
        arrowedLine(image, image_points[0], image_points[2], Scalar(0, 255, 0), 2);
        arrowedLine(image, image_points[0], image_points[3], Scalar(255, 0, 0), 2);
    }
}

void printSystemInfo(bool id0_found, bool id1_found, bool id2_found,
                    double id0_x, double id0_y, double id0_z,
                    double id1_x, double id1_y, double id1_z,
                    double rel_x, double rel_y, double rel_z,
                    double rel_rx, double rel_ry, double rel_rz,
                    double rel_distance, bool stable,
                    double rel1_x, double rel1_y, double rel1_z,
                    double rel1_rx, double rel1_ry, double rel1_rz,
                    double rel1_dist) {
    system("clear");
    
    cout << "==================================================" << endl;
    cout << "   三标签基准系统 (ID0+ID1 -> ID2)              " << endl;
    cout << "==================================================" << endl;
    cout << "  内参来源: " << g_intrinsics_source << endl;
    cout << "  内参值:   " << g_intrinsics_values << endl;
    cout << "  " << g_distortion_info << endl;
    cout << "  基准标签: ID0 (主) + ID1 (辅助稳定)" << endl;
    cout << "  目标标签: ID2" << endl;
    cout << "  ROS DOMAIN ID: " << ROS_DOMAIN_ID << endl;
    cout << "  话题名: " << TOPIC_NAME << endl;
    cout << "  稳定状态: " << (stable ? "✅ 已稳定" : "⏳ 收敛中...") << endl;
    cout << "==================================================" << endl;
    cout << fixed << setprecision(3);
    cout << endl;
    cout << "  标签检测状态:" << endl;
    cout << "    ID0: " << (id0_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << "    ID1: " << (id1_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << "    ID2: " << (id2_found ? "✅ 检测到" : "❌ 未检测") << endl;
    cout << endl;
    
    if (id0_found) {
        cout << "  ID0 相机坐标系 (mm):" << endl;
        cout << "    X: " << id0_x * 1000 << "  Y: " << id0_y * 1000 << "  Z: " << id0_z * 1000 << endl;
    }
    if (id1_found) {
        cout << "  ID1 相机坐标系 (mm):" << endl;
        cout << "    X: " << id1_x * 1000 << "  Y: " << id1_y * 1000 << "  Z: " << id1_z * 1000 << endl;
    }
    
    cout << endl;
    if (id0_found && id2_found) {
        cout << "  ID2 相对于 ID0 的位姿 (基准坐标系):" << endl;
        cout << "    位置 (mm):  X: " << rel_x * 1000 << endl;
        cout << "                Y: " << rel_y * 1000 << endl;
        cout << "                Z: " << rel_z * 1000 << endl;
        cout << "    距离:      " << rel_distance * 1000 << " mm" << endl;
        cout << "    姿态 (度): Rx: " << rel_rx * 180.0 / CV_PI << "°" << endl;
        cout << "                Ry: " << rel_ry * 180.0 / CV_PI << "°" << endl;
        cout << "                Rz: " << rel_rz * 180.0 / CV_PI << "°" << endl;
    } else {
        cout << "  请确保 ID0 和 ID2 标签在视野内..." << endl;
    }

    if (id0_found && id1_found) {
        cout << endl;
        cout << "  ID1 相对于 ID0 的位姿 (基准坐标系):" << endl;
        cout << "    位置 (mm):  X: " << rel1_x * 1000 << endl;
        cout << "                Y: " << rel1_y * 1000 << endl;
        cout << "                Z: " << rel1_z * 1000 << endl;
        cout << "    距离:      " << rel1_dist * 1000 << " mm" << endl;
        cout << "    姿态 (度): Rx: " << rel1_rx * 180.0 / CV_PI << "°" << endl;
        cout << "                Ry: " << rel1_ry * 180.0 / CV_PI << "°" << endl;
        cout << "                Rz: " << rel1_rz * 180.0 / CV_PI << "°" << endl;
    }
    cout << endl;
    cout << "  话题数据: [x, y, z, distance, rx, ry, rz, id0_ok, id1_ok, id2_ok]" << endl;
    cout << "==================================================" << endl;
    cout << "  按 ESC 键退出" << endl;
    cout << "==================================================" << endl;
}

int main(int argc, char** argv) {
    setenv("ROS_DOMAIN_ID", to_string(ROS_DOMAIN_ID).c_str(), 1);
    
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ThreeTagSystemNode>();
    
    cout << "正在初始化 ZED 相机..." << endl;
    
    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.depth_mode = sl::DEPTH_MODE::NONE;
    init_params.camera_fps = 30;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        cerr << "无法打开 ZED 相机: " << sl::toString(err) << endl;
        rclcpp::shutdown();
        return -1;
    }

    cout << "ZED 相机打开成功!" << endl;

    auto zed_params = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;
    cout << "ZED 出厂内参: fx=" << zed_params.fx << " fy=" << zed_params.fy
         << " cx=" << zed_params.cx << " cy=" << zed_params.cy << endl;

    // Try to load calibrated intrinsics
    string calib_file = "/home/nkk/coordate_change/ros2_ws/src/apriltag_zed_visp/config/zed_calibration.yaml";
    bool use_calib = false;
    double fx, fy, cx, cy;
    Mat dist_coeffs_full;

    ifstream cf(calib_file);
    if (cf.is_open()) {
        // Simple YAML parser (same as apriltag_detector)
        string line;
        vector<double> K(9), D;
        while (getline(cf, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.find("camera_matrix:") != string::npos) {
                getline(cf, line); getline(cf, line); getline(cf, line);
                size_t s = line.find('['), e = line.find(']');
                if (s != string::npos && e != string::npos) {
                    stringstream ss(line.substr(s+1, e-s-1));
                    string v; int i = 0;
                    while (getline(ss, v, ',') && i < 9) {
                        v.erase(0, v.find_first_not_of(" \t"));
                        K[i++] = stod(v);
                    }
                }
            }
            else if (line.find("distortion_coefficients:") != string::npos) {
                getline(cf, line); getline(cf, line); getline(cf, line);
                size_t s = line.find('['), e = line.find(']');
                if (s != string::npos && e != string::npos) {
                    stringstream ss(line.substr(s+1, e-s-1));
                    string v;
                    while (getline(ss, v, ',')) {
                        v.erase(0, v.find_first_not_of(" \t"));
                        if (!v.empty()) D.push_back(stod(v));
                    }
                }
            }
        }
        cf.close();
        if (K[0] > 0) {
            fx = K[0]; fy = K[4]; cx = K[2]; cy = K[5];
            use_calib = true;
            cout << "✅ 标定内参已加载: fx=" << fx << " fy=" << fy
                 << " cx=" << cx << " cy=" << cy << endl;
            g_intrinsics_source = "标定 (chessboard)";
            dist_coeffs_full = Mat(static_cast<int>(D.size()), 1, CV_64F);
            for (size_t i = 0; i < D.size(); ++i) dist_coeffs_full.at<double>(static_cast<int>(i)) = D[i];
        }
    }
    if (!use_calib) {
        fx = zed_params.fx; fy = zed_params.fy;
        cx = zed_params.cx; cy = zed_params.cy;
        cout << "⚠️ 未找到标定文件，使用 ZED 出厂内参" << endl;
        g_intrinsics_source = "ZED 出厂 (factory)";
    }

    {
        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy;
        g_intrinsics_values = ss.str();
    }
    g_distortion_info = "去畸变: ON (remap + 零畸变PnP)";

    Mat camera_matrix = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // Choose distortion for remap: if calibrated, use calibrated distortion
    Mat remap_dist;
    if (use_calib && !dist_coeffs_full.empty()) {
        remap_dist = dist_coeffs_full.clone();
        cout << "✅ 去畸变使用标定畸变系数 (" << dist_coeffs_full.total() << " coeffs)" << endl;
    } else {
        remap_dist = (Mat_<double>(5, 1) <<
            zed_params.disto[0], zed_params.disto[1],
            zed_params.disto[2], zed_params.disto[3],
            zed_params.disto[4]);
        cout << "⚠️ 去畸变使用 ZED 出厂畸变系数" << endl;
    }

    // Pre-compute undistortion map
    auto cam_info = zed.getCameraInformation().camera_configuration.resolution;
    int img_w = static_cast<int>(cam_info.width);
    int img_h = static_cast<int>(cam_info.height);
    Mat undist_map_x, undist_map_y;
    cv::initUndistortRectifyMap(camera_matrix, remap_dist, Mat(), camera_matrix,
                                Size(img_w, img_h),
                                CV_16SC2, undist_map_x, undist_map_y);

    // PnP uses ZERO distortion (image already remapped)
    Mat dist_coeffs = Mat::zeros(5, 1, CV_64F);
    cout << "✅ 去畸变映射已计算 → PnP 使用零畸变模型" << endl;
    cout << "✅ 使用内参: " << (use_calib ? "标定 (内参+畸变均来自标定)" : "ZED出厂") << endl;

    Ptr<aruco::Dictionary> dictionary = 
        aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_36h11);

    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    params->cornerRefinementMaxIterations = 100;
    params->cornerRefinementMinAccuracy = 0.001;

    AdvancedFilter relative_filter(20, 0.35);  // fast response for assembly tracking

    // Adaptive filtering: track raw relative pose velocity for assembly detection
    Vec3d prev_raw_rel_t(0, 0, 0);
    bool prev_raw_valid = false;

    int frame_count = 0;
    bool id0_found = false, id1_found = false, id2_found = false;
    Vec3d id0_tvec, id0_rvec;
    Vec3d id1_tvec, id1_rvec;
    Vec3d id2_tvec, id2_rvec;
    vector<Point2f> corners_id0, corners_id1;  // stored for combined PnP

    Mat frame;
    sl::Mat zed_img;
    
    namedWindow("三标签基准系统 (ID0+ID1 -> ID2)", WINDOW_NORMAL);
    resizeWindow("三标签基准系统 (ID0+ID1 -> ID2)", 1280, 720);

    while (rclcpp::ok()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            zed.retrieveImage(zed_img, sl::VIEW::LEFT);
            
            Mat cv_img(zed_img.getHeight(), zed_img.getWidth(), CV_8UC4, zed_img.getPtr<sl::uchar1>());
            cvtColor(cv_img, frame, COLOR_BGRA2BGR);

            // Undistort: remap to eliminate lens distortion (detection on rectified image)
            cv::remap(frame, frame, undist_map_x, undist_map_y, cv::INTER_LINEAR);

            vector<int> ids;
            vector<vector<Point2f>> corners;

            aruco::detectMarkers(frame, dictionary, corners, ids, params);

            id0_found = id1_found = id2_found = false;

            if (!ids.empty()) {
                aruco::drawDetectedMarkers(frame, corners, ids, Scalar(0, 255, 0));

                vector<Vec3d> rvecs, tvecs;
                aruco::estimatePoseSingleMarkers(
                    corners, TAG_SIZE, camera_matrix, dist_coeffs, rvecs, tvecs);

                for (size_t i = 0; i < ids.size(); i++) {
                    Scalar axis_color;
                    string tag_label;
                    
                    if (ids[i] == BASE_TAG_ID_0) {
                        id0_found = true;
                        id0_tvec = tvecs[i];
                        id0_rvec = rvecs[i];
                        corners_id0 = corners[i];
                        axis_color = Scalar(0, 255, 0);
                        tag_label = "ID0 (基准)";
                    } else if (ids[i] == BASE_TAG_ID_1) {
                        id1_found = true;
                        id1_tvec = tvecs[i];
                        id1_rvec = rvecs[i];
                        corners_id1 = corners[i];
                        axis_color = Scalar(255, 0, 255);
                        tag_label = "ID1 (辅助)";
                    } else if (ids[i] == TARGET_TAG_ID) {
                        id2_found = true;
                        id2_tvec = tvecs[i];
                        id2_rvec = rvecs[i];
                        axis_color = Scalar(0, 0, 255);
                        tag_label = "ID2 (目标)";
                    } else {
                        continue;
                    }

                    aruco::drawAxis(frame, camera_matrix, dist_coeffs, 
                                   rvecs[i], tvecs[i], TAG_SIZE * 0.5);
                    
                    putText(frame, tag_label, Point(corners[i][0].x, corners[i][0].y - 10),
                           FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 2);
                }

                if (id0_found && id2_found) {
                    frame_count++;

                    // Use improved ID0 pose when ID1 is visible
                    Vec3d id0_tvec_improved = id0_tvec;

                    // Scale correction using known ID0→ID1 distance as reference ruler
                    double scale_factor = 1.0;
                    if (id1_found) {
                        double measured_d01 = norm(id1_tvec - id0_tvec_improved);
                        double known_d01 = sqrt(ID0_TO_ID1_X*ID0_TO_ID1_X +
                                                ID0_TO_ID1_Y*ID0_TO_ID1_Y +
                                                ID0_TO_ID1_Z*ID0_TO_ID1_Z);
                        if (measured_d01 > 0.01 && known_d01 > 0) {
                            scale_factor = known_d01 / measured_d01;
                            // Clamp: reject obvious outliers (>5% deviation)
                            if (scale_factor < 0.95 || scale_factor > 1.05) scale_factor = 1.0;
                        }
                    }

                    Mat R_id0, R_id2;

                    // If ID1 is visible: use ID0+ID1 corners together for better ID0 pose
                    if (id1_found) {
                        // ArUco corner order: TL, TR, BR, BL
                        //   TL=(-h, +h)  TR=(+h, +h)  BR=(+h, -h)  BL=(-h, -h)
                        double h = TAG_SIZE / 2;
                        vector<Point3f> id0_obj = {
                            {-h,  h, 0}, { h,  h, 0}, { h, -h, 0}, {-h, -h, 0}
                        };
                        vector<Point3f> all_obj = id0_obj;
                        vector<Point2f> all_img(corners_id0.begin(), corners_id0.end());
                        // ID1 corners offset by [158.7, 0, 0] in ID0 frame
                        for (auto& p : id0_obj) {
                            all_obj.push_back(Point3f(p.x + ID0_TO_ID1_X, p.y, p.z));
                        }
                        all_img.insert(all_img.end(), corners_id1.begin(), corners_id1.end());
                        Vec3d rvec, tvec;
                        solvePnP(all_obj, all_img, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_IPPE);
                        solvePnPRefineLM(all_obj, all_img, camera_matrix, dist_coeffs, rvec, tvec);
                        R_id0 = rvecToMatrix(rvec);
                        id0_tvec_improved = tvec;
                    } else {
                        R_id0 = rvecToMatrix(id0_rvec);
                    }
                    R_id2 = rvecToMatrix(id2_rvec);

                    // Use combined-PnP ID0 rotation directly (8 corners = stable enough)
                    Mat R_rel = R_id0.t() * R_id2;
                    Mat t_rel_raw = R_id0.t() * (Mat(id2_tvec) - Mat(id0_tvec_improved));

                    // Apply scale correction
                    Mat t_rel = scale_factor * t_rel_raw;

                    Vec3d rel_rvec = matrixToRvec(R_rel);
                    double rel_distance = norm(t_rel);

                    // Adaptive: distinguish camera motion (both tags move together)
                    // from assembly motion (only ID2 moves relative to ID0)
                    // Strategy: always update filter, but use outlier rejection
                    // to skip frames corrupted by motion blur
                    relative_filter.add(
                        t_rel.at<double>(0, 0), t_rel.at<double>(1, 0), t_rel.at<double>(2, 0),
                        rel_rvec[0], rel_rvec[1], rel_rvec[2],
                        rel_distance
                    );

                    // Track assembly velocity from raw relative pose
                    Vec3d cur_raw_rel_t(t_rel.at<double>(0,0), t_rel.at<double>(1,0), t_rel.at<double>(2,0));
                    double assembly_speed = 0.0;
                    if (prev_raw_valid) {
                        assembly_speed = norm(cur_raw_rel_t - prev_raw_rel_t) * 1000.0; // mm/frame
                    }
                    prev_raw_rel_t = cur_raw_rel_t;
                    prev_raw_valid = true;
                    bool assembly_moving = (assembly_speed > 1.0); // >1mm/frame = ID2 moving

                    double smooth_x, smooth_y, smooth_z;
                    double smooth_rx, smooth_ry, smooth_rz, smooth_dist;
                    relative_filter.getSmoothed(smooth_x, smooth_y, smooth_z,
                                               smooth_rx, smooth_ry, smooth_rz, smooth_dist);

                    // Compute ID1 relative to ID0 (for display / verification)
                    double rel1_x = 0, rel1_y = 0, rel1_z = 0;
                    double rel1_rx = 0, rel1_ry = 0, rel1_rz = 0, rel1_dist = 0;
                    if (id1_found) {
                        Mat t_rel1 = R_id0.t() * (Mat(id1_tvec) - Mat(id0_tvec_improved));
                        rel1_x = t_rel1.at<double>(0,0);
                        rel1_y = t_rel1.at<double>(1,0);
                        rel1_z = t_rel1.at<double>(2,0);
                        rel1_dist = sqrt(rel1_x*rel1_x + rel1_y*rel1_y + rel1_z*rel1_z);
                        Mat R_id1 = rvecToMatrix(id1_rvec);
                        Mat R_rel1 = R_id0.t() * R_id1;
                        Vec3d r_rel1 = matrixToRvec(R_rel1);
                        rel1_rx = r_rel1[0]; rel1_ry = r_rel1[1]; rel1_rz = r_rel1[2];
                    }

                    if (frame_count % 3 == 0) {
                        printSystemInfo(id0_found, id1_found, id2_found,
                                       id0_tvec_improved[0], id0_tvec_improved[1], id0_tvec_improved[2],
                                       id1_found ? id1_tvec[0] : 0,
                                       id1_found ? id1_tvec[1] : 0,
                                       id1_found ? id1_tvec[2] : 0,
                                       smooth_x, smooth_y, smooth_z,
                                       smooth_rx, smooth_ry, smooth_rz,
                                       smooth_dist, relative_filter.isStable(),
                                       rel1_x, rel1_y, rel1_z,
                                       rel1_rx, rel1_ry, rel1_rz, rel1_dist);
                    }

                    node->publishRelative(
                        smooth_x * 1000, smooth_y * 1000, smooth_z * 1000,
                        smooth_rx, smooth_ry, smooth_rz,
                        smooth_dist * 1000,
                        id0_found, id1_found, id2_found
                    );

                    Vec3d smooth_rvec_vis(smooth_rx, smooth_ry, smooth_rz);
                    
                    // ID2 → ID0
                    stringstream ss;
                    ss << fixed << setprecision(1);
                    ss << "ID2->ID0: " << smooth_dist * 1000 << "mm"
                       << (assembly_moving ? " [MOVING]" : " [STILL]");
                    if (id1_found && fabs(scale_factor - 1.0) > 0.001)
                        ss << "  s=" << setprecision(4) << scale_factor;
                    ss << setprecision(1);
                    putText(frame, ss.str(), Point(20, 30),
                           FONT_HERSHEY_SIMPLEX, 0.7,
                           assembly_moving ? Scalar(0, 255, 0) : Scalar(0, 255, 255), 2);

                    stringstream ss2;
                    ss2 << "  X=" << smooth_x * 1000
                        << " Y=" << smooth_y * 1000
                        << " Z=" << smooth_z * 1000 << " mm";
                    putText(frame, ss2.str(), Point(20, 55),
                           FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 2);

                    // ID1 → ID0
                    if (id1_found) {
                        stringstream ss3;
                        ss3 << fixed << setprecision(1);
                        ss3 << "ID1->ID0: " << rel1_dist * 1000 << "mm"
                            << "  (ref ~158.7mm)";
                        putText(frame, ss3.str(), Point(20, 75),
                               FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200, 200, 200), 1);

                        stringstream ss4;
                        ss4 << "  X=" << rel1_x * 1000
                            << " Y=" << rel1_y * 1000
                            << " Z=" << rel1_z * 1000 << " mm";
                        putText(frame, ss4.str(), Point(20, 95),
                               FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1);
                    } else {
                        putText(frame, "ID1: MISSING (need ID1 in view)", Point(20, 75),
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    }

                }
            }

            if (!id0_found || !id2_found) {
                putText(frame, "请将 ID0(绿) 和 ID2(红) 标签放入视野", Point(20, frame.rows - 30), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }

            putText(frame, "ID0(绿)=基准 | ID1(紫)=辅助 | ID2(红)=目标", 
                   Point(20, frame.rows - 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

            imshow("三标签基准系统 (ID0+ID1 -> ID2)", frame);
        }

        rclcpp::spin_some(node);

        if (waitKey(10) == 27) {
            cout << "\n用户退出" << endl;
            break;
        }
    }

    zed.close();
    destroyAllWindows();
    rclcpp::shutdown();
    
    return 0;
}
