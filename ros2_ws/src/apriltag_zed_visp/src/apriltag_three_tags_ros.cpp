#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <memory>
#include <chrono>
#include <mutex>
#include <cstdio>

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

const double TAG_SIZE_ID0 = 0.15;  // 15cm
const double TAG_SIZE_ID1 = 0.15;  // 15cm
const double TAG_SIZE_ID2 = 0.06;  // 6cm
const int BASE_TAG_ID_0 = 0;
const int BASE_TAG_ID_1 = 1;
const int TARGET_TAG_ID = 2;
const int ROS_DOMAIN_ID = 36;
const string TOPIC_NAME = "Trace5_zed_relative";

// ===== DEBUG: Z-axis jitter (debug-zed-z-jitter) =====
double g_dbg_pnpz0 = 0, g_dbg_zedz0 = -1, g_dbg_pnpz2 = 0, g_dbg_zedz2 = -1;
int g_dbg_frame = 0;
// ===== END DEBUG =====

const double ID0_TO_ID1_X = 0.1587;
const double ID0_TO_ID1_Y = 0.000;
const double ID0_TO_ID1_Z = 0.000;

// Global display strings (populated at startup)
string g_intrinsics_source = "???";
string g_intrinsics_values = "";
string g_distortion_info = "";

class KalmanFilter1D {
private:
    double x_est;
    double p_est;
    double q;
    double r;
    bool initialized;

public:
    KalmanFilter1D(double process_noise = 0.001, double measurement_noise = 0.01)
        : q(process_noise), r(measurement_noise), initialized(false) {}

    double filter(double measurement) {
        if (!initialized) {
            x_est = measurement;
            p_est = 1.0;
            initialized = true;
            return x_est;
        }

        double x_pred = x_est;
        double p_pred = p_est + q;

        double k = p_pred / (p_pred + r);
        x_est = x_pred + k * (measurement - x_pred);
        p_est = (1 - k) * p_pred;

        return x_est;
    }
};

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
    KalmanFilter1D kf_x;
    KalmanFilter1D kf_y;
    KalmanFilter1D kf_z;
    KalmanFilter1D kf_rx;
    KalmanFilter1D kf_ry;
    KalmanFilter1D kf_rz;
    KalmanFilter1D kf_dist;

public:
    AdvancedFilter(int size = 30, double smooth_factor = 0.3) 
        : max_size(size), alpha(smooth_factor), initialized(false),
          kf_x(0.001, 0.005), kf_y(0.001, 0.005), kf_z(0.001, 0.005),
          kf_rx(0.001, 0.01), kf_ry(0.001, 0.01), kf_rz(0.001, 0.01),
          kf_dist(0.001, 0.005) {
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
            kf_x.filter(x);
            kf_y.filter(y);
            kf_z.filter(z);
            kf_rx.filter(rx);
            kf_ry.filter(ry);
            kf_rz.filter(rz);
            kf_dist.filter(distance);
            return;
        }

        double jump_dist = sqrt(pow(x - last_x, 2) + pow(y - last_y, 2) + pow(z - last_z, 2)) * 1000.0;
        double jump_rot = sqrt(pow(rx - last_rx, 2) + pow(ry - last_ry, 2) + pow(rz - last_rz, 2)) * 180.0 / M_PI;
        if (jump_dist > 100.0 || jump_rot > 8.0) return;

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

    double getTrimmedMean(const deque<double>& data, double trim_percent = 0.3) {
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

    double getWeightedMean(const deque<double>& data) {
        if (data.empty()) return 0;
        double sum = 0;
        double weight_sum = 0;
        int n = data.size();
        for (int i = 0; i < n; i++) {
            double w = (i + 1) * (i + 1);
            sum += data[i] * w;
            weight_sum += w;
        }
        return sum / weight_sum;
    }

    void getSmoothed(double& x, double& y, double& z, 
                    double& rx, double& ry, double& rz, double& distance) {
        lock_guard<mutex> lock(mtx);
        
        if (history.empty()) {
            x = y = z = rx = ry = rz = distance = 0;
            return;
        }

        double raw_z = getTrimmedMean(history_z);
        double raw_x = getTrimmedMean(history_x);
        double raw_y = getTrimmedMean(history_y);
        double raw_rx = getTrimmedMean(history_rx);
        double raw_ry = getTrimmedMean(history_ry);
        double raw_rz = getTrimmedMean(history_rz);
        double raw_dist = getTrimmedMean(history);

        z = kf_z.filter(raw_z);
        x = kf_x.filter(raw_x);
        y = kf_y.filter(raw_y);
        rx = kf_rx.filter(raw_rx);
        ry = kf_ry.filter(raw_ry);
        rz = kf_rz.filter(raw_rz);
        distance = kf_dist.filter(raw_dist);

        z = lowPass(z, last_z, alpha);
        x = lowPass(x, last_x, alpha);
        y = lowPass(y, last_y, alpha);
        rx = lowPass(rx, last_rx, alpha);
        ry = lowPass(ry, last_ry, alpha);
        rz = lowPass(rz, last_rz, alpha);
        distance = lowPass(distance, last_filtered, alpha);

        last_x = x;
        last_y = y;
        last_z = z;
        last_rx = rx;
        last_ry = ry;
        last_rz = rz;
        last_filtered = distance;
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
                    double rel1_dist,
                    bool id0_r, bool id2_r,
                    double rrel_x, double rrel_y, double rrel_z) {
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
    if (id0_r && id2_r) {
        double rdist = sqrt(rrel_x*rrel_x + rrel_y*rrel_y + rrel_z*rrel_z);
        cout << endl;
        cout << "  [右眼] ID2 相对于 ID0 的位姿:" << endl;
        cout << "    位置 (mm):  X: " << rrel_x*1000 << endl;
        cout << "                Y: " << rrel_y*1000 << endl;
        cout << "                Z: " << rrel_z*1000 << endl;
        cout << "    距离:      " << rdist*1000 << " mm" << endl;
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
    init_params.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
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

    // Right camera: get intrinsics + compute undistortion map
    auto zed_params_r = zed.getCameraInformation().camera_configuration.calibration_parameters.right_cam;
    Mat camera_matrix_r = (Mat_<double>(3, 3) << zed_params_r.fx, 0, zed_params_r.cx,
                                                      0, zed_params_r.fy, zed_params_r.cy,
                                                      0, 0, 1);
    Mat dist_coeffs_r = (Mat_<double>(5, 1) << zed_params_r.disto[0], zed_params_r.disto[1],
                                                zed_params_r.disto[2], zed_params_r.disto[3],
                                                zed_params_r.disto[4]);
    Mat undist_map_rx, undist_map_ry;
    cv::initUndistortRectifyMap(camera_matrix_r, dist_coeffs_r, Mat(), camera_matrix_r,
                                Size(img_w, img_h), CV_16SC2, undist_map_rx, undist_map_ry);
    cout << "✅ 右目内参: fx=" << zed_params_r.fx << " fy=" << zed_params_r.fy << endl;

    cout << "✅ 使用内参: " << (use_calib ? "标定 (内参+畸变均来自标定)" : "ZED出厂") << endl;

    Ptr<aruco::Dictionary> dictionary = 
        aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_36h11);

    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    params->cornerRefinementMaxIterations = 100;
    params->cornerRefinementMinAccuracy = 0.001;

    AdvancedFilter relative_filter(30, 0.15);   // ID2→ID0: ultra-heavy smoothing
    AdvancedFilter id1_filter(30, 0.15);         // ID1→ID0

    // Adaptive filtering: track raw relative pose velocity for assembly detection
    Vec3d prev_raw_rel_t(0, 0, 0);
    bool prev_raw_valid = false;

    int frame_count = 0;
    bool id0_found = false, id1_found = false, id2_found = false;
    Vec3d id0_tvec, id0_rvec;
    Vec3d id1_tvec, id1_rvec;
    Vec3d id2_tvec, id2_rvec;

    Mat frame;
    sl::Mat zed_img, zed_img_r;
    // Right camera tag poses
    bool id0_r=false, id1_r=false, id2_r=false;
    Vec3d id0_tvec_r, id0_rvec_r, id1_tvec_r, id1_rvec_r, id2_tvec_r, id2_rvec_r;

    namedWindow("三标签基准系统 (ID0+ID1 -> ID2)", WINDOW_NORMAL);
    resizeWindow("三标签基准系统 (ID0+ID1 -> ID2)", 1280, 720);

    while (rclcpp::ok()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            zed.retrieveImage(zed_img, sl::VIEW::LEFT);
            sl::Mat depth_map;
            zed.retrieveMeasure(depth_map, sl::MEASURE::DEPTH);
            Mat depth_raw(depth_map.getHeight(), depth_map.getWidth(), CV_32FC1, depth_map.getPtr<float>());
            Mat depth_undist;
            cv::remap(depth_raw, depth_undist, undist_map_x, undist_map_y, cv::INTER_LINEAR);

            Mat cv_img(zed_img.getHeight(), zed_img.getWidth(), CV_8UC4, zed_img.getPtr<sl::uchar1>());
            cvtColor(cv_img, frame, COLOR_BGRA2BGR);

            // Undistort: remap to eliminate lens distortion
            cv::remap(frame, frame, undist_map_x, undist_map_y, cv::INTER_LINEAR);
                Mat frame_left = frame.clone();  // saved for display

            vector<int> ids;
            vector<vector<Point2f>> corners;

            aruco::detectMarkers(frame, dictionary, corners, ids, params);

            id0_found = id1_found = id2_found = false;

            if (!ids.empty()) {
                aruco::drawDetectedMarkers(frame, corners, ids, Scalar(0, 255, 0));


                vector<Point2f> c0, c1;
                for (size_t i = 0; i < ids.size(); i++) {
                    Scalar axis_color; string tag_label; double tag_sz;
                    Vec3d rv, tv;

                    if (ids[i] == BASE_TAG_ID_0) {
                        id0_found = true; tag_sz = TAG_SIZE_ID0; c0 = corners[i];
                        axis_color = Scalar(0, 255, 0); tag_label = "ID0";
                    } else if (ids[i] == BASE_TAG_ID_1) {
                        id1_found = true; tag_sz = TAG_SIZE_ID1; c1 = corners[i];
                        axis_color = Scalar(255, 0, 255); tag_label = "ID1";
                    } else if (ids[i] == TARGET_TAG_ID) {
                        id2_found = true; tag_sz = TAG_SIZE_ID2;
                        axis_color = Scalar(0, 0, 255); tag_label = "ID2 (6cm)";
                    } else { continue; }

                    // Per-tag PnP with correct size
                    double h = tag_sz / 2;
                    vector<Point3f> obj = {{-h,h,0},{h,h,0},{h,-h,0},{-h,-h,0}};
                    solvePnP(obj, corners[i], camera_matrix, dist_coeffs, rv, tv, false, SOLVEPNP_IPPE);

                    // ===== DEBUG: record raw PnP Z =====
                    {
                        double _pnpz = tv[2];
                        if (ids[i] == BASE_TAG_ID_0) g_dbg_pnpz0 = _pnpz;
                        else if (ids[i] == TARGET_TAG_ID) g_dbg_pnpz2 = _pnpz;
                    }
                    // ===== END DEBUG =====

                    // ZED depth at tag CORNERS (high contrast → reliable stereo match)
                    // Center has no texture → depth is invalid there
                    float depth_sum = 0; int depth_n = 0;
                    for (auto& p : corners[i]) {
                        int px=(int)p.x, py=(int)p.y;
                        if (px>0 && px<depth_undist.cols && py>0 && py<depth_undist.rows) {
                            float d = depth_undist.at<float>(py, px);
                            if (d > 0.1 && std::isfinite(d)) { depth_sum += d; depth_n++; }
                        }
                    }
                    if (depth_n >= 3) {
                        float d_med = depth_sum / depth_n;
                        // ===== DEBUG: record ZED depth Z =====
                        if (ids[i] == BASE_TAG_ID_0) g_dbg_zedz0 = d_med;
                        else if (ids[i] == TARGET_TAG_ID) g_dbg_zedz2 = d_med;
                        // ===== END DEBUG =====
                        if (std::abs(d_med - tv[2]) < 0.3)
                            tv[2] = tv[2] * 0.3 + d_med * 0.7;
                    }

                    if (ids[i] == BASE_TAG_ID_0) { id0_rvec = rv; id0_tvec = tv; }
                    else if (ids[i] == BASE_TAG_ID_1) { id1_rvec = rv; id1_tvec = tv; }
                    else { id2_rvec = rv; id2_tvec = tv; }

                    aruco::drawAxis(frame, camera_matrix, dist_coeffs, rv, tv, tag_sz * 0.5);
                    
                    putText(frame, tag_label, Point(corners[i][0].x, corners[i][0].y - 10),
                           FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 2);
                }

                // ---- RIGHT CAMERA: independent tag detection ----
                zed.retrieveImage(zed_img_r, sl::VIEW::RIGHT);
                Mat cv_img_r(zed_img_r.getHeight(), zed_img_r.getWidth(), CV_8UC4, zed_img_r.getPtr<sl::uchar1>());
                cvtColor(cv_img_r, frame, COLOR_BGRA2BGR);
                cv::remap(frame, frame, undist_map_rx, undist_map_ry, cv::INTER_LINEAR);

                vector<int> ids_r; vector<vector<Point2f>> corners_r;
                aruco::detectMarkers(frame, dictionary, corners_r, ids_r, params);
                id0_r = id1_r = id2_r = false;
                for (size_t i = 0; i < ids_r.size(); i++) {
                    double tag_sz;
                    if (ids_r[i] == BASE_TAG_ID_0) tag_sz = TAG_SIZE_ID0;
                    else if (ids_r[i] == BASE_TAG_ID_1) tag_sz = TAG_SIZE_ID1;
                    else if (ids_r[i] == TARGET_TAG_ID) tag_sz = TAG_SIZE_ID2;
                    else continue;
                    double h = tag_sz / 2;
                    vector<Point3f> obj = {{-h,h,0},{h,h,0},{h,-h,0},{-h,-h,0}};
                    Vec3d rv, tv;
                    solvePnP(obj, corners_r[i], camera_matrix_r, dist_coeffs, rv, tv, false, SOLVEPNP_IPPE);
                    if (ids_r[i] == BASE_TAG_ID_0) { id0_r=true; id0_tvec_r=tv; id0_rvec_r=rv; }
                    else if (ids_r[i] == BASE_TAG_ID_1) { id1_r=true; id1_tvec_r=tv; id1_rvec_r=rv; }
                    else { id2_r=true; id2_tvec_r=tv; id2_rvec_r=rv; }
                }

                if (id0_found && id2_found) {
                    frame_count++;

                    double scale_factor = 1.0;
                    bool has_scale = false;
                    if (id1_found) {
                        double measured_d01 = norm(id1_tvec - id0_tvec);
                        double known_d01 = sqrt(ID0_TO_ID1_X*ID0_TO_ID1_X +
                                                ID0_TO_ID1_Y*ID0_TO_ID1_Y +
                                                ID0_TO_ID1_Z*ID0_TO_ID1_Z);
                        if (measured_d01 > 0.01 && known_d01 > 0) {
                            scale_factor = known_d01 / measured_d01;
                            if (scale_factor > 0.8 && scale_factor < 1.2) {
                                has_scale = true;
                            } else {
                                scale_factor = 1.0;
                            }
                        }
                    }

                    Mat R_id0 = rvecToMatrix(id0_rvec);
                    Mat R_id2 = rvecToMatrix(id2_rvec);
                    // Average ID0+ID1 rotation (same physical orientation → noise cancels)
                    if (id1_found) {
                        Vec3d rv_avg = (id0_rvec + id1_rvec) * 0.5;
                        R_id0 = rvecToMatrix(rv_avg);  // use averaged rotation as reference
                    }

                    Mat R_rel = R_id0.t() * R_id2;

                    Vec3d t_rel_raw_vec(id2_tvec[0] - id0_tvec[0],
                                        id2_tvec[1] - id0_tvec[1],
                                        id2_tvec[2] - id0_tvec[2]);

                    Mat t_rel_raw = R_id0.t() * Mat(t_rel_raw_vec);

                    // Dual-path: also compute via ID1, average to halve noise
                    if (id1_found) {
                        Mat R_id1 = rvecToMatrix(id1_rvec);
                        Mat R_rel2 = R_id1.t() * R_id2;
                        Vec3d r1 = matrixToRvec(R_rel);
                        Vec3d r2 = matrixToRvec(R_rel2);
                        R_rel = rvecToMatrix((r1 + r2) * 0.5);  // average rotation

                        Mat t2 = R_id1.t() * Mat(Vec3d(id2_tvec[0]-id1_tvec[0],
                                                       id2_tvec[1]-id1_tvec[1],
                                                       id2_tvec[2]-id1_tvec[2]));
                        t2.at<double>(0) += ID0_TO_ID1_X;
                        t2.at<double>(1) += ID0_TO_ID1_Y;
                        t2.at<double>(2) += ID0_TO_ID1_Z;
                        t_rel_raw = (t_rel_raw + t2) * 0.5;  // average translation
                    }

                    // Right camera path: independent PnP → average with left
                    if (id0_r && id2_r) {
                        Mat R_id0_r = rvecToMatrix(id0_rvec_r);
                        Mat R_id2_r = rvecToMatrix(id2_rvec_r);
                        Mat R_rel_r = R_id0_r.t() * R_id2_r;
                        Mat t_r = R_id0_r.t() * Mat(Vec3d(id2_tvec_r[0]-id0_tvec_r[0],
                                                          id2_tvec_r[1]-id0_tvec_r[1],
                                                          id2_tvec_r[2]-id0_tvec_r[2]));

                        // Dual-path for right too (via ID1 if visible)
                        if (id1_r) {
                            Mat R_id1_r = rvecToMatrix(id1_rvec_r);
                            Mat R_rel2_r = R_id1_r.t() * R_id2_r;
                            Vec3d r1r = matrixToRvec(R_rel_r);
                            Vec3d r2r = matrixToRvec(R_rel2_r);
                            R_rel_r = rvecToMatrix((r1r + r2r) * 0.5);
                            Mat t2_r = R_id1_r.t() * Mat(Vec3d(id2_tvec_r[0]-id1_tvec_r[0],
                                                               id2_tvec_r[1]-id1_tvec_r[1],
                                                               id2_tvec_r[2]-id1_tvec_r[2]));
                            t2_r.at<double>(0)+=ID0_TO_ID1_X; t2_r.at<double>(1)+=ID0_TO_ID1_Y; t2_r.at<double>(2)+=ID0_TO_ID1_Z;
                            t_r = (t_r + t2_r) * 0.5;
                        }

                        // Average left+right: rotation (rvec) and translation
                        Vec3d rv_l = matrixToRvec(R_rel);
                        Vec3d rv_r = matrixToRvec(R_rel_r);
                        R_rel = rvecToMatrix((rv_l + rv_r) * 0.5);
                        t_rel_raw = (t_rel_raw + t_r) * 0.5;
                    }

                    Mat t_rel;
                    if (has_scale) {
                        t_rel = scale_factor * t_rel_raw;
                    } else {
                        t_rel = t_rel_raw.clone();
                    }

                    // ===== DEBUG: output CSV to stderr =====
                    {
                        g_dbg_frame++;
                        double relz_raw = t_rel_raw.at<double>(2,0);
                        double relz_final = t_rel.at<double>(2,0);
                        double sf = has_scale ? scale_factor : 1.0;
                        fprintf(stderr, "DBGZ,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                                g_dbg_frame, g_dbg_pnpz0, g_dbg_zedz0, g_dbg_pnpz2, g_dbg_zedz2,
                                relz_raw, sf, relz_final);
                    }
                    // ===== END DEBUG =====

                    Vec3d rel_rvec = matrixToRvec(R_rel);
                    double rel_distance = norm(t_rel);

                    static Vec3d prev_t_rel(0,0,0);
                    static bool prev_init = false;
                    double delta_t = 0;
                    if (prev_init) {
                        delta_t = norm(t_rel - Mat(prev_t_rel)) * 1000.0; // mm
                    }
                    prev_t_rel = Vec3d(t_rel.at<double>(0,0), t_rel.at<double>(1,0), t_rel.at<double>(2,0));
                    prev_init = true;

                    static double stability_score = 1.0;
                    if (delta_t < 5.0) {
                        stability_score = 0.95 * stability_score + 0.05 * 1.0;
                    } else {
                        stability_score = 0.95 * stability_score + 0.05 * 0.0;
                    }

                    relative_filter.add(
                        t_rel.at<double>(0, 0), t_rel.at<double>(1, 0), t_rel.at<double>(2, 0),
                        rel_rvec[0], rel_rvec[1], rel_rvec[2],
                        rel_distance
                    );

                    bool assembly_moving = (delta_t > 5.0);

                    double smooth_x, smooth_y, smooth_z;
                    double smooth_rx, smooth_ry, smooth_rz, smooth_dist;
                    relative_filter.getSmoothed(smooth_x, smooth_y, smooth_z,
                                               smooth_rx, smooth_ry, smooth_rz, smooth_dist);

                    // Extra angle smoothing (EMA) for stability within 0.5°
                    static double arx=0, ary=0, arz=0;
                    static bool aang_init = false;
                    if (!aang_init) { arx=smooth_rx; ary=smooth_ry; arz=smooth_rz; aang_init=true; }
                    const double AA = 0.04;
                    arx = AA * smooth_rx + (1-AA) * arx;
                    ary = AA * smooth_ry + (1-AA) * ary;
                    arz = AA * smooth_rz + (1-AA) * arz;
                    smooth_rx = arx; smooth_ry = ary; smooth_rz = arz;

                    // Compute ID1→ID0 relative pose + filter
                    double r1x=0, r1y=0, r1z=0, r1rx=0, r1ry=0, r1rz=0, r1d=0;
                    if (id1_found) {
                        Mat R_id0_m = rvecToMatrix(id0_rvec);
                        Mat R_id1_m = rvecToMatrix(id1_rvec);
                        Mat t1 = R_id0_m.t() * (Mat(id1_tvec) - Mat(id0_tvec));
                        double raw_x=t1.at<double>(0,0), raw_y=t1.at<double>(1,0), raw_z=t1.at<double>(2,0);
                        double raw_d = sqrt(raw_x*raw_x + raw_y*raw_y + raw_z*raw_z);
                        Mat R_rel1 = R_id0_m.t() * R_id1_m;
                        Vec3d rr1 = matrixToRvec(R_rel1);
                        id1_filter.add(raw_x, raw_y, raw_z, rr1[0], rr1[1], rr1[2], raw_d);
                        double fd;
                        id1_filter.getSmoothed(r1x, r1y, r1z, r1rx, r1ry, r1rz, fd);
                        r1d = sqrt(r1x*r1x + r1y*r1y + r1z*r1z);
                    }

                    // Camera-motion compensation: use ID1→ID0 as baseline reference
                    // When camera moves, ID1→ID0 raw measurement shifts → same shift in ID2→ID0
                    // Subtract the camera-induced shift from ID2→ID0
                    double corr_x = smooth_x, corr_y = smooth_y, corr_z = smooth_z;
                    double corr_rx = smooth_rx, corr_ry = smooth_ry, corr_rz = smooth_rz;

                    // Extra-strong EMA on distance for stability
                    static double dist_ema = 0; static bool dist_init = false;
                    double cur_dist = sqrt(corr_x*corr_x+corr_y*corr_y+corr_z*corr_z);
                    if (!dist_init) { dist_ema = cur_dist; dist_init = true; }
                    dist_ema = 0.15 * cur_dist + 0.85 * dist_ema;
                    smooth_dist = dist_ema;
                    if (id1_found && id1_filter.isStable()) {
                        // Current raw ID1→ID0 (from this frame)
                        double raw1_x = id1_tvec[0] - id0_tvec[0];  // in camera frame
                        double raw1_y = id1_tvec[1] - id0_tvec[1];
                        double raw1_z = id1_tvec[2] - id0_tvec[2];
                        // Camera-induced shift = raw - filtered (filtered is the "truth")
                        // Apply inverse shift to ID2→ID0 smoothed values
                        corr_x -= (raw1_x - r1x);
                        corr_y -= (raw1_y - r1y);
                        corr_z -= (raw1_z - r1z);
                    }

                    if (true) {
                        // Right-eye relative pose for comparison
                        double rrx=0, rry=0, rrz=0;
                        if (id0_r && id2_r) {
                            Mat R0r = rvecToMatrix(id0_rvec_r);
                            Mat tr = R0r.t() * Mat(Vec3d(id2_tvec_r[0]-id0_tvec_r[0],
                                                          id2_tvec_r[1]-id0_tvec_r[1],
                                                          id2_tvec_r[2]-id0_tvec_r[2]));
                            rrx=tr.at<double>(0); rry=tr.at<double>(1); rrz=tr.at<double>(2);
                        }

                        printSystemInfo(id0_found, id1_found, id2_found,
                                       id0_tvec[0], id0_tvec[1], id0_tvec[2],
                                       id1_found ? id1_tvec[0] : 0,
                                       id1_found ? id1_tvec[1] : 0,
                                       id1_found ? id1_tvec[2] : 0,
                                       corr_x, corr_y, corr_z,
                                       corr_rx, corr_ry, corr_rz,
                                       smooth_dist, relative_filter.isStable(),
                                       r1x, r1y, r1z, r1rx, r1ry, r1rz, r1d,
                                       id0_r, id2_r, rrx, rry, rrz);
                    }

                    node->publishRelative(
                        corr_x * 1000, corr_y * 1000, corr_z * 1000,
                        corr_rx, corr_ry, corr_rz,
                        smooth_dist * 1000,
                        id0_found, id1_found, id2_found
                    );

                    Vec3d smooth_rvec_vis(corr_rx, corr_ry, corr_rz);

                    stringstream ss;
                    ss << fixed << setprecision(1);
                    ss << "L: " << sqrt(corr_x*corr_x+corr_y*corr_y+corr_z*corr_z)*1000 << "mm"
                       << (assembly_moving ? " [MOV]" : "");
                    putText(frame, ss.str(), Point(20, 30),
                           FONT_HERSHEY_SIMPLEX, 0.7,
                           assembly_moving ? Scalar(0, 255, 0) : Scalar(0, 255, 255), 2);

                    stringstream ss2;
                    ss2 << "X=" << corr_x*1000
                        << " Y=" << corr_y*1000
                        << " Z=" << corr_z*1000 << " mm";
                    putText(frame_left, ss2.str(), Point(20, 55), 
                           FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 2);

                    if (id1_found) {
                        stringstream ss3;
                        ss3 << fixed << setprecision(1);
                        ss3 << "ID1->ID0: " << r1d * 1000 << "mm (ref " << ID0_TO_ID1_X*1000 << "mm)";
                        putText(frame_left, ss3.str(), Point(20, 75),
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
                        stringstream ss4;
                        ss4 << "  X=" << r1x*1000 << " Y=" << r1y*1000 << " Z=" << r1z*1000 << " mm";
                        putText(frame, ss4.str(), Point(20, 93),
                               FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);
                    } else {
                        putText(frame_left, "ID1: not in view", Point(20, 75),
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    }

                    // Right camera relative pose comparison
                    if (id0_r && id2_r) {
                        Mat R0r = rvecToMatrix(id0_rvec_r);
                        Mat t_rel_r = R0r.t() * Mat(Vec3d(id2_tvec_r[0]-id0_tvec_r[0],
                                                          id2_tvec_r[1]-id0_tvec_r[1],
                                                          id2_tvec_r[2]-id0_tvec_r[2]));
                        double rdist = norm(t_rel_r);
                        stringstream sr;
                        sr << fixed << setprecision(1);
                        sr << "R眼:" << rdist*1000 << "mm X=" << t_rel_r.at<double>(0)*1000
                           << " Y=" << t_rel_r.at<double>(1)*1000
                           << " Z=" << t_rel_r.at<double>(2)*1000;
                        putText(frame_left, sr.str(), Point(20, 105),
                               FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 100), 1);
                    }
                }
            }

            if (!id0_found || !id2_found) {
                putText(frame_left, "请将 ID0(绿) 和 ID2(红) 标签放入视野", Point(20, frame.rows - 30), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }

            putText(frame_left, "ID0(绿)=基准 | ID1(紫)=辅助 | ID2(红)=目标", 
                   Point(20, frame.rows - 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

            imshow("三标签基准系统 (ID0+ID1 -> ID2)", frame_left);
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
