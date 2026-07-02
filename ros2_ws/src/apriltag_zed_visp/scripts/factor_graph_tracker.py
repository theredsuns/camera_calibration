#!/usr/bin/env python3
"""
Factor-graph optimized AprilTag 3-tag relative pose tracker.
Sliding-window GTSAM optimization with known ID0-ID1 geometry constraint.
"""
import sys, os, time, re
import numpy as np
import cv2
import gtsam
from gtsam import Pose3, Rot3, Point3
from gtsam.symbol_shorthand import X, R
from collections import deque, namedtuple

# === Config ===
TAG_SIZE_ID0 = 0.06
TAG_SIZE_ID1 = 0.06
TAG_SIZE_ID2 = 0.06
BASE_ID0, BASE_ID1, TARGET_ID2 = 0, 1, 2
D01_TRUE = np.array([0.1587, 0.0, 0.0])
WINDOW_SIZE = 10
CALIB_FILE = "/home/nkk/coordate_change/ros2_ws/src/apriltag_zed_visp/config/zed_calibration.yaml"

TagObs = namedtuple('TagObs', ['tag_id','corners','rvec','tvec','reproj_err'])

def load_calib(path):
    with open(path) as f:
        text = f.read()
    blocks = text.split('camera_matrix:')
    ms = list(re.finditer(r'data:\s*\[([^\]]+)\]', blocks[1]))
    K = np.array([float(x) for x in ms[0].group(1).split(',')], dtype=np.float64).reshape(3,3)
    db = text.split('distortion_coefficients:')[1]
    dm = list(re.finditer(r'data:\s*\[([^\]]+)\]', db))
    D = np.array([float(x) for x in dm[0].group(1).split(',')], dtype=np.float64)
    w = int(re.search(r'image_width:\s*(\d+)', text).group(1))
    h = int(re.search(r'image_height:\s*(\d+)', text).group(1))
    return K, D, (w, h)

def detect_tags(frame, dictionary, params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    r = []
    if ids is not None:
        for i, c in zip(ids.flatten(), corners):
            r.append((int(i), c[0]))
    return r

def pnp_tag(corners, tag_size, K, dist):
    h = tag_size / 2
    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None, 999
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.mean(np.linalg.norm(corners.reshape(-1,2) - proj.reshape(-1,2), axis=1))
    return rvec.flatten(), tvec.flatten(), err

def pose_to_gtsam(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return Pose3(Rot3(R), Point3(tvec[0], tvec[1], tvec[2]))

def compute_rel(rvec0, tvec0, rvec2, tvec2):
    R0, _ = cv2.Rodrigues(rvec0)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = R0.T @ R2
    t_rel = R0.T @ (tvec2 - tvec0)
    r_rel, _ = cv2.Rodrigues(R_rel)
    return r_rel.flatten(), t_rel

def check_id1(obs0, obs1):
    if obs0 is None or obs1 is None:
        return False
    R0, _ = cv2.Rodrigues(obs0.rvec)
    t = R0.T @ (obs1.tvec - obs0.tvec)
    return abs(np.linalg.norm(t) - np.linalg.norm(D01_TRUE))*1000 < 5.0

def optimize(window, init_rel_t):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pnp_s = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.002]*3 + [0.02]*3))
    smooth_s = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.003]*3 + [0.03]*3))
    rel_s = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001]*3 + [0.01]*3))

    initial.insert(R(0), pose_to_gtsam(np.zeros(3), init_rel_t))

    valid = []
    for i, fd in enumerate(window):
        if fd['id0'] is None or fd['id2'] is None:
            continue
        valid.append(i)
        p0 = pose_to_gtsam(fd['id0'].rvec, fd['id0'].tvec)
        initial.insert(X(i), p0)
        w = max(0.2, 1.0 - fd['id0'].reproj_err / 3.0)
        graph.add(gtsam.PriorFactorPose3(X(i), p0,
                  gtsam.noiseModel.Diagonal.Sigmas(np.array([0.003]*3 + [0.03]*3) / w)))
        _, rel_t = compute_rel(fd['id0'].rvec, fd['id0'].tvec,
                               fd['id2'].rvec, fd['id2'].tvec)
        graph.add(gtsam.PriorFactorPose3(R(0), pose_to_gtsam(np.zeros(3), rel_t), rel_s))

    for j in range(len(valid)-1):
        a, b = valid[j], valid[j+1]
        graph.add(gtsam.BetweenFactorPose3(X(a), X(b), Pose3(), smooth_s))

    if len(valid) < 2:
        return init_rel_t, 999

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = opt.optimize()
    opt_pose = result.atPose3(R(0))
    t = opt_pose.translation()
    return np.array([t[0], t[1], t[2]]), opt.error()

def main():
    print("Factor Graph AprilTag Tracker (GTSAM)")
    import pyzed.sl as sl

    calib_K, calib_D, (img_w, img_h) = load_calib(CALIB_FILE)
    print(f"Calib: fx={calib_K[0,0]:.1f} fy={calib_K[1,1]:.1f}")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.camera_fps = 30
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed"); return
    print("ZED opened")

    info = zed.get_camera_information()
    zp = info.camera_configuration.calibration_parameters.left_cam
    zed_K = np.array([[zp.fx,0,zp.cx],[0,zp.fy,zp.cy],[0,0,1]], dtype=np.float64)
    zed_D = np.array(zp.disto[:5], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(zed_K, zed_D, None, calib_K, (img_w,img_h), cv2.CV_16SC2)

    D_zero = np.zeros(5, dtype=np.float64)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_p = cv2.aruco.DetectorParameters_create()
    aruco_p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_p.cornerRefinementMaxIterations = 50

    window = deque(maxlen=WINDOW_SIZE)
    ema_t, ema_a = None, 0.15
    zed_img = sl.Mat()
    cv2.namedWindow("Factor Graph Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Factor Graph Tracker", 1280, 720)

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(zed_img, sl.VIEW.LEFT)
        frame = zed_img.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        detections = detect_tags(frame, dictionary, aruco_p)
        obs = {BASE_ID0: None, BASE_ID1: None, TARGET_ID2: None}
        for tid, corners in detections:
            if tid not in obs: continue
            sz = {BASE_ID0:TAG_SIZE_ID0, BASE_ID1:TAG_SIZE_ID1, TARGET_ID2:TAG_SIZE_ID2}[tid]
            rv, tv, err = pnp_tag(corners, sz, calib_K, D_zero)
            if rv is not None and err < 0.015:
                obs[tid] = TagObs(tid, corners, rv, tv, err)
                cv2.polylines(frame, [corners.astype(int)], True,
                             {BASE_ID0:(255,0,0), BASE_ID1:(255,0,255), TARGET_ID2:(0,0,255)}[tid], 2)

        window.append({'id0': obs[BASE_ID0], 'id1': obs[BASE_ID1], 'id2': obs[TARGET_ID2]})
        ok = check_id1(obs[BASE_ID0], obs[BASE_ID1])
        raw_t = None

        if obs[BASE_ID0] is not None and obs[TARGET_ID2] is not None:
            _, raw_t = compute_rel(obs[BASE_ID0].rvec, obs[BASE_ID0].tvec,
                                    obs[TARGET_ID2].rvec, obs[TARGET_ID2].tvec)

            if len(window) >= 3:
                opt_t, err = optimize(list(window), raw_t)
                if ema_t is None: ema_t = opt_t
                else: ema_t = ema_a * opt_t + (1-ema_a) * ema_t
                rd = np.linalg.norm(raw_t)*1000
                od = np.linalg.norm(ema_t)*1000
                cv2.putText(frame, f"RAW: {rd:.1f}mm [{raw_t[0]*1000:.1f} {raw_t[1]*1000:.1f} {raw_t[2]*1000:.1f}]",
                           (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)
                cv2.putText(frame, f"FG:  {od:.1f}mm [{ema_t[0]*1000:.1f} {ema_t[1]*1000:.1f} {ema_t[2]*1000:.1f}] err={err:.1f}",
                           (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"ID1: {'OK' if ok else 'BAD'}  W={len(window)}",
                           (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if ok else (0,0,255), 1)
        else:
            cv2.putText(frame, "Waiting for tags...", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Factor Graph Tracker", frame)
        if cv2.waitKey(10) == 27:
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
