#!/usr/bin/env python3
"""
Factor-graph optimized AprilTag 3-tag relative pose tracker.
Uses GTSAM for sliding-window optimization with known ID0-ID1 geometry constraint.
"""

import sys, time, os
import numpy as np
import cv2
import gtsam
from gtsam import Pose3, Rot3, Point3
from gtsam.symbol_shorthand import X, R
from collections import deque
from dataclasses import dataclass

# === Config ===
TAG_SIZE_ID0 = 0.06  # 6cm
TAG_SIZE_ID1 = 0.06
TAG_SIZE_ID2 = 0.06
BASE_ID0, BASE_ID1, TARGET_ID2 = 0, 1, 2
D01_TRUE = np.array([0.1587, 0.0, 0.0])  # ID1 position in ID0 frame
WINDOW_SIZE = 10       # sliding window frames
PX_THRESH = 0.015      # outlier: reprojection error > 1.5px = suspicious

# Calibration file
CALIB_FILE = "/home/nkk/coordate_change/ros2_ws/src/apriltag_zed_visp/config/zed_calibration.yaml"

@dataclass
class TagObs:
    """Single tag detection in one frame"""
    tag_id: int
    corners: np.ndarray  # 4x2 image points
    rvec: np.ndarray     # PnP rotation vector
    tvec: np.ndarray     # PnP translation vector
    reproj_err: float    # reprojection error in pixels

def load_calibration(path):
    """Load camera matrix from YAML, returns K, dist, img_size"""
    with open(path) as f:
        text = f.read()
    import re
    data_match = re.search(r'data:\s*\[([^\]]+)\]', text)
    # Find camera_matrix data block
    blocks = text.split('camera_matrix:')
    cm_block = blocks[1] if len(blocks) > 1 else ''
    data_matches = list(re.finditer(r'data:\s*\[([^\]]+)\]', cm_block))
    if not data_matches:
        # fallback
        data_matches = list(re.finditer(r'data:\s*\[([^\]]+)\]', text))
        # First block is camera_matrix
        K_vals = [float(x) for x in data_matches[0].group(1).split(',')]
    else:
        K_vals = [float(x) for x in data_matches[0].group(1).split(',')]
    # Distortion
    dist_blocks = text.split('distortion_coefficients:')
    dist_matches = list(re.finditer(r'data:\s*\[([^\]]+)\]', dist_blocks[1]))
    D_vals = [float(x) for x in dist_matches[0].group(1).split(',')]

    # Image size
    w_match = re.search(r'image_width:\s*(\d+)', text)
    h_match = re.search(r'image_height:\s*(\d+)', text)
    w = int(w_match.group(1)) if w_match else 1280
    h = int(h_match.group(1)) if h_match else 720

    K = np.array([[K_vals[0], 0, K_vals[2]],
                   [0, K_vals[4], K_vals[5]],
                   [0, 0, 1]], dtype=np.float64)
    D = np.array(D_vals, dtype=np.float64)
    return K, D, (w, h)

def detect_tags(frame, dictionary, params):
    """Detect tags, return list of (id, corners)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    result = []
    if ids is not None:
        for i, c in zip(ids.flatten(), corners):
            result.append((i, c[0]))
    return result

def pnp_tag(corners, tag_size, K, dist):
    """PnP for a single tag. Returns (rvec, tvec, reproj_err). dist must be zero if image is rectified."""
    h = tag_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None, 999
    # Reprojection error
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.mean(np.linalg.norm(corners.reshape(-1,2) - proj.reshape(-1,2), axis=1))
    return rvec.flatten(), tvec.flatten(), err

def pose_to_gtsam(rvec, tvec):
    """Convert OpenCV rvec/tvec to gtsam Pose3"""
    R, _ = cv2.Rodrigues(rvec)
    rot = Rot3(R)
    trans = Point3(tvec[0], tvec[1], tvec[2])
    return Pose3(rot, trans)

def gtsam_to_vec(pose):
    """gtsam Pose3 → (rvec, tvec)"""
    R = pose.rotation().matrix()
    rvec, _ = cv2.Rodrigues(R)
    t = pose.translation()
    return rvec.flatten(), np.array([t[0], t[1], t[2]])

def optimize_window(window, init_id0_id2, init_cam_id0):
    """
    Factor graph optimization for sliding window.

    Variables:
      X(i): camera→ID0 pose at frame i
      R:    ID0→ID2 relative pose (shared across all frames)

    Factors:
      Prior on X(i): from PnP (weighted by reprojection quality)
      Between X(i), X(i+1): smoothness prior (camera can't teleport)
      Prior on R: from per-frame direct measurement
      Implicit: ID1 is at D01_TRUE from ID0 (used as validation, not variable)
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Noise models
    pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.002, 0.01, 0.01, 0.02]))
    smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.002, 0.002, 0.005, 0.02, 0.02, 0.05]))
    rel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0005, 0.0005, 0.001, 0.005, 0.005, 0.01]))

    # Initial values
    rel_pose = pose_to_gtsam(np.zeros(3), init_id0_id2)
    initial.insert(R(0), rel_pose)

    valid_frames = []
    for i, frame_data in enumerate(window):
        if frame_data['id0'] is None or frame_data['id2'] is None:
            continue
        valid_frames.append(i)
        cam_id0 = pose_to_gtsam(frame_data['id0'].rvec, frame_data['id0'].tvec)
        initial.insert(X(i), cam_id0)

        # Prior on camera→ID0
        weight = max(0.1, 1.0 - frame_data['id0'].reproj_err / 5.0)
        graph.add(gtsam.PriorFactorPose3(X(i), cam_id0,
                  gtsam.noiseModel.Diagonal.Sigmas(np.array([0.002]*3 + [0.02]*3) / weight)))

        # Direct measurement of ID0→ID2
        id2_rel_rvec, id2_rel_tvec = compute_relative(
            frame_data['id0'].rvec, frame_data['id0'].tvec,
            frame_data['id2'].rvec, frame_data['id2'].tvec)
        direct_rel = pose_to_gtsam(np.zeros(3), id2_rel_tvec)
        graph.add(gtsam.PriorFactorPose3(R(0), direct_rel, rel_noise))

    # Smoothness between consecutive frames
    for i in range(len(valid_frames)-1):
        fi, fj = valid_frames[i], valid_frames[i+1]
        graph.add(gtsam.BetweenFactorPose3(X(fi), X(fj), Pose3(), smooth_noise))

    if len(valid_frames) < 2:
        return init_id0_id2, 999

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(20)
    params.setRelativeErrorTol(1e-5)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    error = optimizer.error()

    # Extract optimized ID0→ID2
    opt_rel = result.atPose3(R(0))
    _, opt_t = gtsam_to_vec(opt_rel)
    return opt_t, error

def compute_relative(rvec0, tvec0, rvec2, tvec2):
    """Compute ID2 position in ID0 frame"""
    R0, _ = cv2.Rodrigues(rvec0)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = R0.T @ R2
    t_rel = R0.T @ (tvec2 - tvec0)
    r_rel, _ = cv2.Rodrigues(R_rel)
    return r_rel.flatten(), t_rel

def validate_id1(id0_obs, id1_obs):
    """Check if ID0-ID1 measured distance matches truth"""
    if id0_obs is None or id1_obs is None:
        return False
    R0, _ = cv2.Rodrigues(id0_obs.rvec)
    t1_in_id0 = R0.T @ (id1_obs.tvec - id0_obs.tvec)
    measured_dist = np.linalg.norm(t1_in_id0)
    true_dist = np.linalg.norm(D01_TRUE)
    err = abs(measured_dist - true_dist) * 1000  # mm
    return err < 5.0  # within 5mm = this frame is trustworthy

def main():
    print("Factor Graph AprilTag Tracker (GTSAM)")
    print(f"Window: {WINDOW_SIZE} frames")

    # Load calibration
    K, dist, (img_w, img_h) = load_calibration(CALIB_FILE)
    print(f"Calib: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # ZED camera
    import pyzed.sl as sl
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.camera_fps = 30
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {err}")
        return
    print("ZED opened")

    # Undistortion map
    zed_params = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam
    zed_K = np.array([[zed_params.fx, 0, zed_params.cx],
                       [0, zed_params.fy, zed_params.cy],
                       [0, 0, 1]], dtype=np.float64)
    zed_D = np.array([zed_params.disto[0], zed_params.disto[1],
                       zed_params.disto[2], zed_params.disto[3], zed_params.disto[4]])
    map_x, map_y = cv2.initUndistortRectifyMap(zed_K, zed_D, None, K, (img_w, img_h), cv2.CV_16SC2)
    # Use calibrated K, zero distortion for PnP
    K_pnp = K
    D_pnp = np.zeros(5)

    # ArUco
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.cornerRefinementMaxIterations = 50

    # Sliding window
    window = deque(maxlen=WINDOW_SIZE)
    frame_idx = 0

    # EMA filter for optimized output (light smoothing)
    ema_t = None
    ema_alpha = 0.15

    # Display
    cv2.namedWindow("Factor Graph Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Factor Graph Tracker", 1280, 720)

    zed_img = sl.Mat()

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieveImage(zed_img, sl.VIEW.LEFT)

        frame = zed_img.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Detect tags
        detections = detect_tags(frame, dictionary, aruco_params)
        obs = {BASE_ID0: None, BASE_ID1: None, TARGET_ID2: None}

        for tag_id, corners in detections:
            if tag_id not in obs:
                continue
            sz = {BASE_ID0: TAG_SIZE_ID0, BASE_ID1: TAG_SIZE_ID1, TARGET_ID2: TAG_SIZE_ID2}[tag_id]
            rvec, tvec, err = pnp_tag(corners, sz, K_pnp, D_pnp)
            if rvec is not None and err < PX_THRESH:
                obs[tag_id] = TagObs(tag_id, corners, rvec, tvec, err)

        # Add to window
        window.append({'id0': obs[BASE_ID0], 'id1': obs[BASE_ID1], 'id2': obs[TARGET_ID2]})
        frame_idx += 1

        # Validate with ID1 constraint
        frame_ok = validate_id1(obs[BASE_ID0], obs[BASE_ID1])

        # Compute raw relative pose
        raw_t = None
        if obs[BASE_ID0] is not None and obs[TARGET_ID2] is not None:
            _, raw_t = compute_relative(obs[BASE_ID0].rvec, obs[BASE_ID0].tvec,
                                         obs[TARGET_ID2].rvec, obs[TARGET_ID2].tvec)

        # Factor graph optimization
        if raw_t is not None and len(window) >= 3:
            init_cam = obs[BASE_ID0].tvec if obs[BASE_ID0] else np.zeros(3)
            opt_t, error = optimize_window(list(window), raw_t, init_cam)

            # EMA smoothing
            if ema_t is None:
                ema_t = opt_t
            else:
                ema_t = ema_alpha * opt_t + (1 - ema_alpha) * ema_t

            opt_dist = np.linalg.norm(ema_t) * 1000
            raw_dist = np.linalg.norm(raw_t) * 1000

            # Display
            cv2.putText(frame, f"RAW:  {raw_dist:.1f}mm  [{raw_t[0]*1000:.1f} {raw_t[1]*1000:.1f} {raw_t[2]*1000:.1f}]",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
            cv2.putText(frame, f"FG:    {opt_dist:.1f}mm  [{ema_t[0]*1000:.1f} {ema_t[1]*1000:.1f} {ema_t[2]*1000:.1f}]  err={error:.1f}",
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status = "OK" if frame_ok else "BAD"
            color = (0, 255, 0) if frame_ok else (0, 0, 255)
            cv2.putText(frame, f"ID1 check: {status}  Window: {len(window)}/{WINDOW_SIZE}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Tag labels
            if obs[BASE_ID0]:
                c = obs[BASE_ID0].corners[0].astype(int)
                cv2.putText(frame, "ID0", (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            if obs[BASE_ID1]:
                c = obs[BASE_ID1].corners[0].astype(int)
                cv2.putText(frame, "ID1", (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            if obs[TARGET_ID2]:
                c = obs[TARGET_ID2].corners[0].astype(int)
                cv2.putText(frame, "ID2", (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        else:
            cv2.putText(frame, "Waiting for tags...", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Factor Graph Tracker", frame)
        if cv2.waitKey(10) == 27:
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
