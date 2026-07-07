# ruff: noqa: ANN001, ANN201
# ruff: noqa: ANN001, ANN201
import bisect
from collections import deque
import csv
import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from ship_distance.config import AppConfig


try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


#
from ship_distance.sensor_reader import load_sensor_csv
from ship_distance.video_processor import (
    create_stream_state,
    process_stream_frame,
)
from ship_distance.visualizer import (
    draw_stream_output,
    ensure_bgr_frame,
    make_side_by_side,
)


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)

RECORD_NAME = CONFIG.record.name
RECORD_ROOT = CONFIG.record.root

RGB_VIDEO_PATH = CONFIG.paths.rgb_video
THERMAL_VIDEO_PATH = CONFIG.paths.thermal_video
VIDEO_PATH = RGB_VIDEO_PATH
CSV_PATH = CONFIG.paths.sensor_csv

OUTPUT_DIR = CONFIG.paths.output_dir
OUTPUT_VIDEO_NAME = f"{RECORD_NAME}_rgb_thermal_clean_independent_distance.mp4"

SAVE_OUTPUT_VIDEO = True
SHOW_WINDOW = True
SHOW_TRACK_DETAILS = False
DRAW_HORIZON_LINE = False
MANUAL_HORIZON_BIAS_DEG = 0.0
HORIZON_TILT_ONLY_EMA_STABLE = 0.12
HORIZON_TILT_ONLY_EMA_MOVING = 0.28
HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX = 1.2
HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX = 9.0

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

CAMERA_HEIGHT_M = CONFIG.camera.height_m

DEFAULT_FOV_H_DEG = 65.7
DEFAULT_FOV_V_DEG = 39.9
DEFAULT_THERMAL_FOV_H_DEG = 32.4
DEFAULT_THERMAL_FOV_V_DEG = 24.6
TILT_ZERO_HORIZON_DEG = 90.0

EARTH_RADIUS_M = 6371000.0
REFRACTION_K = 0.13
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M / (1.0 - REFRACTION_K)

HORIZON_DIP_RAD = math.sqrt(2.0 * CAMERA_HEIGHT_M / EFFECTIVE_EARTH_RADIUS_M)
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

MIN_BETA_RAD = math.radians(0.015)
MIN_VALID_DISTANCE_M = 5.0

YOLO_MODEL_PATH = CONFIG.model.yolo_path
YOLO_CONF_FULL = 0.35
YOLO_CONF_DEEP = 0.28
YOLO_IOU_THRES = 0.50
YOLO_IMGSZ_FULL = 960
YOLO_IMGSZ_DEEP = 1280

THERMAL_YOLO_CONF_FULL = 0.30
THERMAL_YOLO_CONF_DEEP = 0.22
THERMAL_YOLO_IMGSZ_FULL = 1280
THERMAL_YOLO_IMGSZ_DEEP = 1280
THERMAL_BLOB_DETECTION_ENABLED = False
THERMAL_BLOB_MIN_AREA = 350
THERMAL_BLOB_MAX_AREA_RATIO = 0.08
THERMAL_BLOB_MIN_ASPECT = 0.55
THERMAL_BLOB_MAX_ASPECT = 14.0
THERMAL_BLOB_BRIGHT_PERCENTILE = 94.0
THERMAL_BLOB_DARK_PERCENTILE = 2.0
THERMAL_BLOB_MIN_CONTRAST = 18.0
THERMAL_DETECT_WHILE_MOVING = False

YOLO_DEVICE = 0 if CUDA_AVAILABLE else "cpu"
YOLO_HALF = True if CUDA_AVAILABLE else False

DETECT_INTERVAL_TRACKING = 6
DETECT_INTERVAL_LOST_FULL = 3
DETECT_INTERVAL_LOST_DEEP = 9
DETECT_INTERVAL_BOTTOM_DEEP = 6

TRACK_MATCH_SCORE_THRES = 0.15
TRACK_MIN_AGE_TO_DISPLAY = 3
TRACK_MIN_CONFIRMED_UPDATES = 2
TRACK_MAX_MISSED_DETECTIONS = 6
TRACK_MAX_STALE_FRAMES = 60
TRACK_DRAW_MAX_STALE_FRAMES = 25

BOX_ALPHA = 0.30
CONF_ALPHA = 0.20

WATERLINE_RATIO_NORMAL = 0.90
WATERLINE_RATIO_ZOOM = 0.86
WATER_HISTORY_LEN = 7

KLT_MAX_CORNERS = 120
KLT_QUALITY_LEVEL = 0.01
KLT_MIN_DISTANCE = 7
KLT_BLOCK_SIZE = 7
KLT_MIN_POINTS = 8
KLT_MAX_STEP_PX = 80.0
KLT_REINIT_EVERY = 12
KLT_TOP_RATIO = 0.08
KLT_BOTTOM_RATIO = 0.78
KLT_SIDE_MARGIN_RATIO = 0.08

RANGE_INIT_SAMPLE_COUNT = 4
RANGE_HISTORY_WINDOW = 25
RANGE_UPDATE_ALPHA_DETECTED = 0.20
RANGE_UPDATE_ALPHA_KLT = 0.06
MAX_ACCEPTED_RAW_JUMP_RATIO = 1.60
RANGE_REJECTS_TO_RELOCK = 12
RANGE_RELATIVE_RATE_PER_SEC = 0.10
RANGE_MIN_RATE_M_PER_SEC = 2.0
RECENT_RAW_WINDOW = 15

OWN_SHIP_BOTTOM_RATIO = 0.90
OWN_SHIP_MIN_HEIGHT_RATIO = 0.30
OWN_SHIP_MAX_AREA_RATIO = 0.40
OWN_SHIP_NEAR_DISTANCE_M = 12.0
OWN_SHIP_NEAR_BOTTOM_RATIO = 0.82

GLOBAL_MAX_CORNERS = 140
GLOBAL_MIN_POINTS = 12
GLOBAL_MAX_FLOW_PX = 150.0
PAN_ACTIVE_FLOW_PX = 18.0
ZOOM_SCALE_EPS = 0.0015
ZOOM_ACTIVE_SCALE = 0.006

HORIZON_BAND_RATIO = 0.14
HORIZON_COLUMN_STEP = 24
HORIZON_MIN_POINTS = 18
HORIZON_MAX_SLOPE = 0.08
HORIZON_FIT_RESIDUAL_PX = 2.8
HORIZON_ROW_STRENGTH_MIN = 10.0
HORIZON_PEAK_RATIO_MIN = 1.55
HORIZON_REFINE_WINDOW = 10
HORIZON_MIN_CONF = 0.42
HORIZON_EMA_VISUAL = 0.035
HORIZON_EMA_TILT = 0.018
HORIZON_BIAS_EMA = 0.012
HORIZON_DETECT_INTERVAL = 5
HORIZON_VISUAL_HOLD_FRAMES = 140
HORIZON_MEDIAN_WINDOW = 21
HORIZON_MAX_VISUAL_JUMP_PX = 18.0
HORIZON_MAX_STEP_STABLE_PX = 0.65
HORIZON_MAX_STEP_MOVING_PX = 3.0
HORIZON_FLOW_ALPHA = 0.18

MERGE_IOU_THRES = 0.22
MERGE_INSIDE_THRES = 0.55
MERGE_HORIZONTAL_OVERLAP_THRES = 0.35
MERGE_VERTICAL_GAP_PX = 180
MERGE_CENTER_DISTANCE_RATIO = 0.82

PANEL_HEIGHT = 158


def detect_horizon_visual(gray, center_y):
    half = int(PROCESS_HEIGHT * HORIZON_BAND_RATIO)
    y1 = int(max(4, center_y - half))
    y2 = int(min(PROCESS_HEIGHT - 4, center_y + half))

    if y2 - y1 < 24:
        return None

    band = cv2.GaussianBlur(gray[y1:y2], (5, 5), 0)
    grad = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3))

    profile = grad.mean(axis=1)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    profile = np.convolve(profile, kernel, mode="same")

    peak_idx = int(np.argmax(profile))
    peak_value = float(profile[peak_idx])
    med_value = float(np.median(profile))

    ratio = peak_value / max(med_value, 1e-6)

    if ratio < HORIZON_PEAK_RATIO_MIN:
        return None

    if peak_value < HORIZON_ROW_STRENGTH_MIN:
        return None

    sub_offset = 0.0

    if 0 < peak_idx < len(profile) - 1:
        left = float(profile[peak_idx - 1])
        right = float(profile[peak_idx + 1])
        denom = left - 2.0 * peak_value + right

        if abs(denom) > 1e-9:
            sub_offset = 0.5 * (left - right) / denom
            sub_offset = max(-1.0, min(1.0, sub_offset))

    row0 = y1 + peak_idx + sub_offset

    conf = max(0.0, min(1.0, (ratio - 1.2) / 1.5))

    lo = max(0, peak_idx - HORIZON_REFINE_WINDOW)
    hi = min(len(profile), peak_idx + HORIZON_REFINE_WINDOW + 1)

    xs = []
    ys = []
    ws = []

    for x_value in range(8, PROCESS_WIDTH - 8, HORIZON_COLUMN_STEP):
        column = grad[lo:hi, x_value - 2 : x_value + 3].mean(axis=1)
        j_value = int(np.argmax(column))
        strength = float(column[j_value])

        if strength < max(4.0, 0.30 * peak_value):
            continue

        xs.append(float(x_value))
        ys.append(float(y1 + lo + j_value))
        ws.append(strength)

    if len(xs) < HORIZON_MIN_POINTS:
        return {"y": float(row0), "slope": 0.0, "conf": conf * 0.7}

    px = np.array(xs)
    py = np.array(ys)
    pw = np.array(ws)

    total_points = len(px)
    slope = 0.0
    intercept = float(row0)
    fit_ok = True

    for _ in range(3):
        if len(px) < HORIZON_MIN_POINTS:
            fit_ok = False
            break

        slope, intercept = np.polyfit(px, py, 1, w=pw)
        residual = np.abs(py - (slope * px + intercept))
        inlier = residual < HORIZON_FIT_RESIDUAL_PX

        if int(np.sum(inlier)) < HORIZON_MIN_POINTS:
            fit_ok = False
            break

        px = px[inlier]
        py = py[inlier]
        pw = pw[inlier]

    if not fit_ok or abs(slope) > HORIZON_MAX_SLOPE:
        return {"y": float(row0), "slope": 0.0, "conf": conf * 0.7}

    inlier_frac = len(px) / max(1, total_points)
    conf = conf * (0.5 + 0.5 * inlier_frac)

    return {
        "y": float(slope * CX + intercept),
        "slope": float(slope),
        "conf": float(conf),
    }


def main():
    if not YOLO_AVAILABLE:
        print("Ultralytics kurulu degil.")
        print("Kurulum: pip install ultralytics")
        return

    rgb_file = Path(RGB_VIDEO_PATH)
    thermal_file = Path(THERMAL_VIDEO_PATH)

    if not rgb_file.exists():
        print(f"RGB video bulunamadi: {RGB_VIDEO_PATH}")
        return

    if not thermal_file.exists():
        print(f"Thermal video bulunamadi: {THERMAL_VIDEO_PATH}")
        return

    rgb_sensor_rows = load_sensor_csv(CSV_PATH, channel="rgb")
    thermal_sensor_rows = load_sensor_csv(CSV_PATH, channel="thermal")

    print("YOLO modeli yukleniyor...")
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO modeli hazir.")
    print(f"Device: {YOLO_DEVICE} | Half: {YOLO_HALF}")
    print(
        f"Kamera yuksekligi: {CAMERA_HEIGHT_M} m | "
        f"Ufuk cukurlugu: {math.degrees(HORIZON_DIP_RAD):.4f} deg | "
        f"Maks. deniz mesafesi: {MAX_SEA_DISTANCE_M / 1000.0:.2f} km"
    )

    rgb_cap = cv2.VideoCapture(str(rgb_file))
    thermal_cap = cv2.VideoCapture(str(thermal_file))

    if not rgb_cap.isOpened():
        print(f"RGB video acilamadi: {RGB_VIDEO_PATH}")
        return

    if not thermal_cap.isOpened():
        print(f"Thermal video acilamadi: {THERMAL_VIDEO_PATH}")
        rgb_cap.release()
        return

    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    thermal_fps = thermal_cap.get(cv2.CAP_PROP_FPS)

    if rgb_fps is None or rgb_fps <= 1:
        rgb_fps = 25.0

    if thermal_fps is None or thermal_fps <= 1:
        thermal_fps = rgb_fps

    video_fps = rgb_fps

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / OUTPUT_VIDEO_NAME

    writer = None

    if SAVE_OUTPUT_VIDEO:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (PROCESS_WIDTH * 2, PROCESS_HEIGHT),
        )

    window_name = "RGB + THERMAL HORIZON-LOCKED DISTANCE"

    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    rgb_state = create_stream_state("RGB", "rgb")
    thermal_state = create_stream_state("THERMAL", "thermal")

    frame_index = 0
    previous_tick = cv2.getTickCount()

    print("Video isleniyor...")
    print(f"RGB     : {RGB_VIDEO_PATH}")
    print(f"Thermal : {THERMAL_VIDEO_PATH}")
    print(f"CSV     : {CSV_PATH}")
    print(f"Output  : {output_video_path}")
    print("Cikis: q")

    while True:
        rgb_ret, rgb_frame = rgb_cap.read()
        thermal_ret, thermal_frame = thermal_cap.read()

        if not rgb_ret and not thermal_ret:
            break

        if not rgb_ret:
            rgb_frame = None

        if not thermal_ret:
            thermal_frame = None

        current_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / max(current_tick - previous_tick, 1)
        previous_tick = current_tick

        rgb_output, rgb_moving = process_stream_frame(
            rgb_frame,
            rgb_state,
            rgb_sensor_rows,
            model,
            frame_index,
            video_fps,
        )
        thermal_output, thermal_moving = process_stream_frame(
            thermal_frame,
            thermal_state,
            thermal_sensor_rows,
            model,
            frame_index,
            video_fps,
        )

        draw_stream_output(
            rgb_output,
            rgb_state,
            rgb_sensor_rows,
            fps,
            frame_index,
            video_fps,
            rgb_moving,
        )
        draw_stream_output(
            thermal_output,
            thermal_state,
            thermal_sensor_rows,
            fps,
            frame_index,
            video_fps,
            thermal_moving,
        )

        combined = make_side_by_side(rgb_output, thermal_output)

        if writer is not None:
            writer.write(combined)

        if SHOW_WINDOW:
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
        elif frame_index % 200 == 0:
            print(
                f"frame={frame_index} | "
                f"rgb_tracks={len(rgb_state['tracks'])} | "
                f"thermal_tracks={len(thermal_state['tracks'])} | "
                f"rgb_mode={rgb_state['mode']} | "
                f"thermal_mode={thermal_state['mode']}"
            )

        frame_index += 1

    rgb_cap.release()
    thermal_cap.release()

    if writer is not None:
        writer.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("Bitti.")

    if SAVE_OUTPUT_VIDEO:
        print(f"Kaydedilen video: {output_video_path}")


if __name__ == "__main__":
    main()
