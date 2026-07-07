# ruff: noqa: ANN001, ANN201
import bisect
from collections import deque
import csv
import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np


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


# RECORD_NAME ="2025_05_25-18_01_10"
# RECORD_NAME = "2025_05_27-17_41_33"
# RECORD_NAME ="2025_05_25-21_39_25"
# RECORD_NAME ="2025_05_16-11_56_52"
# RECORD_NAME = "2025_05_24-13_18_01"
# RECORD_NAME = "2025_05_30-15_55_55"
# RECORD_NAME = "2025_05_22-10_05_16"
RECORD_NAME = "2025_05_25-21_38_27"
RECORD_ROOT = "/home/tuana/records_work/Records_all"

RGB_VIDEO_PATH = f"{RECORD_ROOT}/{RECORD_NAME}/rgb.mp4"
THERMAL_VIDEO_PATH = f"{RECORD_ROOT}/{RECORD_NAME}/thermal.mp4"
VIDEO_PATH = RGB_VIDEO_PATH
CSV_PATH = f"{RECORD_ROOT}/{RECORD_NAME}/sensor_data.csv"

OUTPUT_DIR = "/home/tuana/video_distance_outputs"
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

CAMERA_HEIGHT_M = 10.0

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

YOLO_MODEL_PATH = "yolov8x.pt"
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


def parse_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def find_column(fieldnames, possible_names):
    if not fieldnames:
        return None

    lowered = {name.lower().strip(): name for name in fieldnames}

    for possible in possible_names:
        key = possible.lower().strip()

        if key in lowered:
            return lowered[key]

    for name in fieldnames:
        low = name.lower().strip()

        for possible in possible_names:
            if possible.lower().strip() in low:
                return name

    return None


def parse_time_to_seconds(value, first_absolute_time=None):
    if value is None:
        return None, first_absolute_time

    text = str(value).strip()

    if text == "":
        return None, first_absolute_time

    numeric = parse_float(text)

    if numeric is not None:
        return numeric, first_absolute_time

    parts = text.split()
    time_part = parts[-1] if len(parts) >= 2 else text
    chunks = time_part.split(":")

    if len(chunks) < 2:
        return None, first_absolute_time

    try:
        hour = float(chunks[0])
        minute = float(chunks[1])
        second = float(chunks[2]) if len(chunks) >= 3 else 0.0

        absolute_seconds = hour * 3600.0 + minute * 60.0 + second

        if first_absolute_time is None:
            first_absolute_time = absolute_seconds

        relative_seconds = absolute_seconds - first_absolute_time

        if relative_seconds < 0:
            relative_seconds = 0.0

        return relative_seconds, first_absolute_time

    except ValueError:
        return None, first_absolute_time


def normalize_fov(value, default_value):
    if value is None:
        return default_value

    if 0.01 < value < 3.2:
        value = math.degrees(value)

    if value <= 0.01:
        return default_value

    if value > 120.0:
        return default_value

    return value


def channel_column_names(base_names, channel):
    names = []

    channel_aliases = {
        "rgb": ["rgb", "visible", "vis", "color"],
        "thermal": ["thermal", "therm", "ir", "tir", "th"],
    }

    for base_name in base_names:
        for alias in channel_aliases.get(channel, []):
            names.extend(
                [
                    f"{base_name}_{alias}",
                    f"{alias}_{base_name}",
                    f"{base_name}{alias}",
                    f"{alias}{base_name}",
                ]
            )

    names.extend(base_names)

    unique_names = []
    seen = set()

    for name in names:
        key = name.lower().strip()

        if key not in seen:
            unique_names.append(name)
            seen.add(key)

    return unique_names


def channel_default_fov(channel):
    if channel == "thermal":
        return DEFAULT_THERMAL_FOV_H_DEG, DEFAULT_THERMAL_FOV_V_DEG

    return DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG


def load_sensor_csv(csv_path, channel="rgb"):
    csv_file = Path(csv_path)
    default_fov_h, default_fov_v = channel_default_fov(channel)

    if not csv_file.exists():
        print(f"CSV bulunamadi: {csv_path}")
        return []

    with csv_file.open("r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;	")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)
        fieldnames = reader.fieldnames

        print(f"CSV kolonlari ({channel}):", fieldnames)

        time_col = find_column(
            fieldnames,
            [
                "video_time",
                "time_sec",
                "seconds",
                "second",
                "timestamp",
                "time",
                "datetime",
            ],
        )

        fov_h_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "fov_h",
                    "hfov",
                    "horizontal_fov",
                    "fov_horizontal",
                    "camera_fov_h",
                    "cam_fov_h",
                ],
                channel,
            ),
        )

        fov_v_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "fov_v",
                    "vfov",
                    "vertical_fov",
                    "fov_vertical",
                    "camera_fov_v",
                    "cam_fov_v",
                ],
                channel,
            ),
        )

        zoom_col = find_column(
            fieldnames,
            channel_column_names(
                ["zoom", "zoom_value", "camera_zoom", "cam_zoom"], channel
            ),
        )

        tilt_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "tilt",
                    "camera_tilt",
                    "ptz_tilt",
                    "tilt_angle",
                    "pitch",
                    "camera_pitch",
                ],
                channel,
            ),
        )

        roll_col = find_column(
            fieldnames,
            channel_column_names(
                ["roll", "camera_roll", "roll_angle"], channel
            ),
        )

        pan_col = find_column(
            fieldnames,
            channel_column_names(
                ["pan", "camera_pan", "ptz_pan", "pan_angle", "yaw"], channel
            ),
        )

        print(f"time kolonu ({channel}):", time_col)
        print(f"fov_h kolonu ({channel}):", fov_h_col)
        print(f"fov_v kolonu ({channel}):", fov_v_col)
        print(f"zoom kolonu ({channel}):", zoom_col)
        print(f"tilt kolonu ({channel}):", tilt_col)
        print(f"roll kolonu ({channel}):", roll_col)
        print(f"pan kolonu ({channel}):", pan_col)

        rows = []
        first_absolute_time = None

        for index, row in enumerate(reader):
            if time_col:
                second, first_absolute_time = parse_time_to_seconds(
                    row.get(time_col), first_absolute_time
                )
            else:
                second = float(index)

            if second is None:
                second = float(index)

            fov_h = parse_float(row.get(fov_h_col)) if fov_h_col else None
            fov_v = parse_float(row.get(fov_v_col)) if fov_v_col else None
            zoom = parse_float(row.get(zoom_col)) if zoom_col else None
            tilt = parse_float(row.get(tilt_col)) if tilt_col else None
            roll = parse_float(row.get(roll_col)) if roll_col else None
            pan = parse_float(row.get(pan_col)) if pan_col else None

            rows.append(
                {
                    "second": float(second),
                    "fov_h": normalize_fov(fov_h, default_fov_h),
                    "fov_v": normalize_fov(fov_v, default_fov_v),
                    "zoom": zoom,
                    "tilt": tilt,
                    "roll": roll,
                    "pan": pan,
                }
            )

    rows.sort(key=lambda item: item["second"])

    print(f"Okunan sensor satiri ({channel}): {len(rows)}")

    return rows


def interpolate_value(a_value, b_value, ratio):
    if a_value is None and b_value is None:
        return None

    if a_value is None:
        return b_value

    if b_value is None:
        return a_value

    return a_value + (b_value - a_value) * ratio


def get_sensor_for_time(sensor_rows, video_second):
    if not sensor_rows:
        return {
            "fov_h": DEFAULT_FOV_H_DEG,
            "fov_v": DEFAULT_FOV_V_DEG,
            "zoom": None,
            "tilt": None,
            "roll": None,
            "pan": None,
            "source": "DEFAULT",
        }

    if len(sensor_rows) == 1:
        row = sensor_rows[0]

        return {**row, "source": "CSV"}

    seconds = [row["second"] for row in sensor_rows]
    idx = bisect.bisect_right(seconds, video_second) - 1

    if idx < 0:
        idx = 0

    if idx >= len(sensor_rows) - 1:
        row = sensor_rows[-1]

        return {**row, "source": "CSV"}

    row_a = sensor_rows[idx]
    row_b = sensor_rows[idx + 1]

    time_a = row_a["second"]
    time_b = row_b["second"]

    if time_b <= time_a:
        ratio = 0.0
    else:
        ratio = (video_second - time_a) / (time_b - time_a)
        ratio = max(0.0, min(1.0, ratio))

    return {
        "fov_h": interpolate_value(row_a["fov_h"], row_b["fov_h"], ratio),
        "fov_v": interpolate_value(row_a["fov_v"], row_b["fov_v"], ratio),
        "zoom": interpolate_value(row_a["zoom"], row_b["zoom"], ratio),
        "tilt": interpolate_value(row_a["tilt"], row_b["tilt"], ratio),
        "roll": interpolate_value(row_a["roll"], row_b["roll"], ratio),
        "pan": interpolate_value(row_a["pan"], row_b["pan"], ratio),
        "source": "CSV_INTERP",
    }


def smooth_sensor(previous_sensor, new_sensor):
    if previous_sensor is None:
        return new_sensor.copy()

    alphas = {
        "fov_h": 0.45,
        "fov_v": 0.45,
        "zoom": 0.45,
        "tilt": 0.20,
        "roll": 0.20,
        "pan": 0.35,
    }

    smoothed = new_sensor.copy()

    for key, alpha in alphas.items():
        old = previous_sensor.get(key)
        new = new_sensor.get(key)

        if old is None:
            smoothed[key] = new
        elif new is None:
            smoothed[key] = old
        else:
            smoothed[key] = (1.0 - alpha) * old + alpha * new

    smoothed["source"] = new_sensor.get("source", "CSV")

    return smoothed


def focal_from_fov(fov_h_deg, fov_v_deg):
    fx_value = (PROCESS_WIDTH / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    fy_value = (PROCESS_HEIGHT / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)

    return fx_value, fy_value


def resolve_pitch_down_from_tilt(tilt_deg):
    if tilt_deg is None:
        return 0.0

    if 100.0 <= tilt_deg <= 180.0:
        return tilt_deg - 130.0

    if 45.0 <= tilt_deg <= 100.0:
        return TILT_ZERO_HORIZON_DEG - tilt_deg

    if -45.0 <= tilt_deg <= 45.0:
        return tilt_deg

    if -135.0 <= tilt_deg <= -45.0:
        return TILT_ZERO_HORIZON_DEG + tilt_deg

    return 0.0


def sea_distance_from_depression(alpha_rad):
    if alpha_rad <= HORIZON_DIP_RAD:
        return None

    tan_a = math.tan(alpha_rad)
    radius = EFFECTIVE_EARTH_RADIUS_M

    disc = (radius * tan_a) ** 2 - 2.0 * radius * CAMERA_HEIGHT_M

    if disc <= 0.0:
        return MAX_SEA_DISTANCE_M

    distance = radius * tan_a - math.sqrt(disc)

    return min(distance, MAX_SEA_DISTANCE_M)


def pixel_row_to_angle(y_value, fy_value):
    return math.atan((y_value - CY) / fy_value)


def predict_horizon_y_from_tilt(sensor_info, pitch_bias_rad=0.0):
    _, fy_value = focal_from_fov(sensor_info["fov_h"], sensor_info["fov_v"])
    pitch_down = resolve_pitch_down_from_tilt(sensor_info.get("tilt"))
    angle = HORIZON_DIP_RAD - math.radians(pitch_down) + pitch_bias_rad
    angle = max(-1.2, min(1.2, angle))

    return CY + fy_value * math.tan(angle)


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


def create_horizon_state():
    return {
        "y": None,
        "slope": 0.0,
        "mode": "TILT_ONLY",
        "visual_miss": 0,
        "pitch_bias_rad": math.radians(MANUAL_HORIZON_BIAS_DEG),
        "visual_history": deque(maxlen=HORIZON_MEDIAN_WINDOW),
        "slope_history": deque(maxlen=HORIZON_MEDIAN_WINDOW),
        "flow_y": 0.0,
    }


def clamp_horizon_y(y_value):
    return max(PROCESS_HEIGHT * 0.02, min(PROCESS_HEIGHT * 0.90, y_value))


def limit_horizon_step(current_y, target_y, max_step):
    diff = target_y - current_y

    if diff > max_step:
        return current_y + max_step

    if diff < -max_step:
        return current_y - max_step

    return target_y


def update_horizon(
    horizon_state, gray, sensor_info, frame_index, camera_moving
):
    target_y = clamp_horizon_y(
        predict_horizon_y_from_tilt(
            sensor_info, horizon_state["pitch_bias_rad"]
        )
    )

    if horizon_state["y"] is None:
        horizon_state["y"] = target_y
        horizon_state["slope"] = 0.0
        horizon_state["mode"] = "TILT_ONLY"
        return horizon_state

    if camera_moving:
        alpha = HORIZON_TILT_ONLY_EMA_MOVING
        max_step = HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX
    else:
        alpha = HORIZON_TILT_ONLY_EMA_STABLE
        max_step = HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX

    blended_y = (1.0 - alpha) * horizon_state["y"] + alpha * target_y
    horizon_state["y"] = clamp_horizon_y(
        limit_horizon_step(horizon_state["y"], blended_y, max_step)
    )
    horizon_state["slope"] = 0.0
    horizon_state["mode"] = "TILT_ONLY"
    horizon_state["visual_miss"] = 0
    horizon_state["flow_y"] = 0.0

    return horizon_state


def horizon_y_at(horizon_state, x_value):
    return horizon_state["y"] + horizon_state["slope"] * (x_value - CX)


def sea_distance_from_image_point(
    pixel_x, pixel_y, sensor_info, horizon_state
):
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    y_horizon = horizon_y_at(horizon_state, pixel_x)
    beta = math.atan((pixel_y - y_horizon) / fy_value)

    base = {
        "raw_distance": None,
        "forward": None,
        "lateral": None,
        "beta_deg": math.degrees(beta),
        "fx": fx_value,
        "fy": fy_value,
        "horizon_y": y_horizon,
    }

    if beta <= MIN_BETA_RAD:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

    alpha = HORIZON_DIP_RAD + beta
    distance = sea_distance_from_depression(alpha)

    if distance is None:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

    phi = math.atan((pixel_x - CX) / fx_value)
    forward_m = distance * math.cos(phi)
    lateral_m = distance * math.sin(phi)

    if distance < MIN_VALID_DISTANCE_M:
        return {
            **base,
            "valid": False,
            "reason": "distance_out_of_range",
            "distance": None,
            "raw_distance": distance,
            "forward": forward_m,
            "lateral": lateral_m,
        }

    return {
        **base,
        "valid": True,
        "reason": "OK",
        "distance": distance,
        "raw_distance": distance,
        "forward": forward_m,
        "lateral": lateral_m,
    }


def get_class_name(model, cls_id):
    names = model.names

    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))

    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]

    return str(cls_id)


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter_area / max(area_a + area_b - inter_area, 1)


def overlap_ratio_small_inside_large(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter_area / min(area_a, area_b)


def horizontal_overlap_ratio(box_a, box_b):
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))

    width_a = max(1, ax2 - ax1)
    width_b = max(1, bx2 - bx1)

    return inter_w / min(width_a, width_b)


def vertical_gap_px(box_a, box_b):
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b

    if ay2 < by1:
        return by1 - ay2

    if by2 < ay1:
        return ay1 - by2

    return 0


def center_x_distance_ratio(box_a, box_b):
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    center_a = (ax1 + ax2) / 2.0
    center_b = (bx1 + bx2) / 2.0

    width_a = max(1, ax2 - ax1)
    width_b = max(1, bx2 - bx1)

    return abs(center_a - center_b) / max(width_a, width_b)


def box_to_int(box):
    x1, y1, x2, y2 = box

    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def visible_box(box):
    x1, y1, x2, y2 = box_to_int(box)

    return (
        max(0, x1),
        max(0, y1),
        min(PROCESS_WIDTH - 1, x2),
        min(PROCESS_HEIGHT - 1, y2),
    )


def clamp_track_box(box):
    x1, y1, x2, y2 = box

    x1 = max(-2.0 * PROCESS_WIDTH, min(3.0 * PROCESS_WIDTH, x1))
    y1 = max(-2.0 * PROCESS_HEIGHT, min(3.0 * PROCESS_HEIGHT, y1))
    x2 = max(x1 + 2.0, min(3.0 * PROCESS_WIDTH, x2))
    y2 = max(y1 + 2.0, min(3.0 * PROCESS_HEIGHT, y2))

    return x1, y1, x2, y2


def get_waterline_ratio(sensor_info):
    if sensor_info["fov_h"] < 15.0:
        return WATERLINE_RATIO_ZOOM

    return WATERLINE_RATIO_NORMAL


def get_water_point_from_box(box, sensor_info):
    x1, y1, x2, y2 = box
    height = max(1, y2 - y1)

    water_x = (x1 + x2) / 2.0
    water_y = y1 + get_waterline_ratio(sensor_info) * height

    return water_x, water_y


def is_own_ship_box(box):
    x1, y1, x2, y2 = box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height
    frame_area = PROCESS_WIDTH * PROCESS_HEIGHT

    if (
        y2 >= PROCESS_HEIGHT * OWN_SHIP_BOTTOM_RATIO
        and height >= PROCESS_HEIGHT * OWN_SHIP_MIN_HEIGHT_RATIO
    ):
        return True

    if area >= frame_area * OWN_SHIP_MAX_AREA_RATIO:
        return True

    if (
        y2 >= PROCESS_HEIGHT * 0.97
        and height >= PROCESS_HEIGHT * 0.18
        and width >= PROCESS_WIDTH * 0.35
    ):
        return True

    return False


def filter_detection(det, sensor_info, horizon_state):
    x1, y1, x2, y2 = det["box"]

    width = x2 - x1
    height = y2 - y1
    area = width * height
    frame_area = PROCESS_WIDTH * PROCESS_HEIGHT
    is_thermal = det.get("channel") == "thermal"

    if is_thermal:
        min_width = 14
        min_height = 8
    else:
        min_width = 6
        min_height = 5

    if width < min_width or height < min_height:
        return False

    if is_thermal:
        if sensor_info["fov_h"] < 15.0:
            min_area = 70
        elif sensor_info["fov_h"] < 30.0:
            min_area = 110
        else:
            min_area = 180
    elif sensor_info["fov_h"] < 15.0:
        min_area = 40
    elif sensor_info["fov_h"] < 30.0:
        min_area = 90
    else:
        min_area = 220

    if area < min_area:
        return False

    if area > frame_area * 0.45:
        return False

    aspect = width / max(height, 1)

    if is_thermal:
        if aspect < 0.45 or aspect > 16.0:
            return False
        if y1 > PROCESS_HEIGHT * 0.82:
            return False
        if det["water_y"] > PROCESS_HEIGHT * 0.94:
            return False
        if (
            det.get("source", "").startswith("thermal_blob")
            and area < THERMAL_BLOB_MIN_AREA
        ):
            return False
    elif aspect < 0.25 or aspect > 18.0:
        return False

    if is_own_ship_box(det["box"]):
        return False

    y_horizon = horizon_y_at(horizon_state, det["water_x"])

    if det["water_y"] <= y_horizon + 0.5:
        return False

    result = sea_distance_from_image_point(
        det["water_x"], det["water_y"], sensor_info, horizon_state
    )

    if (
        result["valid"]
        and result["distance"] < OWN_SHIP_NEAR_DISTANCE_M
        and y2 > PROCESS_HEIGHT * OWN_SHIP_NEAR_BOTTOM_RATIO
    ):
        return False

    return True


def build_search_regions(sensor_info, horizon_state, mode):
    y_h = int(max(10, min(PROCESS_HEIGHT - 40, horizon_state["y"])))

    regions = [("full", 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT)]

    if mode == "full_only":
        return regions

    regions.append(
        (
            "horizon_strip",
            0,
            max(0, y_h - 46),
            PROCESS_WIDTH,
            min(PROCESS_HEIGHT, y_h + 110),
        )
    )

    if mode == "bottom_deep":
        regions.extend(
            [
                (
                    "bottom_75",
                    0,
                    int(PROCESS_HEIGHT * 0.25),
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT,
                ),
                (
                    "bottom_55",
                    0,
                    int(PROCESS_HEIGHT * 0.45),
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT,
                ),
            ]
        )

        tile_w = 640
        step = 420
        y0 = max(0, min(int(PROCESS_HEIGHT * 0.35), y_h - 20))

    else:
        regions.append(
            (
                "below_horizon",
                0,
                max(0, y_h - 20),
                PROCESS_WIDTH,
                PROCESS_HEIGHT,
            )
        )

        if sensor_info["fov_h"] < 20.0:
            tile_w = 520
            step = 360
        else:
            tile_w = 640
            step = 460

        y0 = max(0, y_h - 20)

    x_value = 0

    while x_value < PROCESS_WIDTH:
        x2 = min(PROCESS_WIDTH, x_value + tile_w)
        x1 = max(0, x2 - tile_w)

        regions.append((f"tile_{x1}", x1, y0, x2, PROCESS_HEIGHT))

        if x2 >= PROCESS_WIDTH:
            break

        x_value += step

    return regions


def prepare_frame_for_detection(frame, channel):
    if channel != "thermal":
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray.astype(np.uint8))
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def create_thermal_candidate_mask(crop_gray):
    if crop_gray.size == 0:
        return None

    gray = cv2.normalize(crop_gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = gray.astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    mean_value = float(np.mean(eq))
    std_value = float(np.std(eq))

    high_percentile = float(np.percentile(eq, THERMAL_BLOB_BRIGHT_PERCENTILE))
    low_percentile = float(np.percentile(eq, THERMAL_BLOB_DARK_PERCENTILE))

    high_threshold = max(high_percentile, mean_value + 0.45 * std_value)
    low_threshold = min(low_percentile, mean_value - 0.45 * std_value)

    bright = (eq >= high_threshold).astype(np.uint8) * 255
    dark = (eq <= low_threshold).astype(np.uint8) * 255

    if std_value < THERMAL_BLOB_MIN_CONTRAST:
        return None

    mask = bright

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def detect_thermal_blobs(frame, sensor_info, horizon_state, mode):
    if not THERMAL_BLOB_DETECTION_ENABLED:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = build_search_regions(sensor_info, horizon_state, mode)
    detections = []

    for region in regions:
        region_name, x1, y1, x2, y2 = region

        if region_name == "full":
            continue

        crop_gray = gray[y1:y2, x1:x2]

        if crop_gray.size == 0 or (y2 - y1) < 20:
            continue

        mask = create_thermal_candidate_mask(crop_gray)

        if mask is None:
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)

            if bw <= 0 or bh <= 0:
                continue

            area = bw * bh
            area_ratio = area / float(PROCESS_WIDTH * PROCESS_HEIGHT)
            aspect = bw / max(bh, 1)

            if area < THERMAL_BLOB_MIN_AREA:
                continue

            if area_ratio > THERMAL_BLOB_MAX_AREA_RATIO:
                continue

            if (
                aspect < THERMAL_BLOB_MIN_ASPECT
                or aspect > THERMAL_BLOB_MAX_ASPECT
            ):
                continue

            pad_x = max(4, int(bw * 0.10))
            pad_y = max(3, int(bh * 0.12))

            abs_box = (
                int(max(0, x1 + bx - pad_x)),
                int(max(0, y1 + by - pad_y)),
                int(min(PROCESS_WIDTH - 1, x1 + bx + bw + pad_x)),
                int(min(PROCESS_HEIGHT - 1, y1 + by + bh + pad_y)),
            )

            water_x, water_y = get_water_point_from_box(abs_box, sensor_info)

            patch = crop_gray[by : by + bh, bx : bx + bw]
            surrounding = crop_gray[
                max(0, by - bh) : min(crop_gray.shape[0], by + 2 * bh),
                max(0, bx - bw) : min(crop_gray.shape[1], bx + 2 * bw),
            ]

            if patch.size == 0 or surrounding.size == 0:
                contrast_score = 0.35
            else:
                contrast = abs(
                    float(np.mean(patch)) - float(np.mean(surrounding))
                )
                contrast_score = max(0.25, min(0.75, contrast / 80.0))

            det = {
                "box": abs_box,
                "conf": contrast_score,
                "water_x": water_x,
                "water_y": water_y,
                "source": f"thermal_blob_{region_name}",
                "channel": "thermal",
            }

            if filter_detection(det, sensor_info, horizon_state):
                detections.append(det)

    return detections


def run_yolo_region(
    frame,
    model,
    region,
    sensor_info,
    horizon_state,
    conf_thres,
    imgsz,
    channel="rgb",
):
    region_name, x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0 or (y2 - y1) < 24:
        return []

    detections = []

    results = model.predict(
        crop,
        conf=conf_thres,
        imgsz=imgsz,
        iou=YOLO_IOU_THRES,
        verbose=False,
        classes=[8],
        max_det=25 if channel == "thermal" else 50,
        device=YOLO_DEVICE,
        half=YOLO_HALF,
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            name = get_class_name(model, cls_id)

            if name != "boat":
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()

            abs_box = (
                int(max(0, bx1 + x1)),
                int(max(0, by1 + y1)),
                int(min(PROCESS_WIDTH - 1, bx2 + x1)),
                int(min(PROCESS_HEIGHT - 1, by2 + y1)),
            )

            water_x, water_y = get_water_point_from_box(abs_box, sensor_info)

            det = {
                "box": abs_box,
                "conf": conf,
                "water_x": water_x,
                "water_y": water_y,
                "source": region_name,
                "channel": channel,
            }

            if filter_detection(det, sensor_info, horizon_state):
                detections.append(det)

    return detections


def detection_quality_score(det):
    x1, y1, x2, y2 = det["box"]

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height

    area_score = min(area / (PROCESS_WIDTH * PROCESS_HEIGHT), 0.35)
    water_score = det["water_y"] / PROCESS_HEIGHT

    return 2.2 * det["conf"] + 1.2 * area_score + 0.8 * water_score


def same_vessel(det_a, det_b):
    box_a = det_a["box"]
    box_b = det_b["box"]

    if calculate_iou(box_a, box_b) > MERGE_IOU_THRES:
        return True

    if overlap_ratio_small_inside_large(box_a, box_b) > MERGE_INSIDE_THRES:
        return True

    if (
        horizontal_overlap_ratio(box_a, box_b)
        >= MERGE_HORIZONTAL_OVERLAP_THRES
        and vertical_gap_px(box_a, box_b) <= MERGE_VERTICAL_GAP_PX
        and center_x_distance_ratio(box_a, box_b)
        <= MERGE_CENTER_DISTANCE_RATIO
    ):
        return True

    return False


def merge_detection_group(group, sensor_info):
    main_det = max(group, key=detection_quality_score)

    water_x, water_y = get_water_point_from_box(main_det["box"], sensor_info)

    return {
        "box": main_det["box"],
        "conf": max(det["conf"] for det in group),
        "water_x": water_x,
        "water_y": water_y,
        "source": main_det["source"],
    }


def merge_same_vessel_detections(detections, sensor_info):
    detections = sorted(detections, key=detection_quality_score, reverse=True)

    groups = []

    for det in detections:
        placed = False

        for group in groups:
            if any(same_vessel(det, other) for other in group):
                group.append(det)
                placed = True
                break

        if not placed:
            groups.append([det])

    merged = [merge_detection_group(group, sensor_info) for group in groups]

    merged = sorted(merged, key=detection_quality_score, reverse=True)

    kept = []

    for det in merged:
        if not any(same_vessel(det, kept_det) for kept_det in kept):
            kept.append(det)

    return kept


def detect_boats(
    frame, model, sensor_info, horizon_state, mode, channel="rgb"
):
    is_thermal = channel == "thermal"

    if is_thermal:
        if sensor_info["fov_h"] < 20.0:
            imgsz = THERMAL_YOLO_IMGSZ_DEEP
        else:
            imgsz = THERMAL_YOLO_IMGSZ_FULL

        conf_full = THERMAL_YOLO_CONF_FULL
        conf_deep = THERMAL_YOLO_CONF_DEEP
        detection_frame = prepare_frame_for_detection(frame, "thermal")
    else:
        if sensor_info["fov_h"] < 20.0:
            imgsz = YOLO_IMGSZ_DEEP
        else:
            imgsz = YOLO_IMGSZ_FULL

        conf_full = YOLO_CONF_FULL
        conf_deep = YOLO_CONF_DEEP
        detection_frame = frame

    regions = build_search_regions(sensor_info, horizon_state, mode)
    detections = []

    for region in regions:
        detections.extend(
            run_yolo_region(
                detection_frame,
                model,
                region,
                sensor_info,
                horizon_state,
                conf_full,
                imgsz,
                channel=channel,
            )
        )

    if is_thermal and not detections:
        detections.extend(
            detect_thermal_blobs(frame, sensor_info, horizon_state, mode)
        )

    if not detections and mode in ("deep", "bottom_deep"):
        for region in regions:
            detections.extend(
                run_yolo_region(
                    detection_frame,
                    model,
                    region,
                    sensor_info,
                    horizon_state,
                    conf_deep,
                    imgsz,
                    channel=channel,
                )
            )

        if is_thermal and not detections:
            detections.extend(
                detect_thermal_blobs(frame, sensor_info, horizon_state, "deep")
            )

    merged = merge_same_vessel_detections(detections, sensor_info)

    if is_thermal:
        merged = sorted(merged, key=detection_quality_score, reverse=True)[:10]

    return merged


def estimate_global_motion(previous_gray, current_gray, tracks):
    if previous_gray is None:
        return 0.0, 0.0, False

    mask = np.full((PROCESS_HEIGHT, PROCESS_WIDTH), 255, dtype=np.uint8)

    for track in tracks.values():
        x1, y1, x2, y2 = visible_box(track["box"])

        if x2 > x1 and y2 > y1:
            pad = 14
            mask[
                max(0, y1 - pad) : min(PROCESS_HEIGHT, y2 + pad),
                max(0, x1 - pad) : min(PROCESS_WIDTH, x2 + pad),
            ] = 0

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=GLOBAL_MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=16,
        mask=mask,
        blockSize=7,
    )

    if points is None or len(points) < GLOBAL_MIN_POINTS:
        return 0.0, 0.0, False

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if next_points is None or status is None:
        return 0.0, 0.0, False

    ok = status.flatten() == 1

    good_prev = points[ok].reshape(-1, 2)
    good_next = next_points[ok].reshape(-1, 2)

    if len(good_prev) < GLOBAL_MIN_POINTS:
        return 0.0, 0.0, False

    flow = good_next - good_prev

    dx_med = float(np.median(flow[:, 0]))
    dy_med = float(np.median(flow[:, 1]))

    residual = np.sqrt((flow[:, 0] - dx_med) ** 2 + (flow[:, 1] - dy_med) ** 2)
    keep = residual < max(3.0, float(np.percentile(residual, 75)) + 3.0)

    if int(np.sum(keep)) < GLOBAL_MIN_POINTS:
        return 0.0, 0.0, False

    dx_value = float(np.median(flow[keep, 0]))
    dy_value = float(np.median(flow[keep, 1]))

    dx_value = max(-GLOBAL_MAX_FLOW_PX, min(GLOBAL_MAX_FLOW_PX, dx_value))
    dy_value = max(-GLOBAL_MAX_FLOW_PX, min(GLOBAL_MAX_FLOW_PX, dy_value))

    return dx_value, dy_value, True


def rescale_point(x_value, y_value, scale_x, scale_y):
    return (CX + (x_value - CX) * scale_x, CY + (y_value - CY) * scale_y)


def apply_fov_rescale(tracks, horizon_state, scale_x, scale_y):
    if horizon_state["y"] is not None:
        horizon_state["y"] = clamp_horizon_y(
            CY + (horizon_state["y"] - CY) * scale_y
        )

    big_change = max(abs(scale_x - 1.0), abs(scale_y - 1.0)) > 0.02

    for track in tracks.values():
        x1, y1, x2, y2 = track["box"]

        nx1, ny1 = rescale_point(x1, y1, scale_x, scale_y)
        nx2, ny2 = rescale_point(x2, y2, scale_x, scale_y)

        track["box"] = clamp_track_box((nx1, ny1, nx2, ny2))

        if track.get("prev_measured_box") is not None:
            px1, py1, px2, py2 = track["prev_measured_box"]

            qx1, qy1 = rescale_point(px1, py1, scale_x, scale_y)
            qx2, qy2 = rescale_point(px2, py2, scale_x, scale_y)

            track["prev_measured_box"] = (qx1, qy1, qx2, qy2)

        vx1, vy1, vx2, vy2 = track.get("velocity_box", (0.0, 0.0, 0.0, 0.0))
        track["velocity_box"] = (
            vx1 * scale_x,
            vy1 * scale_y,
            vx2 * scale_x,
            vy2 * scale_y,
        )

        old_hist = list(track["water_hist"])
        track["water_hist"].clear()

        for value in old_hist:
            track["water_hist"].append(CY + (value - CY) * scale_y)

        points = track.get("klt_points")

        if points is not None:
            if big_change:
                track["klt_points"] = None
            else:
                points[:, 0, 0] = CX + (points[:, 0, 0] - CX) * scale_x
                points[:, 0, 1] = CY + (points[:, 0, 1] - CY) * scale_y


def init_klt_points(gray, box):
    x1, y1, x2, y2 = visible_box(box)

    width = x2 - x1
    height = y2 - y1

    if width < 15 or height < 15:
        return None

    rx1 = x1 + int(width * KLT_SIDE_MARGIN_RATIO)
    rx2 = x2 - int(width * KLT_SIDE_MARGIN_RATIO)
    ry1 = y1 + int(height * KLT_TOP_RATIO)
    ry2 = y1 + int(height * KLT_BOTTOM_RATIO)

    rx1 = max(0, min(PROCESS_WIDTH - 2, rx1))
    rx2 = max(rx1 + 2, min(PROCESS_WIDTH - 1, rx2))
    ry1 = max(0, min(PROCESS_HEIGHT - 2, ry1))
    ry2 = max(ry1 + 2, min(PROCESS_HEIGHT - 1, ry2))

    roi = gray[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=KLT_MAX_CORNERS,
        qualityLevel=KLT_QUALITY_LEVEL,
        minDistance=KLT_MIN_DISTANCE,
        blockSize=KLT_BLOCK_SIZE,
    )

    if points is None or len(points) < KLT_MIN_POINTS:
        return None

    points[:, 0, 0] += rx1
    points[:, 0, 1] += ry1

    return points.astype(np.float32)


def apply_klt_to_track(track, previous_gray, current_gray):
    if previous_gray is None or current_gray is None:
        return False

    points_prev = track.get("klt_points")

    if points_prev is None or len(points_prev) < KLT_MIN_POINTS:
        points_prev = init_klt_points(previous_gray, track["box"])

        if points_prev is None:
            return False

    points_next, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points_prev,
        None,
        winSize=(23, 23),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    if points_next is None or status is None:
        track["klt_points"] = None
        return False

    ok = status.flatten() == 1

    good_prev = points_prev[ok]
    good_next = points_next[ok]

    if len(good_prev) < KLT_MIN_POINTS:
        track["klt_points"] = None
        return False

    flow = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)

    dx_med = float(np.median(flow[:, 0]))
    dy_med = float(np.median(flow[:, 1]))

    residual = np.sqrt((flow[:, 0] - dx_med) ** 2 + (flow[:, 1] - dy_med) ** 2)
    keep = residual < max(8.0, float(np.percentile(residual, 75)) + 8.0)

    if int(np.sum(keep)) < KLT_MIN_POINTS:
        track["klt_points"] = None
        return False

    dx_value = float(np.median(flow[keep, 0]))
    dy_value = float(np.median(flow[keep, 1]))

    dx_value = max(-KLT_MAX_STEP_PX, min(KLT_MAX_STEP_PX, dx_value))
    dy_value = max(-KLT_MAX_STEP_PX, min(KLT_MAX_STEP_PX, dy_value))

    x1, y1, x2, y2 = track["box"]

    track["box"] = clamp_track_box(
        (x1 + dx_value, y1 + dy_value, x2 + dx_value, y2 + dy_value)
    )

    track["klt_points"] = good_next[keep].reshape(-1, 1, 2).astype(np.float32)

    return True


def update_track_velocity(track, measured_box):
    previous = track.get("prev_measured_box")
    current = tuple(float(value) for value in measured_box)

    if previous is None:
        track["velocity_box"] = (0.0, 0.0, 0.0, 0.0)
        track["prev_measured_box"] = current
        return

    measured_velocity = tuple(
        cur - prev for cur, prev in zip(current, previous)
    )
    old_velocity = track.get("velocity_box", (0.0, 0.0, 0.0, 0.0))

    track["velocity_box"] = tuple(
        0.80 * old + 0.20 * new
        for old, new in zip(old_velocity, measured_velocity)
    )

    track["prev_measured_box"] = current


def shift_track_box(track, dx_value, dy_value):
    x1, y1, x2, y2 = track["box"]

    track["box"] = clamp_track_box(
        (x1 + dx_value, y1 + dy_value, x2 + dx_value, y2 + dy_value)
    )


def apply_velocity_prediction(track):
    velocity = track.get("velocity_box", (0.0, 0.0, 0.0, 0.0))
    box = track["box"]

    predicted = tuple(b + v for b, v in zip(box, velocity))

    track["box"] = clamp_track_box(predicted)


def smooth_value(old_value, new_value, alpha):
    if old_value is None:
        return new_value

    return (1.0 - alpha) * old_value + alpha * new_value


def smooth_box(old_box, new_box, alpha):
    if old_box is None:
        return tuple(float(value) for value in new_box)

    return tuple(
        (1.0 - alpha) * old + alpha * new for old, new in zip(old_box, new_box)
    )


def refresh_track_water_point(track, sensor_info):
    vx1, vy1, vx2, vy2 = visible_box(track["box"])

    if vx2 <= vx1 or vy2 <= vy1:
        return

    water_x, water_y = get_water_point_from_box(
        (vx1, vy1, vx2, vy2), sensor_info
    )

    track["water_hist"].append(water_y)
    track["water_x"] = water_x
    track["water_y"] = median(track["water_hist"])


def create_new_track(track_id, det, current_gray, frame_index):
    return {
        "id": track_id,
        "box": tuple(float(value) for value in det["box"]),
        "water_x": det["water_x"],
        "water_y": det["water_y"],
        "water_hist": deque([det["water_y"]], maxlen=WATER_HISTORY_LEN),
        "conf": det["conf"],
        "source": det["source"],
        "channel": det.get("channel", "rgb"),
        "missed": 0,
        "frames_since_update": 0,
        "age": 1,
        "confirmed_updates": 1,
        "last_result": None,
        "velocity_box": (0.0, 0.0, 0.0, 0.0),
        "prev_measured_box": tuple(float(v) for v in det["box"]),
        "klt_points": init_klt_points(current_gray, det["box"]),
        "klt_ok": False,
        "global_frame_index": frame_index,
        "range_locked_m": None,
        "range_last_frame": None,
        "range_init_samples": deque(maxlen=RANGE_INIT_SAMPLE_COUNT),
        "range_history": deque(maxlen=RANGE_HISTORY_WINDOW),
        "range_reject_count": 0,
        "recent_raws": deque(maxlen=RECENT_RAW_WINDOW),
    }


def track_match_score(det, track):
    det_box = det["box"]
    track_box = box_to_int(track["box"])

    iou = calculate_iou(det_box, track_box)
    h_overlap = horizontal_overlap_ratio(det_box, track_box)
    center_ratio = center_x_distance_ratio(det_box, track_box)

    center_score = max(0.0, 1.0 - center_ratio)

    water_y_diff = abs(det["water_y"] - track["water_y"])
    water_score = max(0.0, 1.0 - water_y_diff / 180.0)

    return (
        0.40 * iou
        + 0.25 * h_overlap
        + 0.22 * center_score
        + 0.13 * water_score
    )


def update_tracks(
    detections,
    tracks,
    next_track_id,
    previous_gray,
    current_gray,
    sensor_info,
    detection_was_run,
    frame_index,
    skip_optical,
    global_flow,
    global_flow_ok,
):
    gdx, gdy = global_flow

    for track in tracks.values():
        track["global_frame_index"] = frame_index
        track["frames_since_update"] = track.get("frames_since_update", 0) + 1
        track["klt_ok"] = False

        if not skip_optical:
            klt_ok = apply_klt_to_track(track, previous_gray, current_gray)

            if klt_ok:
                track["klt_ok"] = True
            elif global_flow_ok:
                shift_track_box(track, gdx, gdy)
            else:
                apply_velocity_prediction(track)

            if track.get("frames_since_update", 0) % KLT_REINIT_EVERY == 0:
                track["klt_points"] = init_klt_points(
                    current_gray, track["box"]
                )

        refresh_track_water_point(track, sensor_info)

    matched_track_ids = set()
    matched_detection_indices = set()

    candidates = []

    for det_idx, det in enumerate(detections):
        for track_id, track in tracks.items():
            score = track_match_score(det, track)

            if score >= TRACK_MATCH_SCORE_THRES:
                candidates.append((score, det_idx, track_id))

    candidates.sort(reverse=True, key=lambda item: item[0])

    for _, det_idx, track_id in candidates:
        if det_idx in matched_detection_indices:
            continue

        if track_id in matched_track_ids:
            continue

        det = detections[det_idx]
        track = tracks[track_id]

        update_track_velocity(track, det["box"])

        track["box"] = smooth_box(track["box"], det["box"], BOX_ALPHA)
        track["conf"] = smooth_value(track["conf"], det["conf"], CONF_ALPHA)
        track["source"] = det["source"]
        track["channel"] = det.get("channel", track.get("channel", "rgb"))
        track["missed"] = 0
        track["frames_since_update"] = 0
        track["age"] += 1
        track["confirmed_updates"] += 1
        track["klt_points"] = init_klt_points(current_gray, det["box"])

        track["water_hist"].append(det["water_y"])
        track["water_x"] = det["water_x"]
        track["water_y"] = median(track["water_hist"])

        matched_track_ids.add(track_id)
        matched_detection_indices.add(det_idx)

    for det_idx, det in enumerate(detections):
        if det_idx in matched_detection_indices:
            continue

        tracks[next_track_id] = create_new_track(
            next_track_id, det, current_gray, frame_index
        )

        next_track_id += 1

    for track_id in list(tracks.keys()):
        track = tracks[track_id]

        if detection_was_run and track_id not in matched_track_ids:
            track["missed"] += 1

        vx1, vy1, vx2, vy2 = visible_box(track["box"])
        visible_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)

        fully_outside = visible_area <= 0

        if (
            fully_outside
            or track["missed"] > TRACK_MAX_MISSED_DETECTIONS
            or track["frames_since_update"] > TRACK_MAX_STALE_FRAMES
        ):
            del tracks[track_id]

    return tracks, next_track_id


def clamp_range_change(previous_distance, candidate_distance, max_delta):
    if previous_distance is None:
        return candidate_distance

    lower = previous_distance - max_delta
    upper = previous_distance + max_delta

    return max(lower, min(upper, candidate_distance))


def calculate_track_distance(track, sensor_info, horizon_state, video_fps):
    raw_result = sea_distance_from_image_point(
        track["water_x"], track["water_y"], sensor_info, horizon_state
    )

    if not raw_result["valid"]:
        last = track.get("last_result")

        if raw_result["reason"] == "at_or_beyond_horizon" and (
            last is None or not last.get("valid")
        ):
            track["last_result"] = raw_result
            return raw_result

        if last is not None and last.get("valid"):
            return last

        track["last_result"] = raw_result
        return raw_result

    raw_distance = raw_result["distance"]
    track["recent_raws"].append(raw_distance)

    previous_locked = track["range_locked_m"]

    if previous_locked is None:
        track["range_init_samples"].append(raw_distance)

        locked_distance = median(track["range_init_samples"])

        if len(track["range_init_samples"]) >= RANGE_INIT_SAMPLE_COUNT:
            track["range_locked_m"] = locked_distance
            track["range_last_frame"] = track.get("global_frame_index", 0)

        result = raw_result.copy()
        result["raw_distance"] = raw_distance
        result["distance"] = locked_distance

        track["last_result"] = result
        return result

    current_frame = track.get("global_frame_index", 0)
    previous_frame = track.get("range_last_frame")

    if previous_frame is None:
        frame_delta = 1
    else:
        frame_delta = max(1, current_frame - previous_frame)

    track["range_last_frame"] = current_frame

    dt_value = frame_delta / max(video_fps, 1.0)

    max_rate = max(
        RANGE_MIN_RATE_M_PER_SEC, previous_locked * RANGE_RELATIVE_RATE_PER_SEC
    )
    max_delta = max_rate * dt_value

    ratio = max(raw_distance, previous_locked) / max(
        min(raw_distance, previous_locked), 1e-6
    )

    if ratio > MAX_ACCEPTED_RAW_JUMP_RATIO:
        track["range_reject_count"] += 1

        if (
            track["range_reject_count"] >= RANGE_REJECTS_TO_RELOCK
            and len(track["recent_raws"]) >= 5
        ):
            previous_locked = median(track["recent_raws"])
            track["range_reject_count"] = 0
            candidate_distance = previous_locked
            max_delta = previous_locked
        else:
            candidate_distance = previous_locked
    else:
        track["range_reject_count"] = 0

        if track.get("frames_since_update", 0) <= 2:
            alpha = RANGE_UPDATE_ALPHA_DETECTED
        else:
            alpha = RANGE_UPDATE_ALPHA_KLT

        candidate_distance = (
            1.0 - alpha
        ) * previous_locked + alpha * raw_distance

    locked_distance = clamp_range_change(
        previous_locked, candidate_distance, max_delta
    )

    locked_distance = max(
        MIN_VALID_DISTANCE_M, min(MAX_SEA_DISTANCE_M, locked_distance)
    )

    track["range_locked_m"] = locked_distance
    track["range_history"].append(locked_distance)

    if len(track["range_history"]) >= 5:
        stable_distance = 0.85 * locked_distance + 0.15 * median(
            track["range_history"]
        )
    else:
        stable_distance = locked_distance

    result = raw_result.copy()
    result["raw_distance"] = raw_distance
    result["distance"] = stable_distance

    track["last_result"] = result

    return result


def any_track_near_bottom(tracks):
    for track in tracks.values():
        if track.get("frames_since_update", 0) > TRACK_DRAW_MAX_STALE_FRAMES:
            continue

        _, _, _, y2 = box_to_int(track["box"])

        if y2 > PROCESS_HEIGHT * 0.72:
            return True

    return False


def active_track_count(tracks):
    count = 0

    for track in tracks.values():
        if (
            track.get("frames_since_update", 0) <= TRACK_DRAW_MAX_STALE_FRAMES
            and track.get("confirmed_updates", 0)
            >= TRACK_MIN_CONFIRMED_UPDATES
        ):
            count += 1

    return count


def should_run_detection(frame_index, tracks, camera_moving, force_detection):
    if camera_moving:
        return False, "camera_moving"

    if force_detection:
        return True, "deep"

    if active_track_count(tracks) > 0:
        if (
            any_track_near_bottom(tracks)
            and frame_index % DETECT_INTERVAL_BOTTOM_DEEP == 0
        ):
            return True, "bottom_deep"

        if frame_index % DETECT_INTERVAL_TRACKING == 0:
            return True, "full_only"

        return False, "klt_track"

    if frame_index % DETECT_INTERVAL_LOST_DEEP == 0:
        return True, "deep"

    if frame_index % DETECT_INTERVAL_LOST_FULL == 0:
        return True, "full_only"

    return False, "lost_wait"


def format_distance(distance_m):
    if distance_m is None:
        return "?"

    if distance_m >= 1000.0:
        return f"{distance_m / 1000.0:.2f} km"

    return f"{distance_m:.1f} m"


def measure_text(text, scale, thickness=1):
    size, base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    return size[0], size[1] + base


def rects_overlap(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def place_label(occupied, x_value, y_pref, width, height):
    step = height + 8

    candidates = [y_pref - i * step for i in range(6)]
    candidates += [y_pref + i * step for i in range(1, 6)]

    for y_value in candidates:
        if y_value - height < PANEL_HEIGHT + 6:
            continue

        if y_value > PROCESS_HEIGHT - 4:
            continue

        rect = (
            x_value - 4,
            y_value - height - 4,
            x_value + width + 4,
            y_value + 4,
        )

        if not any(rects_overlap(rect, other) for other in occupied):
            occupied.append(rect)
            return y_value

    fallback = min(PROCESS_HEIGHT - 6, max(PANEL_HEIGHT + height, y_pref))
    occupied.append(
        (x_value - 4, fallback - height - 4, x_value + width + 4, fallback + 4)
    )

    return fallback


def draw_text_bg(
    frame,
    text,
    org,
    scale=0.48,
    color=(0, 255, 255),
    bg=(0, 0, 0),
    thickness=1,
):
    x_value, y_value = org

    size, base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    width, height = size

    cv2.rectangle(
        frame,
        (x_value - 4, y_value - height - 6),
        (x_value + width + 4, y_value + base + 4),
        bg,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x_value, y_value),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_horizon_line(frame, horizon_state):
    if not DRAW_HORIZON_LINE:
        return

    y_left = int(round(horizon_y_at(horizon_state, 0)))
    y_right = int(round(horizon_y_at(horizon_state, PROCESS_WIDTH - 1)))

    cv2.line(
        frame,
        (0, y_left),
        (PROCESS_WIDTH - 1, y_right),
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def draw_tracks(frame, tracks, sensor_info, horizon_state, video_fps):
    occupied = []

    ordered = sorted(tracks.items(), key=lambda item: item[1]["water_y"])

    for track_id, track in ordered:
        if track["age"] < TRACK_MIN_AGE_TO_DISPLAY:
            continue

        if track["confirmed_updates"] < TRACK_MIN_CONFIRMED_UPDATES:
            continue

        if track.get("frames_since_update", 0) > TRACK_DRAW_MAX_STALE_FRAMES:
            continue

        if track.get("channel") == "thermal":
            if track.get("conf", 0.0) < THERMAL_YOLO_CONF_DEEP:
                continue
            if (
                track.get("confirmed_updates", 0)
                < TRACK_MIN_CONFIRMED_UPDATES + 1
            ):
                continue

        x1, y1, x2, y2 = visible_box(track["box"])

        if x2 <= x1 or y2 <= y1:
            continue

        result = calculate_track_distance(
            track, sensor_info, horizon_state, video_fps
        )

        water_x = int(round(track["water_x"]))
        water_y = int(round(track["water_y"]))

        water_x = max(0, min(PROCESS_WIDTH - 1, water_x))
        water_y = max(0, min(PROCESS_HEIGHT - 1, water_y))

        at_horizon = (
            not result["valid"] and result["reason"] == "at_or_beyond_horizon"
        )

        if result["valid"]:
            color = (0, 255, 0)
            distance_text = format_distance(result["distance"])
        elif at_horizon:
            color = (0, 200, 255)
            distance_text = f">{format_distance(MAX_SEA_DISTANCE_M)}"
        else:
            color = (0, 200, 255)
            distance_text = "?"

        mode_text = "KLT" if track.get("klt_ok") else "PRED"
        label = f"id={track_id} | {distance_text} | {mode_text}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.drawMarker(
            frame,
            (water_x, water_y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

        label_w, label_h = measure_text(label, 0.50)
        label_y = place_label(
            occupied, x1 + 4, max(180, y1 - 8), label_w, label_h
        )

        draw_text_bg(
            frame,
            label,
            (x1 + 4, label_y),
            scale=0.50,
            color=color,
            bg=(0, 0, 0),
            thickness=1,
        )

        if SHOW_TRACK_DETAILS:
            if result["valid"]:
                detail = (
                    f"raw={format_distance(result['raw_distance'])} | "
                    f"beta={result['beta_deg']:.3f}deg | "
                    f"src={track['source']}"
                )
            else:
                detail = (
                    f"{result['reason']} | src={track['source']} | "
                    f"stale={track.get('frames_since_update', 0)}"
                )

            detail_w, detail_h = measure_text(detail, 0.40)
            detail_y = place_label(
                occupied,
                x1 + 4,
                min(PROCESS_HEIGHT - 10, y2 + 22),
                detail_w,
                detail_h,
            )

            draw_text_bg(
                frame,
                detail,
                (x1 + 4, detail_y),
                scale=0.40,
                color=(0, 255, 255),
                bg=(0, 0, 0),
                thickness=1,
            )


def draw_panel(
    frame,
    sensor_info,
    horizon_state,
    fps,
    video_second,
    track_count,
    mode,
    camera_moving,
):
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    zoom_text = (
        "?" if sensor_info["zoom"] is None else f"{sensor_info['zoom']:.4f}"
    )
    tilt_text = (
        "?" if sensor_info["tilt"] is None else f"{sensor_info['tilt']:.3f}"
    )

    moving_text = "MOVING" if camera_moving else "stable"
    bias_deg = math.degrees(horizon_state["pitch_bias_rad"])

    cv2.rectangle(frame, (0, 0), (PROCESS_WIDTH, PANEL_HEIGHT), (0, 0, 0), -1)

    lines = [
        (
            f"HORIZON-LOCKED DISTANCE | FPS={fps:.1f} | "
            f"t={video_second:.2f}s | record={RECORD_NAME}"
        ),
        (
            f"FOV_H={sensor_info['fov_h']:.3f} | "
            f"FOV_V={sensor_info['fov_v']:.3f} | "
            f"FX={fx_value:.0f} | FY={fy_value:.0f} | zoom={zoom_text}"
        ),
        (
            f"horizon={horizon_state['mode']}@{horizon_state['y']:.1f}px "
            f"slope={horizon_state['slope']:.4f} bias={bias_deg:+.3f}deg | "
            f"tilt={tilt_text} | cam_h={CAMERA_HEIGHT_M:.1f}m | "
            f"d_max={MAX_SEA_DISTANCE_M / 1000.0:.1f}km"
        ),
        (
            f"tracks={track_count} | mode={mode} | cam={moving_text} | "
            "tilt/FOV horizon only | spherical earth + refraction"
        ),
    ]

    y_value = 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y_value),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_value += 31


def ensure_bgr_frame(frame):
    if frame is None:
        return None

    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


def create_stream_state(name, channel):
    return {
        "name": name,
        "channel": channel,
        "tracks": {},
        "next_track_id": 1,
        "sensor_smooth": None,
        "previous_gray": None,
        "previous_fx": None,
        "previous_fy": None,
        "horizon_state": create_horizon_state(),
        "was_moving": False,
        "force_detection": False,
        "mode": "init",
    }


def process_stream_frame(
    frame, stream_state, sensor_rows, model, frame_index, video_fps
):
    frame = ensure_bgr_frame(frame)

    if frame is None:
        frame = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8)
        draw_text_bg(
            frame,
            f"{stream_state['name']} frame yok",
            (20, PANEL_HEIGHT + 35),
            scale=0.70,
            color=(0, 200, 255),
            bg=(0, 0, 0),
            thickness=2,
        )
        return frame, False

    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    channel = stream_state.get("channel", "rgb")

    video_second = frame_index / video_fps

    sensor_raw = get_sensor_for_time(sensor_rows, video_second)
    stream_state["sensor_smooth"] = smooth_sensor(
        stream_state["sensor_smooth"], sensor_raw
    )
    sensor_info = stream_state["sensor_smooth"]

    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    if (
        stream_state["previous_fx"] is not None
        and stream_state["previous_fx"] > 0
    ):
        scale_x = fx_value / stream_state["previous_fx"]
        scale_y = fy_value / stream_state["previous_fy"]
    else:
        scale_x = 1.0
        scale_y = 1.0

    scale_change = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
    zooming = scale_change > ZOOM_ACTIVE_SCALE

    if scale_change > ZOOM_SCALE_EPS:
        apply_fov_rescale(
            stream_state["tracks"],
            stream_state["horizon_state"],
            scale_x,
            scale_y,
        )

    stream_state["previous_fx"] = fx_value
    stream_state["previous_fy"] = fy_value

    if not zooming:
        gdx, gdy, gflow_ok = estimate_global_motion(
            stream_state["previous_gray"], current_gray, stream_state["tracks"]
        )
    else:
        gdx, gdy, gflow_ok = 0.0, 0.0, False

    panning = gflow_ok and math.hypot(gdx, gdy) > PAN_ACTIVE_FLOW_PX
    camera_moving = zooming or panning

    if stream_state["was_moving"] and not camera_moving:
        stream_state["force_detection"] = True

    stream_state["was_moving"] = camera_moving
    stream_state["horizon_state"]["flow_y"] = 0.0

    update_horizon(
        stream_state["horizon_state"],
        current_gray,
        sensor_info,
        frame_index,
        camera_moving,
    )

    detection_camera_moving = camera_moving

    if channel == "thermal" and THERMAL_DETECT_WHILE_MOVING:
        detection_camera_moving = False

    run_detection, mode = should_run_detection(
        frame_index,
        stream_state["tracks"],
        detection_camera_moving,
        stream_state["force_detection"],
    )
    stream_state["mode"] = mode

    detections = []

    if run_detection:
        stream_state["force_detection"] = False
        detections = detect_boats(
            frame,
            model,
            sensor_info,
            stream_state["horizon_state"],
            mode,
            channel=channel,
        )

    tracks, next_track_id = update_tracks(
        detections=detections,
        tracks=stream_state["tracks"],
        next_track_id=stream_state["next_track_id"],
        previous_gray=stream_state["previous_gray"],
        current_gray=current_gray,
        sensor_info=sensor_info,
        detection_was_run=run_detection,
        frame_index=frame_index,
        skip_optical=zooming,
        global_flow=(gdx, gdy),
        global_flow_ok=gflow_ok,
    )

    stream_state["tracks"] = tracks
    stream_state["next_track_id"] = next_track_id
    stream_state["previous_gray"] = current_gray.copy()

    return frame, camera_moving


def draw_stream_output(
    frame,
    stream_state,
    sensor_rows,
    fps,
    frame_index,
    video_fps,
    camera_moving,
):
    video_second = frame_index / video_fps
    sensor_info = stream_state["sensor_smooth"]

    if sensor_info is None:
        sensor_info = get_sensor_for_time(sensor_rows, video_second)

    draw_horizon_line(frame, stream_state["horizon_state"])
    draw_tracks(
        frame,
        stream_state["tracks"],
        sensor_info,
        stream_state["horizon_state"],
        video_fps,
    )
    draw_panel(
        frame,
        sensor_info,
        stream_state["horizon_state"],
        fps,
        video_second,
        len(stream_state["tracks"]),
        stream_state["mode"],
        camera_moving,
    )
    draw_text_bg(
        frame,
        stream_state["name"],
        (20, PANEL_HEIGHT + 28),
        scale=0.70,
        color=(255, 255, 255),
        bg=(0, 0, 0),
        thickness=2,
    )


def make_side_by_side(left_frame, right_frame):
    if left_frame is None:
        left_frame = np.zeros(
            (PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8
        )

    if right_frame is None:
        right_frame = np.zeros(
            (PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8
        )

    left_frame = cv2.resize(left_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    right_frame = cv2.resize(right_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    return np.hstack((left_frame, right_frame))


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
