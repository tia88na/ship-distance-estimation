from geometry import clamp_horizon_y


"""Tracking, KLT motion, and range smoothing helpers."""

from collections import deque
from statistics import median

import cv2
from detector import (
    box_to_int,
    calculate_iou,
    center_x_distance_ratio,
    clamp_track_box,
    get_water_point_from_box,
    horizontal_overlap_ratio,
    visible_box,
)
from geometry import (
    CY,
    MAX_SEA_DISTANCE_M,
    MIN_VALID_DISTANCE_M,
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    sea_distance_from_image_point,
)
import numpy as np


CX = PROCESS_WIDTH / 2.0

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

GLOBAL_MAX_CORNERS = 140
GLOBAL_MIN_POINTS = 12
GLOBAL_MAX_FLOW_PX = 150.0

TRACK_MATCH_SCORE_THRES = 0.15
TRACK_MIN_CONFIRMED_UPDATES = 2
TRACK_MAX_MISSED_DETECTIONS = 6
TRACK_MAX_STALE_FRAMES = 60
TRACK_DRAW_MAX_STALE_FRAMES = 25

BOX_ALPHA = 0.30
CONF_ALPHA = 0.20
WATER_HISTORY_LEN = 7

RANGE_INIT_SAMPLE_COUNT = 4
RANGE_HISTORY_WINDOW = 25
RANGE_UPDATE_ALPHA_DETECTED = 0.20
RANGE_UPDATE_ALPHA_KLT = 0.06
MAX_ACCEPTED_RAW_JUMP_RATIO = 1.60
RANGE_REJECTS_TO_RELOCK = 12
RANGE_RELATIVE_RATE_PER_SEC = 0.10
RANGE_MIN_RATE_M_PER_SEC = 2.0
RECENT_RAW_WINDOW = 15

DETECT_INTERVAL_TRACKING = 6
DETECT_INTERVAL_LOST_FULL = 3
DETECT_INTERVAL_LOST_DEEP = 9
DETECT_INTERVAL_BOTTOM_DEEP = 6


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


class ShipTracker:
    """
    Video boyunca gemi/tekne takibi için kullanılan fonksiyonları OOP arayüzü
    altında toplar.

    Bu class mevcut tracking akışını değiştirmez. KLT optical flow, global
    kamera hareketi, detection-track eşleştirme ve mesafe yumuşatma
    fonksiyonlarını tek bir mantıksal yapı altında gösterir.
    """

    estimate_global_motion = staticmethod(estimate_global_motion)
    rescale_point = staticmethod(rescale_point)
    apply_fov_rescale = staticmethod(apply_fov_rescale)
    init_klt_points = staticmethod(init_klt_points)
    apply_klt_to_track = staticmethod(apply_klt_to_track)
    update_track_velocity = staticmethod(update_track_velocity)
    shift_track_box = staticmethod(shift_track_box)
    apply_velocity_prediction = staticmethod(apply_velocity_prediction)
    smooth_value = staticmethod(smooth_value)
    smooth_box = staticmethod(smooth_box)
    refresh_track_water_point = staticmethod(refresh_track_water_point)
    create_new_track = staticmethod(create_new_track)
    track_match_score = staticmethod(track_match_score)
    update_tracks = staticmethod(update_tracks)
    clamp_range_change = staticmethod(clamp_range_change)
    calculate_track_distance = staticmethod(calculate_track_distance)
    any_track_near_bottom = staticmethod(any_track_near_bottom)
    active_track_count = staticmethod(active_track_count)
    should_run_detection = staticmethod(should_run_detection)
