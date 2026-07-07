"""Visualization helpers for drawing tracks, labels, panels, and output frames."""

import cv2
import numpy as np

from ship_distance.detector import box_to_int, visible_box
from ship_distance.geometry import (
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    format_distance,
    horizon_y_at,
)
from ship_distance.tracker import calculate_track_distance


SHOW_TRACK_DETAILS = False
DRAW_HORIZON_LINE = False
TRACK_MIN_AGE_TO_DISPLAY = 3
TRACK_MIN_CONFIRMED_UPDATES = 2
TRACK_DRAW_MAX_STALE_FRAMES = 25

PANEL_HEIGHT = 158


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
