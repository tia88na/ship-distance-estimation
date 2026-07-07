"""Video stream processing helpers for RGB and thermal frames."""

import cv2

from ship_distance.detector import detect_boats
from ship_distance.geometry import create_horizon_state, update_horizon
from ship_distance.sensor_reader import get_sensor_for_time, smooth_sensor
from ship_distance.tracker import (
    active_track_count,
    any_track_near_bottom,
    apply_fov_rescale,
    estimate_global_motion,
    should_run_detection,
    update_tracks,
)


PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

GLOBAL_MAX_FLOW_PX = 150.0
PAN_ACTIVE_FLOW_PX = 18.0
ZOOM_SCALE_EPS = 0.0015
ZOOM_ACTIVE_SCALE = 0.006
THERMAL_DETECT_WHILE_MOVING = False


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


class VideoProcessor:
    """
    RGB ve termal video akışlarının frame bazlı işlenmesini OOP arayüzü altında
    toplar.

    Bu class mevcut process akışını değiştirmez. Stream state oluşturma ve
    frame işleme fonksiyonlarını tek bir yapı altında gösterir.
    """

    create_stream_state = staticmethod(create_stream_state)
    process_stream_frame = staticmethod(process_stream_frame)
