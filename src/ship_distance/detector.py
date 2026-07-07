"""Boat detection, thermal preprocessing, and detection merge helpers."""

import cv2
import numpy as np

from ship_distance.geometry import horizon_y_at, sea_distance_from_image_point


PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

YOLO_IOU_THRES = 0.50
YOLO_DEVICE = 0
YOLO_HALF = False

YOLO_CONF_FULL = 0.35
YOLO_CONF_DEEP = 0.28
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

WATERLINE_RATIO_NORMAL = 0.90
WATERLINE_RATIO_ZOOM = 0.86

OWN_SHIP_BOTTOM_RATIO = 0.90
OWN_SHIP_MIN_HEIGHT_RATIO = 0.30
OWN_SHIP_MAX_AREA_RATIO = 0.40
OWN_SHIP_NEAR_DISTANCE_M = 12.0
OWN_SHIP_NEAR_BOTTOM_RATIO = 0.82

MERGE_IOU_THRES = 0.22
MERGE_INSIDE_THRES = 0.55
MERGE_HORIZONTAL_OVERLAP_THRES = 0.35
MERGE_VERTICAL_GAP_PX = 180
MERGE_CENTER_DISTANCE_RATIO = 0.82


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


class BoatDetector:
    """
    RGB ve termal görüntülerde gemi/tekne tespiti için kullanılan fonksiyonları
    OOP arayüzü altında toplar.

    Bu class mevcut detection akışını değiştirmez. YOLO tespiti, termal aday
    çıkarımı, kutu filtreleme ve aynı gemiye ait kutuları birleştirme gibi
    işlemleri tek bir yapı altında gösterir.
    """

    get_class_name = staticmethod(get_class_name)
    calculate_iou = staticmethod(calculate_iou)
    overlap_ratio_small_inside_large = staticmethod(
        overlap_ratio_small_inside_large
    )
    horizontal_overlap_ratio = staticmethod(horizontal_overlap_ratio)
    vertical_gap_px = staticmethod(vertical_gap_px)
    center_x_distance_ratio = staticmethod(center_x_distance_ratio)
    box_to_int = staticmethod(box_to_int)
    visible_box = staticmethod(visible_box)
    clamp_track_box = staticmethod(clamp_track_box)
    get_waterline_ratio = staticmethod(get_waterline_ratio)
    get_water_point_from_box = staticmethod(get_water_point_from_box)
    is_own_ship_box = staticmethod(is_own_ship_box)
    filter_detection = staticmethod(filter_detection)
    build_search_regions = staticmethod(build_search_regions)
    prepare_frame_for_detection = staticmethod(prepare_frame_for_detection)
    create_thermal_candidate_mask = staticmethod(create_thermal_candidate_mask)
    detect_thermal_blobs = staticmethod(detect_thermal_blobs)
    run_yolo_region = staticmethod(run_yolo_region)
    detection_quality_score = staticmethod(detection_quality_score)
    same_vessel = staticmethod(same_vessel)
    merge_detection_group = staticmethod(merge_detection_group)
    merge_same_vessel_detections = staticmethod(merge_same_vessel_detections)
    detect_boats = staticmethod(detect_boats)
