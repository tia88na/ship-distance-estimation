"""Tekne tespiti, termal ön işleme ve detection birleştirme yardımcıları.

Bu dosya RGB ve termal görüntülerde tekne/gemi tespiti için kullanılan ana
yardımcı fonksiyonları içerir. YOLO bölgesel inference işlemi, termal aday
maskeleme, bounding box filtreleme, su hattı noktası hesaplama ve aynı hedefe
ait detection kutularını birleştirme adımları burada yönetilir.

Bu sürümde ana hedef detection kalitesini iyileştirmektir. Mesafe hesabı
değiştirilmeden önce yanlış bbox kaynaklı hatalar azaltılır. Özellikle dar
zoom kayıtlarında alt kısımdaki bina/çatı/sahil parçalarının ve termal
görüntüde gemi parçası gibi görünen kenar yapılarının gemi olarak algılanması
engellenmeye çalışılır.
"""

from typing import TypeAlias, cast

import cv2
from geometry import horizon_y_at, sea_distance_from_image_point
import numpy as np
from sensor_reader import SensorRow


Box: TypeAlias = tuple[float, float, float, float]
IntBox: TypeAlias = tuple[int, int, int, int]
Region: TypeAlias = tuple[str, int, int, int, int]
Detection: TypeAlias = dict[str, object]
HorizonState: TypeAlias = dict[str, object]

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

YOLO_IOU_THRES = 0.50
YOLO_DEVICE = 0
YOLO_HALF = False

# Full frame detection daha yüksek confidence ile çalışır. Full frame içinde
# bina/çatı gibi false positive riski daha yüksektir.
YOLO_CONF_FULL = 0.42

# Ufuk bandı ve tile bölgeleri küçük/uzak hedefler için kullanıldığı için daha
# düşük confidence ile ikinci seviyede taranır.
YOLO_CONF_DEEP = 0.20

YOLO_IMGSZ_FULL = 960
YOLO_IMGSZ_DEEP = 1536

# Termal full-frame eşiği RGB'ye göre biraz daha düşük tutulur ama önceki
# sürüme göre sertleştirildi. Dar FOV termalde gemi olmayan sıcak/soğuk
# parçalar kolayca boat sanılabildiği için full-frame eşiği fazla düşük
# olmamalıdır.
THERMAL_YOLO_CONF_FULL = 0.40
THERMAL_YOLO_CONF_DEEP = 0.24
THERMAL_YOLO_IMGSZ_FULL = 1280
THERMAL_YOLO_IMGSZ_DEEP = 1536

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

STRONG_ZOOM_FOV_H_DEG = 5.0
ZOOM_FOV_H_DEG = 15.0
MID_FOV_H_DEG = 30.0

# Dar FOV test videolarında hedefler genellikle ufuk çizgisine yakın deniz
# bandında olur. Çok aşağıdaki kutular çoğunlukla sahil, bina veya çatı
# false positive üretir.
STRONG_ZOOM_SEARCH_DEPTH_PX = 230
ZOOM_SEARCH_DEPTH_PX = 300
MID_FOV_SEARCH_DEPTH_PX = 390

# Detection filtreleme için search depth'ten biraz daha toleranslı sınır.
STRONG_ZOOM_VALID_DEPTH_PX = 260
ZOOM_VALID_DEPTH_PX = 340
MID_FOV_VALID_DEPTH_PX = 440

BOTTOM_STRUCTURE_Y1_RATIO = 0.62
BOTTOM_STRUCTURE_WATER_RATIO = 0.76
BOTTOM_STRUCTURE_AREA_RATIO = 0.012

# Termal dar FOV görüntülerde ekran kenarında çıkan küçük/orta kutular çoğu
# zaman gerçek gemi değil, yakın geminin/kıyı yapısının parçalarıdır. Gerçek
# gemi ekrana kenardan giriyorsa birkaç frame sonra daha tam görüneceği için
# ilk fragment kutularını elemek daha güvenlidir.
THERMAL_EDGE_MARGIN_RATIO = 0.025
THERMAL_EDGE_FRAGMENT_MAX_WIDTH_RATIO = 0.30
THERMAL_EDGE_FRAGMENT_MAX_AREA_RATIO = 0.070
THERMAL_EDGE_FRAGMENT_MAX_ASPECT = 4.2
THERMAL_FRAGMENT_LOW_CONF = 0.78

# Termal dar FOV'da full-frame kaynaklı düşük güvenli küçük parçalar false
# positive üretmeye çok yatkındır.
THERMAL_NARROW_FULL_MIN_CONF = 0.55
THERMAL_NARROW_DEEP_MIN_CONF = 0.30


def get_sensor_fov_h(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden yatay FOV değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Yatay FOV değeri. Okunamazsa geniş açı varsayımı döner.
    """
    try:
        return float(sensor_info.get("fov_h", 65.7))
    except (TypeError, ValueError):
        return 65.7


def get_sensor_fov_v(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden dikey FOV değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Dikey FOV değeri. Okunamazsa geniş açı varsayımı döner.
    """
    try:
        return float(sensor_info.get("fov_v", 39.9))
    except (TypeError, ValueError):
        return 39.9


def get_sensor_zoom(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden zoom değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Zoom değeri. Okunamazsa 0.0 döner.
    """
    try:
        return float(sensor_info.get("zoom", 0.0))
    except (TypeError, ValueError):
        return 0.0


def is_narrow_fov(sensor_info: SensorRow) -> bool:
    """Mevcut sensör bilgisinin dar FOV / zoomlu olup olmadığını kontrol eder.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        FOV dar veya zoom yüksekse True.
    """
    fov_h = get_sensor_fov_h(sensor_info)
    fov_v = get_sensor_fov_v(sensor_info)
    zoom = get_sensor_zoom(sensor_info)

    return fov_h < ZOOM_FOV_H_DEG or fov_v < 9.0 or zoom >= 0.85


def get_class_name(model: object, cls_id: int) -> str:
    """YOLO modelindeki class id değerini okunabilir sınıf adına çevirir.

    Args:
        model: Ultralytics YOLO model nesnesi.
        cls_id: Model sonucundan gelen sınıf id değeri.

    Returns:
        Sınıf adı bulunursa sınıf adı, bulunamazsa id değerinin string hali.
    """
    names = model.names

    if isinstance(names, dict):
        return str(names.get(cls_id, str(cls_id)))

    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])

    return str(cls_id)


def calculate_iou(box_a: Box, box_b: Box) -> float:
    """İki bounding box arasındaki IoU değerini hesaplar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    return inter_area / max(area_a + area_b - inter_area, 1.0)


def overlap_ratio_small_inside_large(box_a: Box, box_b: Box) -> float:
    """Küçük kutunun büyük kutu içinde ne kadar kaldığını hesaplar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    return inter_area / min(area_a, area_b)


def horizontal_overlap_ratio(box_a: Box, box_b: Box) -> float:
    """İki kutunun yatay eksende ne kadar örtüştüğünü hesaplar."""
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))

    width_a = max(1.0, ax2 - ax1)
    width_b = max(1.0, bx2 - bx1)

    return inter_w / min(width_a, width_b)


def vertical_gap_px(box_a: Box, box_b: Box) -> float:
    """İki kutu arasındaki dikey boşluğu piksel cinsinden hesaplar."""
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b

    if ay2 < by1:
        return by1 - ay2

    if by2 < ay1:
        return ay1 - by2

    return 0.0


def center_x_distance_ratio(box_a: Box, box_b: Box) -> float:
    """İki kutu merkezinin yatay uzaklığını normalize eder."""
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    center_a = (ax1 + ax2) / 2.0
    center_b = (bx1 + bx2) / 2.0

    width_a = max(1.0, ax2 - ax1)
    width_b = max(1.0, bx2 - bx1)

    return abs(center_a - center_b) / max(width_a, width_b)


def box_to_int(box: Box) -> IntBox:
    """Float koordinatlı kutuyu integer koordinatlı kutuya çevirir."""
    x1, y1, x2, y2 = box

    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def visible_box(box: Box) -> IntBox:
    """Bounding box koordinatlarını görüntü sınırlarına kırpar."""
    x1, y1, x2, y2 = box_to_int(box)

    return (
        max(0, x1),
        max(0, y1),
        min(PROCESS_WIDTH - 1, x2),
        min(PROCESS_HEIGHT - 1, y2),
    )


def clamp_track_box(box: Box) -> Box:
    """Track kutusunu aşırı taşmalara karşı güvenli aralıkta tutar."""
    x1, y1, x2, y2 = box

    x1 = max(-2.0 * PROCESS_WIDTH, min(3.0 * PROCESS_WIDTH, x1))
    y1 = max(-2.0 * PROCESS_HEIGHT, min(3.0 * PROCESS_HEIGHT, y1))
    x2 = max(x1 + 2.0, min(3.0 * PROCESS_WIDTH, x2))
    y2 = max(y1 + 2.0, min(3.0 * PROCESS_HEIGHT, y2))

    return x1, y1, x2, y2


def get_waterline_ratio(sensor_info: SensorRow) -> float:
    """Bounding box içindeki su hattı oranını FOV değerine göre seçer."""
    if get_sensor_fov_h(sensor_info) < ZOOM_FOV_H_DEG:
        return WATERLINE_RATIO_ZOOM

    return WATERLINE_RATIO_NORMAL


def get_water_point_from_box(
    box: Box, sensor_info: SensorRow
) -> tuple[float, float]:
    """Bounding box içinden mesafe hesabında kullanılacak su hattı noktasını alır."""
    x1, y1, x2, y2 = box
    height = max(1.0, y2 - y1)

    water_x = (x1 + x2) / 2.0
    water_y = y1 + get_waterline_ratio(sensor_info) * height

    return water_x, water_y


def is_own_ship_box(box: Box) -> bool:
    """Detection kutusunun kameraya ait gemi parçası olup olmadığını kontrol eder."""
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
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


def max_search_depth_below_horizon(sensor_info: SensorRow, mode: str) -> int:
    """YOLO aramasının ufuk altında ne kadar derine ineceğini belirler."""
    if mode == "bottom_deep":
        return PROCESS_HEIGHT

    fov_h = get_sensor_fov_h(sensor_info)

    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        return STRONG_ZOOM_SEARCH_DEPTH_PX

    if fov_h < ZOOM_FOV_H_DEG:
        return ZOOM_SEARCH_DEPTH_PX

    if fov_h < MID_FOV_H_DEG:
        return MID_FOV_SEARCH_DEPTH_PX

    return PROCESS_HEIGHT


def max_valid_depth_below_horizon(sensor_info: SensorRow) -> int:
    """Detection su hattı için izin verilen maksimum ufuk altı derinliği."""
    fov_h = get_sensor_fov_h(sensor_info)

    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        return STRONG_ZOOM_VALID_DEPTH_PX

    if fov_h < ZOOM_FOV_H_DEG:
        return ZOOM_VALID_DEPTH_PX

    if fov_h < MID_FOV_H_DEG:
        return MID_FOV_VALID_DEPTH_PX

    return PROCESS_HEIGHT


def is_probable_bottom_structure(
    det: Detection, sensor_info: SensorRow, horizon_state: HorizonState
) -> bool:
    """RGB görüntüde bina/çatı/sahil parçası olabilecek kutuları ayıklar."""
    if det.get("channel") == "thermal":
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    aspect = width / max(height, 1.0)

    water_y = float(det["water_y"])
    y_horizon = horizon_y_at(horizon_state, float(det["water_x"]))
    depth_below_horizon = water_y - y_horizon

    if (
        get_sensor_fov_h(sensor_info) < ZOOM_FOV_H_DEG
        and y1 > PROCESS_HEIGHT * BOTTOM_STRUCTURE_Y1_RATIO
        and water_y > PROCESS_HEIGHT * BOTTOM_STRUCTURE_WATER_RATIO
    ):
        return True

    if (
        y1 > PROCESS_HEIGHT * BOTTOM_STRUCTURE_Y1_RATIO
        and area_ratio > BOTTOM_STRUCTURE_AREA_RATIO
        and (aspect < 0.55 or aspect > 9.5)
    ):
        return True

    if depth_below_horizon > max_valid_depth_below_horizon(sensor_info):
        return True

    if y2 > PROCESS_HEIGHT * 0.92 and area_ratio > 0.018:
        return True

    return False


def is_thermal_edge_fragment(
    det: Detection, sensor_info: SensorRow
) -> bool:
    """Dar FOV termalde ekran kenarındaki parça kutuları ayıklar.

    Args:
        det: Kontrol edilecek detection.
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Detection yakın gemi/yapı parçası gibi görünüyorsa True.
    """
    if det.get("channel") != "thermal":
        return False

    if not is_narrow_fov(sensor_info):
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / max(height, 1.0)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    width_ratio = width / PROCESS_WIDTH
    conf = float(det.get("conf", 0.0))
    source = str(det.get("source", ""))

    left_edge = x1 <= PROCESS_WIDTH * THERMAL_EDGE_MARGIN_RATIO
    right_edge = x2 >= PROCESS_WIDTH * (1.0 - THERMAL_EDGE_MARGIN_RATIO)
    touches_side = left_edge or right_edge

    if not touches_side:
        return False

    # Çok büyük ana gemi kutularını değil, kenarda görünen küçük/orta parçaları
    # hedefliyoruz.
    if (
        width_ratio <= THERMAL_EDGE_FRAGMENT_MAX_WIDTH_RATIO
        and area_ratio <= THERMAL_EDGE_FRAGMENT_MAX_AREA_RATIO
        and aspect <= THERMAL_EDGE_FRAGMENT_MAX_ASPECT
    ):
        return True

    # Full-frame kaynaklı ve yüksek güvenli olmayan kenar kutuları da risklidir.
    if source == "full" and conf < THERMAL_FRAGMENT_LOW_CONF:
        return True

    # Kenara yapışık, alt/orta bölgede ve dikey-kompakt görünen kutular termal
    # yapısal parçalar olabilir.
    if (
        y1 > PROCESS_HEIGHT * 0.25
        and width_ratio <= 0.36
        and area_ratio <= 0.090
        and aspect <= 3.2
    ):
        return True

    return False


def is_thermal_low_quality_fragment(
    det: Detection, sensor_info: SensorRow
) -> bool:
    """Dar FOV termalde düşük güvenli küçük parça detectionlarını ayıklar."""
    if det.get("channel") != "thermal":
        return False

    if not is_narrow_fov(sensor_info):
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    width_ratio = width / PROCESS_WIDTH
    conf = float(det.get("conf", 0.0))
    source = str(det.get("source", ""))

    if source == "full" and conf < THERMAL_NARROW_FULL_MIN_CONF:
        return True

    if source != "full" and conf < THERMAL_NARROW_DEEP_MIN_CONF:
        return True

    if (
        area_ratio < 0.012
        and width_ratio < 0.22
        and conf < THERMAL_FRAGMENT_LOW_CONF
    ):
        return True

    return False


def filter_detection(
    det: Detection, sensor_info: SensorRow, horizon_state: HorizonState
) -> bool:
    """Ham detection sonucunu geometri ve boyut kurallarına göre filtreler."""
    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1
    area = width * height
    frame_area = PROCESS_WIDTH * PROCESS_HEIGHT
    is_thermal = det.get("channel") == "thermal"
    fov_h = get_sensor_fov_h(sensor_info)

    if is_thermal:
        min_width = 14
        min_height = 8
    else:
        min_width = 6
        min_height = 5

    if width < min_width or height < min_height:
        return False

    if is_thermal:
        if fov_h < ZOOM_FOV_H_DEG:
            min_area = 95
        elif fov_h < MID_FOV_H_DEG:
            min_area = 130
        else:
            min_area = 180
    elif fov_h < STRONG_ZOOM_FOV_H_DEG:
        min_area = 26
    elif fov_h < ZOOM_FOV_H_DEG:
        min_area = 40
    elif fov_h < MID_FOV_H_DEG:
        min_area = 90
    else:
        min_area = 220

    if area < min_area:
        return False

    if area > frame_area * 0.45:
        return False

    aspect = width / max(height, 1.0)

    if is_thermal:
        if aspect < 0.45 or aspect > 16.0:
            return False

        if y1 > PROCESS_HEIGHT * 0.82:
            return False

        if float(det["water_y"]) > PROCESS_HEIGHT * 0.94:
            return False

        if is_thermal_edge_fragment(det, sensor_info):
            return False

        if is_thermal_low_quality_fragment(det, sensor_info):
            return False

        if (
            str(det.get("source", "")).startswith("thermal_blob")
            and area < THERMAL_BLOB_MIN_AREA
        ):
            return False
    else:
        if aspect < 0.35 or aspect > 16.0:
            return False

        if fov_h < ZOOM_FOV_H_DEG and aspect < 0.55:
            return False

    if is_own_ship_box(box):
        return False

    y_horizon = horizon_y_at(horizon_state, float(det["water_x"]))

    if float(det["water_y"]) <= y_horizon + 0.5:
        return False

    if is_probable_bottom_structure(det, sensor_info, horizon_state):
        return False

    result = sea_distance_from_image_point(
        float(det["water_x"]),
        float(det["water_y"]),
        sensor_info,
        horizon_state,
    )

    if (
        result["valid"]
        and float(result["distance"]) < OWN_SHIP_NEAR_DISTANCE_M
        and y2 > PROCESS_HEIGHT * OWN_SHIP_NEAR_BOTTOM_RATIO
    ):
        return False

    return True


def build_search_regions(
    sensor_info: SensorRow, horizon_state: HorizonState, mode: str
) -> list[Region]:
    """YOLO'nun çalışacağı görüntü bölgelerini üretir."""
    y_h = int(max(10, min(PROCESS_HEIGHT - 40, float(horizon_state["y"]))))
    fov_h = get_sensor_fov_h(sensor_info)

    regions: list[Region] = [("full", 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT)]

    if mode == "full_only":
        return regions

    search_depth = max_search_depth_below_horizon(sensor_info, mode)
    y_bottom = min(PROCESS_HEIGHT, y_h + search_depth)

    regions.append(
        (
            "horizon_strip",
            0,
            max(0, y_h - 70),
            PROCESS_WIDTH,
            min(PROCESS_HEIGHT, y_h + 170),
        )
    )

    regions.append(("sea_band", 0, max(0, y_h - 30), PROCESS_WIDTH, y_bottom))

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

    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        tile_w = 420
        step = 260
        tile_y0 = max(0, y_h - 55)
    elif fov_h < ZOOM_FOV_H_DEG:
        tile_w = 520
        step = 340
        tile_y0 = max(0, y_h - 45)
    else:
        tile_w = 640
        step = 460
        tile_y0 = max(0, y_h - 30)

    x_value = 0

    while x_value < PROCESS_WIDTH:
        x2 = min(PROCESS_WIDTH, x_value + tile_w)
        x1 = max(0, x2 - tile_w)

        regions.append((f"tile_{x1}", x1, tile_y0, x2, y_bottom))

        if x2 >= PROCESS_WIDTH:
            break

        x_value += step

    return regions


def prepare_frame_for_detection(frame: np.ndarray, channel: str) -> np.ndarray:
    """Detection öncesi frame'i kanal tipine göre hazırlar."""
    if channel != "thermal":
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray.astype(np.uint8))
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def create_thermal_candidate_mask(crop_gray: np.ndarray) -> np.ndarray | None:
    """Termal crop içinde sıcak/soğuk aday bölgeler için maske üretir."""
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

    _ = dark

    if std_value < THERMAL_BLOB_MIN_CONTRAST:
        return None

    mask = bright

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def detect_thermal_blobs(
    frame: np.ndarray,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    mode: str,
) -> list[Detection]:
    """YOLO sonuç yoksa termal görüntüden blob tabanlı aday detection üretir."""
    if not THERMAL_BLOB_DETECTION_ENABLED:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = build_search_regions(sensor_info, horizon_state, mode)
    detections: list[Detection] = []

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
                float(max(0, x1 + bx - pad_x)),
                float(max(0, y1 + by - pad_y)),
                float(min(PROCESS_WIDTH - 1, x1 + bx + bw + pad_x)),
                float(min(PROCESS_HEIGHT - 1, y1 + by + bh + pad_y)),
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

            det: Detection = {
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


def region_confidence(
    region_name: str,
    base_full_conf: float,
    base_deep_conf: float,
    channel: str,
    sensor_info: SensorRow,
) -> float:
    """Region adına göre YOLO confidence eşiğini seçer."""
    if region_name == "full":
        conf = base_full_conf
    elif region_name in {"horizon_strip", "sea_band"}:
        conf = base_deep_conf
    elif region_name.startswith("tile_"):
        conf = base_deep_conf
    else:
        conf = base_full_conf

    if channel == "thermal" and is_narrow_fov(sensor_info):
        if region_name == "full":
            return max(conf, THERMAL_NARROW_FULL_MIN_CONF)

        return max(conf, THERMAL_NARROW_DEEP_MIN_CONF)

    return conf


def run_yolo_region(
    frame: np.ndarray,
    model: object,
    region: Region,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    conf_thres: float,
    imgsz: int,
    channel: str = "rgb",
) -> list[Detection]:
    """Tek bir region üzerinde YOLO inference çalıştırır."""
    region_name, x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0 or (y2 - y1) < 24:
        return []

    detections: list[Detection] = []

    results = model.predict(
        crop,
        conf=conf_thres,
        imgsz=imgsz,
        iou=YOLO_IOU_THRES,
        verbose=False,
        classes=[8],
        max_det=18 if channel == "thermal" else 60,
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
                float(max(0, bx1 + x1)),
                float(max(0, by1 + y1)),
                float(min(PROCESS_WIDTH - 1, bx2 + x1)),
                float(min(PROCESS_HEIGHT - 1, by2 + y1)),
            )

            water_x, water_y = get_water_point_from_box(abs_box, sensor_info)

            det: Detection = {
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


def detection_quality_score(det: Detection) -> float:
    """Detection kutusunun birleştirme sırasındaki kalite skorunu hesaplar."""
    x1, y1, x2, y2 = cast(Box, det["box"])

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height

    area_score = min(area / (PROCESS_WIDTH * PROCESS_HEIGHT), 0.35)
    water_score = float(det["water_y"]) / PROCESS_HEIGHT

    source = str(det.get("source", ""))
    region_bonus = 0.0

    if source == "horizon_strip":
        region_bonus = 0.25
    elif source == "sea_band":
        region_bonus = 0.18
    elif source.startswith("tile_"):
        region_bonus = 0.12

    return (
        2.2 * float(det["conf"])
        + 1.2 * area_score
        + 0.8 * water_score
        + region_bonus
    )


def same_vessel(det_a: Detection, det_b: Detection) -> bool:
    """İki detection'ın aynı gemiye ait olup olmadığını kontrol eder."""
    box_a = cast(Box, det_a["box"])
    box_b = cast(Box, det_b["box"])

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


def merge_detection_group(
    group: list[Detection], sensor_info: SensorRow
) -> Detection:
    """Aynı hedefe ait detection grubunu tek detection'a indirger."""
    main_det = max(group, key=detection_quality_score)

    water_x, water_y = get_water_point_from_box(
        cast(Box, main_det["box"]), sensor_info
    )

    return {
        "box": main_det["box"],
        "conf": max(float(det["conf"]) for det in group),
        "water_x": water_x,
        "water_y": water_y,
        "source": main_det["source"],
        "channel": main_det.get("channel", "rgb"),
    }


def merge_same_vessel_detections(
    detections: list[Detection], sensor_info: SensorRow
) -> list[Detection]:
    """Aynı gemiye ait tekrar detection kutularını birleştirir."""
    detections = sorted(detections, key=detection_quality_score, reverse=True)

    groups: list[list[Detection]] = []

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

    kept: list[Detection] = []

    for det in merged:
        if not any(same_vessel(det, kept_det) for kept_det in kept):
            kept.append(det)

    return kept


def detect_boats(
    frame: np.ndarray,
    model: object,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    mode: str,
    channel: str = "rgb",
) -> list[Detection]:
    """RGB veya termal frame üzerinde tekne detection akışını çalıştırır."""
    is_thermal = channel == "thermal"
    fov_h = get_sensor_fov_h(sensor_info)

    if is_thermal:
        if fov_h < MID_FOV_H_DEG:
            imgsz = THERMAL_YOLO_IMGSZ_DEEP
        else:
            imgsz = THERMAL_YOLO_IMGSZ_FULL

        conf_full = THERMAL_YOLO_CONF_FULL
        conf_deep = THERMAL_YOLO_CONF_DEEP
        detection_frame = prepare_frame_for_detection(frame, "thermal")
    else:
        if fov_h < MID_FOV_H_DEG:
            imgsz = YOLO_IMGSZ_DEEP
        else:
            imgsz = YOLO_IMGSZ_FULL

        conf_full = YOLO_CONF_FULL
        conf_deep = YOLO_CONF_DEEP
        detection_frame = frame

    regions = build_search_regions(sensor_info, horizon_state, mode)
    detections: list[Detection] = []

    for region in regions:
        region_name = region[0]
        conf = region_confidence(
            region_name, conf_full, conf_deep, channel, sensor_info
        )

        detections.extend(
            run_yolo_region(
                detection_frame,
                model,
                region,
                sensor_info,
                horizon_state,
                conf,
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
            region_name = region[0]
            conf = region_confidence(
                region_name, conf_full, conf_deep, channel, sensor_info
            )

            detections.extend(
                run_yolo_region(
                    detection_frame,
                    model,
                    region,
                    sensor_info,
                    horizon_state,
                    conf,
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
        merged = sorted(merged, key=detection_quality_score, reverse=True)[:6]

    return merged
