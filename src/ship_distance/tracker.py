"""KLT tabanlı takip, kamera hareketi ve mesafe yumuşatma yardımcıları.

Bu dosya detection sonuçlarını zaman içinde track haline getirir. KLT optical
flow, global kamera hareketi, detection-track eşleştirme, FOV değişimine göre
track kutusu ölçekleme ve mesafe değerinin frame'ler arasında yumuşatılması
burada yapılır.
"""

from collections import deque
from statistics import median
from typing import TypeAlias, cast

import cv2
import numpy as np

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
    clamp_horizon_y,
    sea_distance_from_image_point,
)
from sensor_reader import SensorRow


Box: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]
Detection: TypeAlias = dict[str, object]
Track: TypeAlias = dict[str, object]
TrackMap: TypeAlias = dict[int, Track]
HorizonState: TypeAlias = dict[str, object]
DistanceResult: TypeAlias = dict[str, object]

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


def estimate_global_motion(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    tracks: TrackMap,
) -> tuple[float, float, bool]:
    """Track dışındaki görüntü bölgelerinden global kamera hareketini hesaplar.

    Kamera pan hareketi varsa tüm görüntü benzer yönde kayar. Bu fonksiyon
    mevcut track kutularını maskeler ve kalan arka plan noktalarından median
    optical flow değeri üretir.

    Args:
        previous_gray: Bir önceki gri frame.
        current_gray: Mevcut gri frame.
        tracks: Aktif track sözlüğü.

    Returns:
        x hareketi, y hareketi ve hareket tahmininin güvenilir olup olmadığı.
    """
    if previous_gray is None:
        return 0.0, 0.0, False

    # Track kutuları global motion hesabına katılmamalıdır. Çünkü hedeflerin
    # kendi hareketi kamera hareketi gibi algılanabilir.
    mask = np.full((PROCESS_HEIGHT, PROCESS_WIDTH), 255, dtype=np.uint8)

    for track in tracks.values():
        x1, y1, x2, y2 = visible_box(cast(Box, track["box"]))

        if x2 > x1 and y2 > y1:
            pad = 14
            mask[
                max(0, y1 - pad) : min(PROCESS_HEIGHT, y2 + pad),
                max(0, x1 - pad) : min(PROCESS_WIDTH, x2 + pad),
            ] = 0

    # Arka plan üzerinde takip edilebilir köşe noktaları seçilir.
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

    # Seçilen noktaların bir sonraki frame'deki karşılıkları KLT ile bulunur.
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

    # Median değer, tekil hatalı optical flow noktalarına karşı daha dayanıklıdır.
    dx_med = float(np.median(flow[:, 0]))
    dy_med = float(np.median(flow[:, 1]))

    # Median'dan fazla sapan noktalar outlier kabul edilip temizlenir.
    residual = np.sqrt((flow[:, 0] - dx_med) ** 2 + (flow[:, 1] - dy_med) ** 2)
    keep = residual < max(3.0, float(np.percentile(residual, 75)) + 3.0)

    if int(np.sum(keep)) < GLOBAL_MIN_POINTS:
        return 0.0, 0.0, False

    dx_value = float(np.median(flow[keep, 0]))
    dy_value = float(np.median(flow[keep, 1]))

    # Aşırı optical flow değerleri kamera hareketi için güvenli aralıkta tutulur.
    dx_value = max(-GLOBAL_MAX_FLOW_PX, min(GLOBAL_MAX_FLOW_PX, dx_value))
    dy_value = max(-GLOBAL_MAX_FLOW_PX, min(GLOBAL_MAX_FLOW_PX, dy_value))

    return dx_value, dy_value, True


def rescale_point(
    x_value: float,
    y_value: float,
    scale_x: float,
    scale_y: float,
) -> Point:
    """Bir noktayı görüntü merkezi etrafında FOV ölçeğine göre yeniden konumlar.

    Args:
        x_value: Noktanın x koordinatı.
        y_value: Noktanın y koordinatı.
        scale_x: Yatay ölçek katsayısı.
        scale_y: Dikey ölçek katsayısı.

    Returns:
        Ölçeklenmiş x ve y koordinatı.
    """
    return (CX + (x_value - CX) * scale_x, CY + (y_value - CY) * scale_y)


def apply_fov_rescale(
    tracks: TrackMap,
    horizon_state: HorizonState,
    scale_x: float,
    scale_y: float,
) -> None:
    """FOV/zoom değişiminde track ve ufuk bilgisini yeni ölçeğe uyarlar.

    Args:
        tracks: Aktif track sözlüğü.
        horizon_state: Güncel ufuk çizgisi durumu.
        scale_x: Yatay FOV ölçek değişimi.
        scale_y: Dikey FOV ölçek değişimi.
    """
    # Zoom değişince ufuk çizgisi görüntü merkezine göre yukarı/aşağı kayabilir.
    if horizon_state["y"] is not None:
        horizon_state["y"] = clamp_horizon_y(
            CY + (float(horizon_state["y"]) - CY) * scale_y
        )

    # Büyük FOV değişimlerinde eski KLT noktaları güvenilmez olur.
    big_change = max(abs(scale_x - 1.0), abs(scale_y - 1.0)) > 0.02

    for track in tracks.values():
        x1, y1, x2, y2 = cast(Box, track["box"])

        nx1, ny1 = rescale_point(x1, y1, scale_x, scale_y)
        nx2, ny2 = rescale_point(x2, y2, scale_x, scale_y)

        track["box"] = clamp_track_box((nx1, ny1, nx2, ny2))

        # Detection ile ölçülen son kutu da aynı ölçeğe taşınır.
        previous_measured_box = track.get("prev_measured_box")

        if previous_measured_box is not None:
            px1, py1, px2, py2 = cast(Box, previous_measured_box)

            qx1, qy1 = rescale_point(px1, py1, scale_x, scale_y)
            qx2, qy2 = rescale_point(px2, py2, scale_x, scale_y)

            track["prev_measured_box"] = (qx1, qy1, qx2, qy2)

        # Box velocity de görüntü ölçeğiyle birlikte ölçeklenir.
        vx1, vy1, vx2, vy2 = cast(
            Box, track.get("velocity_box", (0.0, 0.0, 0.0, 0.0))
        )
        track["velocity_box"] = (
            vx1 * scale_x,
            vy1 * scale_y,
            vx2 * scale_x,
            vy2 * scale_y,
        )

        # Su hattı geçmişi y ekseninde ölçeklenir.
        water_hist = cast(deque[float], track["water_hist"])
        old_hist = list(water_hist)
        water_hist.clear()

        for value in old_hist:
            water_hist.append(CY + (value - CY) * scale_y)

        points = cast(np.ndarray | None, track.get("klt_points"))

        if points is not None:
            if big_change:
                track["klt_points"] = None
            else:
                points[:, 0, 0] = CX + (points[:, 0, 0] - CX) * scale_x
                points[:, 0, 1] = CY + (points[:, 0, 1] - CY) * scale_y


def init_klt_points(gray: np.ndarray, box: Box) -> np.ndarray | None:
    """Track kutusu içinde KLT için takip edilecek köşe noktalarını başlatır.

    Args:
        gray: Gri seviye frame.
        box: Track veya detection bounding box değeri.

    Returns:
        KLT noktaları veya yeterli nokta bulunamazsa None.
    """
    x1, y1, x2, y2 = visible_box(box)

    width = x2 - x1
    height = y2 - y1

    # Çok küçük kutularda güvenilir köşe noktası üretilemez.
    if width < 15 or height < 15:
        return None

    # Kutunun kenarları gürültülü olabileceği için ROI biraz içeriden seçilir.
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

    # ROI içindeki lokal noktalar full frame koordinatına çevrilir.
    points[:, 0, 0] += rx1
    points[:, 0, 1] += ry1

    return points.astype(np.float32)


def apply_klt_to_track(
    track: Track,
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray | None,
) -> bool:
    """Tek bir track kutusunu KLT optical flow ile bir frame ileri taşır.

    Args:
        track: Güncellenecek track sözlüğü.
        previous_gray: Bir önceki gri frame.
        current_gray: Mevcut gri frame.

    Returns:
        KLT güncellemesi başarılıysa True.
    """
    if previous_gray is None or current_gray is None:
        return False

    points_prev = cast(np.ndarray | None, track.get("klt_points"))

    # Track içinde yeterli KLT noktası yoksa yeniden başlatılır.
    if points_prev is None or len(points_prev) < KLT_MIN_POINTS:
        points_prev = init_klt_points(previous_gray, cast(Box, track["box"]))

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

    # Median'dan çok sapan noktalar track hareketini bozmasın diye elenir.
    residual = np.sqrt((flow[:, 0] - dx_med) ** 2 + (flow[:, 1] - dy_med) ** 2)
    keep = residual < max(8.0, float(np.percentile(residual, 75)) + 8.0)

    if int(np.sum(keep)) < KLT_MIN_POINTS:
        track["klt_points"] = None
        return False

    dx_value = float(np.median(flow[keep, 0]))
    dy_value = float(np.median(flow[keep, 1]))

    # Tek frame'de aşırı kutu zıplamasını engellemek için hareket sınırlandırılır.
    dx_value = max(-KLT_MAX_STEP_PX, min(KLT_MAX_STEP_PX, dx_value))
    dy_value = max(-KLT_MAX_STEP_PX, min(KLT_MAX_STEP_PX, dy_value))

    x1, y1, x2, y2 = cast(Box, track["box"])

    track["box"] = clamp_track_box(
        (x1 + dx_value, y1 + dy_value, x2 + dx_value, y2 + dy_value)
    )

    # Bir sonraki frame'de kullanmak üzere başarılı noktalar saklanır.
    track["klt_points"] = good_next[keep].reshape(-1, 1, 2).astype(np.float32)

    return True


def update_track_velocity(track: Track, measured_box: Box) -> None:
    """Detection ile ölçülen kutudan track velocity değerini günceller.

    Args:
        track: Güncellenecek track.
        measured_box: Detection sonucundan gelen ölçülen kutu.
    """
    previous = cast(Box | None, track.get("prev_measured_box"))
    current = tuple(float(value) for value in measured_box)

    if previous is None:
        track["velocity_box"] = (0.0, 0.0, 0.0, 0.0)
        track["prev_measured_box"] = current
        return

    measured_velocity = tuple(
        cur - prev for cur, prev in zip(current, previous)
    )
    old_velocity = cast(Box, track.get("velocity_box", (0.0, 0.0, 0.0, 0.0)))

    # Ani detection hareketlerini yumuşatmak için velocity EMA ile güncellenir.
    track["velocity_box"] = tuple(
        0.80 * old + 0.20 * new
        for old, new in zip(old_velocity, measured_velocity)
    )

    track["prev_measured_box"] = current


def shift_track_box(track: Track, dx_value: float, dy_value: float) -> None:
    """Track kutusunu verilen x/y kayması kadar taşır.

    Args:
        track: Taşınacak track.
        dx_value: x ekseni kayması.
        dy_value: y ekseni kayması.
    """
    x1, y1, x2, y2 = cast(Box, track["box"])

    track["box"] = clamp_track_box(
        (x1 + dx_value, y1 + dy_value, x2 + dx_value, y2 + dy_value)
    )


def apply_velocity_prediction(track: Track) -> None:
    """Detection/KLT yoksa track kutusunu velocity bilgisiyle tahmin eder.

    Args:
        track: Güncellenecek track.
    """
    velocity = cast(Box, track.get("velocity_box", (0.0, 0.0, 0.0, 0.0)))
    box = cast(Box, track["box"])

    predicted = tuple(b + v for b, v in zip(box, velocity))

    track["box"] = clamp_track_box(cast(Box, predicted))


def smooth_value(
    old_value: float | None,
    new_value: float,
    alpha: float,
) -> float:
    """Tek bir sayısal değeri ağırlıklı ortalama ile yumuşatır.

    Args:
        old_value: Önceki değer.
        new_value: Yeni ölçülen değer.
        alpha: Yeni değerin ağırlığı.

    Returns:
        Yumuşatılmış değer.
    """
    if old_value is None:
        return new_value

    return (1.0 - alpha) * old_value + alpha * new_value


def smooth_box(old_box: Box | None, new_box: Box, alpha: float) -> Box:
    """Bounding box koordinatlarını ağırlıklı ortalama ile yumuşatır.

    Args:
        old_box: Önceki kutu.
        new_box: Yeni ölçülen kutu.
        alpha: Yeni kutunun ağırlığı.

    Returns:
        Yumuşatılmış bounding box.
    """
    if old_box is None:
        return tuple(float(value) for value in new_box)

    return tuple(
        (1.0 - alpha) * old + alpha * new for old, new in zip(old_box, new_box)
    )


def refresh_track_water_point(track: Track, sensor_info: SensorRow) -> None:
    """Track kutusundan güncel su hattı noktasını hesaplar.

    Args:
        track: Güncellenecek track.
        sensor_info: Mevcut frame'e ait sensör bilgisi.
    """
    vx1, vy1, vx2, vy2 = visible_box(cast(Box, track["box"]))

    if vx2 <= vx1 or vy2 <= vy1:
        return

    water_x, water_y = get_water_point_from_box(
        (float(vx1), float(vy1), float(vx2), float(vy2)), sensor_info
    )

    water_hist = cast(deque[float], track["water_hist"])

    # Su hattı y değeri median ile yumuşatılır. Bu, mesafe hesabındaki zıplamayı
    # azaltır.
    water_hist.append(water_y)
    track["water_x"] = water_x
    track["water_y"] = median(water_hist)


def create_new_track(
    track_id: int,
    det: Detection,
    current_gray: np.ndarray,
    frame_index: int,
) -> Track:
    """Yeni detection sonucundan yeni track sözlüğü oluşturur.

    Args:
        track_id: Yeni track id değeri.
        det: Detection sonucu.
        current_gray: Mevcut gri frame.
        frame_index: Mevcut frame index değeri.

    Returns:
        Yeni oluşturulmuş track sözlüğü.
    """
    det_box = cast(Box, det["box"])
    water_y = float(det["water_y"])

    return {
        "id": track_id,
        "box": tuple(float(value) for value in det_box),
        "water_x": float(det["water_x"]),
        "water_y": water_y,
        "water_hist": deque([water_y], maxlen=WATER_HISTORY_LEN),
        "conf": float(det["conf"]),
        "source": str(det["source"]),
        "channel": det.get("channel", "rgb"),
        "missed": 0,
        "frames_since_update": 0,
        "age": 1,
        "confirmed_updates": 1,
        "last_result": None,
        "velocity_box": (0.0, 0.0, 0.0, 0.0),
        "prev_measured_box": tuple(float(value) for value in det_box),
        "klt_points": init_klt_points(current_gray, det_box),
        "klt_ok": False,
        "global_frame_index": frame_index,
        "range_locked_m": None,
        "range_last_frame": None,
        "range_init_samples": deque(maxlen=RANGE_INIT_SAMPLE_COUNT),
        "range_history": deque(maxlen=RANGE_HISTORY_WINDOW),
        "range_reject_count": 0,
        "recent_raws": deque(maxlen=RECENT_RAW_WINDOW),
    }


def track_match_score(det: Detection, track: Track) -> float:
    """Detection ve track arasındaki eşleşme skorunu hesaplar.

    Args:
        det: Yeni detection sonucu.
        track: Mevcut track.

    Returns:
        0 ile 1 aralığına yakın eşleşme skoru.
    """
    det_box = cast(Box, det["box"])
    track_box = box_to_int(cast(Box, track["box"]))

    iou = calculate_iou(det_box, track_box)
    h_overlap = horizontal_overlap_ratio(det_box, track_box)
    center_ratio = center_x_distance_ratio(det_box, track_box)

    center_score = max(0.0, 1.0 - center_ratio)

    # Su hattı yakınlığı, aynı geminin takip edilip edilmediği için ek sinyal
    # olarak kullanılır.
    water_y_diff = abs(float(det["water_y"]) - float(track["water_y"]))
    water_score = max(0.0, 1.0 - water_y_diff / 180.0)

    return (
        0.40 * iou
        + 0.25 * h_overlap
        + 0.22 * center_score
        + 0.13 * water_score
    )


def update_tracks(
    detections: list[Detection],
    tracks: TrackMap,
    next_track_id: int,
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    sensor_info: SensorRow,
    detection_was_run: bool,
    frame_index: int,
    skip_optical: bool,
    global_flow: tuple[float, float],
    global_flow_ok: bool,
) -> tuple[TrackMap, int]:
    """Aktif track listesini detection, KLT ve prediction ile günceller.

    Args:
        detections: Mevcut frame'de bulunan detection listesi.
        tracks: Önceden aktif olan track sözlüğü.
        next_track_id: Yeni track açılırsa kullanılacak id.
        previous_gray: Bir önceki gri frame.
        current_gray: Mevcut gri frame.
        sensor_info: Mevcut frame'e ait sensör bilgisi.
        detection_was_run: Bu frame'de detection çalışıp çalışmadığı.
        frame_index: Mevcut frame index değeri.
        skip_optical: Optical flow'un atlanıp atlanmayacağı.
        global_flow: Global kamera hareketi.
        global_flow_ok: Global kamera hareketinin güvenilir olup olmadığı.

    Returns:
        Güncellenmiş track sözlüğü ve bir sonraki track id değeri.
    """
    gdx, gdy = global_flow

    for track in tracks.values():
        track["global_frame_index"] = frame_index
        track["frames_since_update"] = int(track.get("frames_since_update", 0)) + 1
        track["klt_ok"] = False

        if not skip_optical:
            klt_ok = apply_klt_to_track(track, previous_gray, current_gray)

            if klt_ok:
                track["klt_ok"] = True
            elif global_flow_ok:
                shift_track_box(track, gdx, gdy)
            else:
                apply_velocity_prediction(track)

            # KLT noktaları belirli aralıklarla yeniden başlatılır. Bu, zamanla
            # bozulan köşe noktalarını tazeler.
            if int(track.get("frames_since_update", 0)) % KLT_REINIT_EVERY == 0:
                track["klt_points"] = init_klt_points(
                    current_gray, cast(Box, track["box"])
                )

        refresh_track_water_point(track, sensor_info)

    matched_track_ids: set[int] = set()
    matched_detection_indices: set[int] = set()

    candidates: list[tuple[float, int, int]] = []

    # Her detection, her aktif track ile skorlanır.
    for det_idx, det in enumerate(detections):
        for track_id, track in tracks.items():
            score = track_match_score(det, track)

            if score >= TRACK_MATCH_SCORE_THRES:
                candidates.append((score, det_idx, track_id))

    # En iyi eşleşmeler önce uygulanır.
    candidates.sort(reverse=True, key=lambda item: item[0])

    for _, det_idx, track_id in candidates:
        if det_idx in matched_detection_indices:
            continue

        if track_id in matched_track_ids:
            continue

        det = detections[det_idx]
        track = tracks[track_id]
        det_box = cast(Box, det["box"])

        update_track_velocity(track, det_box)

        track["box"] = smooth_box(cast(Box, track["box"]), det_box, BOX_ALPHA)
        track["conf"] = smooth_value(
            cast(float | None, track.get("conf")),
            float(det["conf"]),
            CONF_ALPHA,
        )
        track["source"] = str(det["source"])
        track["channel"] = det.get("channel", track.get("channel", "rgb"))
        track["missed"] = 0
        track["frames_since_update"] = 0
        track["age"] = int(track["age"]) + 1
        track["confirmed_updates"] = int(track["confirmed_updates"]) + 1
        track["klt_points"] = init_klt_points(current_gray, det_box)

        water_hist = cast(deque[float], track["water_hist"])
        water_hist.append(float(det["water_y"]))
        track["water_x"] = float(det["water_x"])
        track["water_y"] = median(water_hist)

        matched_track_ids.add(track_id)
        matched_detection_indices.add(det_idx)

    # Eşleşmeyen detection'lar yeni track olarak açılır.
    for det_idx, det in enumerate(detections):
        if det_idx in matched_detection_indices:
            continue

        tracks[next_track_id] = create_new_track(
            next_track_id, det, current_gray, frame_index
        )

        next_track_id += 1

    # Uzun süre güncellenmeyen, görüntü dışına çıkan veya fazla missed alan
    # track'ler silinir.
    for track_id in list(tracks.keys()):
        track = tracks[track_id]

        if detection_was_run and track_id not in matched_track_ids:
            track["missed"] = int(track.get("missed", 0)) + 1

        vx1, vy1, vx2, vy2 = visible_box(cast(Box, track["box"]))
        visible_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)

        fully_outside = visible_area <= 0

        if (
            fully_outside
            or int(track["missed"]) > TRACK_MAX_MISSED_DETECTIONS
            or int(track["frames_since_update"]) > TRACK_MAX_STALE_FRAMES
        ):
            del tracks[track_id]

    return tracks, next_track_id


def clamp_range_change(
    previous_distance: float | None,
    candidate_distance: float,
    max_delta: float,
) -> float:
    """Mesafe değerinin tek frame'de izin verilen aralıktan fazla değişmesini önler.

    Args:
        previous_distance: Önceki kilitli mesafe.
        candidate_distance: Yeni aday mesafe.
        max_delta: İzin verilen maksimum değişim.

    Returns:
        Sınırlandırılmış mesafe değeri.
    """
    if previous_distance is None:
        return candidate_distance

    lower = previous_distance - max_delta
    upper = previous_distance + max_delta

    return max(lower, min(upper, candidate_distance))


def calculate_track_distance(
    track: Track,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    video_fps: float,
) -> DistanceResult:
    """Track için su hattı noktasından mesafe hesaplar ve sonucu yumuşatır.

    Args:
        track: Mesafesi hesaplanacak track.
        sensor_info: Mevcut frame'e ait sensör bilgisi.
        horizon_state: Güncel ufuk çizgisi durumu.
        video_fps: Video FPS değeri.

    Returns:
        Mesafe hesaplama sonucu.
    """
    raw_result = cast(
        DistanceResult,
        sea_distance_from_image_point(
            float(track["water_x"]),
            float(track["water_y"]),
            sensor_info,
            horizon_state,
        ),
    )

    if not raw_result["valid"]:
        last = cast(DistanceResult | None, track.get("last_result"))

        # Ufuk çizgisi ötesindeki ilk sonuç geçersiz olarak saklanır.
        if raw_result["reason"] == "at_or_beyond_horizon" and (
            last is None or not last.get("valid")
        ):
            track["last_result"] = raw_result
            return raw_result

        # Yeni sonuç geçersiz ama önceki geçerli sonuç varsa ekranda son geçerli
        # mesafe korunur.
        if last is not None and last.get("valid"):
            return last

        track["last_result"] = raw_result
        return raw_result

    raw_distance = float(raw_result["distance"])
    recent_raws = cast(deque[float], track["recent_raws"])
    recent_raws.append(raw_distance)

    previous_locked = cast(float | None, track["range_locked_m"])

    if previous_locked is None:
        init_samples = cast(deque[float], track["range_init_samples"])
        init_samples.append(raw_distance)

        locked_distance = median(init_samples)

        # İlk birkaç örnek alındıktan sonra mesafe kilidi başlatılır.
        if len(init_samples) >= RANGE_INIT_SAMPLE_COUNT:
            track["range_locked_m"] = locked_distance
            track["range_last_frame"] = int(track.get("global_frame_index", 0))

        result = raw_result.copy()
        result["raw_distance"] = raw_distance
        result["distance"] = locked_distance

        track["last_result"] = result
        return result

    current_frame = int(track.get("global_frame_index", 0))
    previous_frame = cast(int | None, track.get("range_last_frame"))

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
        track["range_reject_count"] = int(track["range_reject_count"]) + 1

        # Çok sayıda ham mesafe reddedilirse son ham örneklerin median değeriyle
        # yeniden kilit kurulur.
        if (
            int(track["range_reject_count"]) >= RANGE_REJECTS_TO_RELOCK
            and len(recent_raws) >= 5
        ):
            previous_locked = median(recent_raws)
            track["range_reject_count"] = 0
            candidate_distance = previous_locked
            max_delta = previous_locked
        else:
            candidate_distance = previous_locked
    else:
        track["range_reject_count"] = 0

        if int(track.get("frames_since_update", 0)) <= 2:
            alpha = RANGE_UPDATE_ALPHA_DETECTED
        else:
            alpha = RANGE_UPDATE_ALPHA_KLT

        candidate_distance = (
            1.0 - alpha
        ) * previous_locked + alpha * raw_distance

    locked_distance = clamp_range_change(
        previous_locked, candidate_distance, max_delta
    )

    # Mesafe fiziksel olarak geçerli aralıkta tutulur.
    locked_distance = max(
        MIN_VALID_DISTANCE_M, min(MAX_SEA_DISTANCE_M, locked_distance)
    )

    track["range_locked_m"] = locked_distance
    range_history = cast(deque[float], track["range_history"])
    range_history.append(locked_distance)

    if len(range_history) >= 5:
        stable_distance = 0.85 * locked_distance + 0.15 * median(
            range_history
        )
    else:
        stable_distance = locked_distance

    result = raw_result.copy()
    result["raw_distance"] = raw_distance
    result["distance"] = stable_distance

    track["last_result"] = result

    return result


def any_track_near_bottom(tracks: TrackMap) -> bool:
    """Aktif track'lerden birinin görüntünün alt kısmına yakın olup olmadığını bakar.

    Args:
        tracks: Aktif track sözlüğü.

    Returns:
        Alt bölgeye yakın track varsa True.
    """
    for track in tracks.values():
        if int(track.get("frames_since_update", 0)) > TRACK_DRAW_MAX_STALE_FRAMES:
            continue

        _, _, _, y2 = box_to_int(cast(Box, track["box"]))

        if y2 > PROCESS_HEIGHT * 0.72:
            return True

    return False


def active_track_count(tracks: TrackMap) -> int:
    """Ekranda güvenilir kabul edilen aktif track sayısını hesaplar.

    Args:
        tracks: Aktif track sözlüğü.

    Returns:
        Yaşı ve update sayısı yeterli olan aktif track sayısı.
    """
    count = 0

    for track in tracks.values():
        if (
            int(track.get("frames_since_update", 0))
            <= TRACK_DRAW_MAX_STALE_FRAMES
            and int(track.get("confirmed_updates", 0))
            >= TRACK_MIN_CONFIRMED_UPDATES
        ):
            count += 1

    return count


def should_run_detection(
    frame_index: int,
    tracks: TrackMap,
    camera_moving: bool,
    force_detection: bool,
) -> tuple[bool, str]:
    """Mevcut frame'de detection çalışıp çalışmayacağına karar verir.

    Args:
        frame_index: Mevcut frame index değeri.
        tracks: Aktif track sözlüğü.
        camera_moving: Kamera hareket halinde mi.
        force_detection: Detection zorlanacak mı.

    Returns:
        Detection kararı ve çalışma modu.
    """
    # Kamera hareketliyken detection daha fazla false positive üretebilir.
    if camera_moving:
        return False, "camera_moving"

    if force_detection:
        return True, "deep"

    if active_track_count(tracks) > 0:
        # Track alt bölgedeyse daha kapsamlı bottom search tetiklenir.
        if (
            any_track_near_bottom(tracks)
            and frame_index % DETECT_INTERVAL_BOTTOM_DEEP == 0
        ):
            return True, "bottom_deep"

        if frame_index % DETECT_INTERVAL_TRACKING == 0:
            return True, "full_only"

        return False, "klt_track"

    # Track kaybedilmişse belirli aralıklarla deep/full arama yapılır.
    if frame_index % DETECT_INTERVAL_LOST_DEEP == 0:
        return True, "deep"

    if frame_index % DETECT_INTERVAL_LOST_FULL == 0:
        return True, "full_only"

    return False, "lost_wait"
