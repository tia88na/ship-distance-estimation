"""RGB ve termal video frame işleme yardımcıları.

Bu dosya tek bir video frame'i işlerken kullanılan ana akışı içerir. Her frame
için sensör bilgisi alınır, FOV değişimi takip edilir, kamera hareketi tahmin
edilir, ufuk çizgisi güncellenir, gerektiğinde tekne tespiti yapılır ve mevcut
track listesi güncellenir.

RGB ve termal akışların state bilgileri ayrı tutulur. Dar FOV veya yüksek
zoom kullanılan termal görüntülerde, bbox kaynaklı mesafe sapmasını azaltmak
için yalnızca termal track'lere ek güvenlik düzeltmesi uygulanır.
"""

import math
from typing import TypeAlias, cast

import cv2
import numpy as np

from detector import detect_boats
from distance_butterfly_api import DistanceButterflyApi
from distance_hl_api import DistanceHlApi
from geometry import create_horizon_state, focal_from_fov, update_horizon
from sensor_reader import SensorRow, get_sensor_for_time, smooth_sensor
from tracker import (
    apply_fov_rescale,
    estimate_global_motion,
    should_run_detection,
    update_tracks,
)
from visualizer import PANEL_HEIGHT, draw_text_bg, ensure_bgr_frame


StreamState: TypeAlias = dict[str, object]
TrackMap: TypeAlias = dict[int, dict[str, object]]
HorizonState: TypeAlias = dict[str, object]

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

PAN_ACTIVE_FLOW_PX = 18.0
ZOOM_SCALE_EPS = 0.0015
ZOOM_ACTIVE_SCALE = 0.006
THERMAL_DETECT_WHILE_MOVING = False

# Termal görüntüde FOV çok dar veya zoom çok yüksek olduğunda, YOLO kutusu
# çoğu zaman geminin tamamını değil geminin sadece kabin/gövde parçasını alır.
# Bu durumda bbox-size mesafe hesabı aynı fiziksel gemiyi RGB'ye göre daha uzak
# gösterebilir. Aşağıdaki eşikler sadece termal akışta kullanılır.
THERMAL_NARROW_FOV_H_DEG = 12.0
THERMAL_NARROW_FOV_V_DEG = 9.0
THERMAL_HIGH_ZOOM = 0.85
THERMAL_LARGE_BBOX_AREA_RATIO = 0.035
THERMAL_PARTIAL_BBOX_WIDTH_RATIO = 0.55


def create_stream_state(name: str, channel: str) -> StreamState:
    """Bir video akışı için başlangıç state sözlüğünü oluşturur.

    RGB ve termal akışlar aynı işlem adımlarını kullanır. Ancak her akışın
    kendi track listesi, sensör yumuşatma değeri, önceki frame bilgisi ve ufuk
    durumu ayrı tutulur.

    Args:
        name: Ekranda/loglarda kullanılacak akış adı.
        channel: İşlenecek kanal tipi. Genellikle "rgb" veya "thermal".

    Returns:
        Frame işleme sırasında güncellenecek stream state sözlüğü.
    """
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


def safe_float(value: object, default: float) -> float:
    """Bir değeri güvenli şekilde float'a çevirir.

    Args:
        value: Float'a çevrilecek değer.
        default: Dönüşüm başarısızsa kullanılacak değer.

    Returns:
        Float değer.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Sayısal değeri verilen aralıkta sınırlar.

    Args:
        value: Sınırlandırılacak değer.
        min_value: Alt sınır.
        max_value: Üst sınır.

    Returns:
        Sınırlandırılmış değer.
    """
    return max(min_value, min(max_value, value))


def weighted_log_blend(
    value_a: float, weight_a: float, value_b: float, weight_b: float
) -> float:
    """İki pozitif mesafeyi logaritmik ağırlıkla birleştirir.

    Mesafe hataları genellikle çarpansal büyüdüğü için logaritmik ortalama,
    doğrudan aritmetik ortalamaya göre daha stabil davranır.

    Args:
        value_a: İlk mesafe.
        weight_a: İlk mesafe ağırlığı.
        value_b: İkinci mesafe.
        weight_b: İkinci mesafe ağırlığı.

    Returns:
        Ağırlıklandırılmış mesafe.
    """
    value_a = max(value_a, 1.0)
    value_b = max(value_b, 1.0)
    total_weight = max(weight_a + weight_b, 1e-6)

    return math.exp(
        (math.log(value_a) * weight_a + math.log(value_b) * weight_b)
        / total_weight
    )


def get_track_box(track: dict[str, object]) -> tuple[float, float, float, float] | None:
    """Track sözlüğünden bbox değerini bulur.

    Tracker tarafında farklı isimlendirmeler kullanılabildiği için birkaç yaygın
    alan adı kontrol edilir.

    Args:
        track: Tek bir track sözlüğü.

    Returns:
        Bbox tuple değeri veya bulunamazsa None.
    """
    for key in ("box", "bbox", "xyxy"):
        value = track.get(key)

        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return (
                safe_float(value[0], 0.0),
                safe_float(value[1], 0.0),
                safe_float(value[2], 0.0),
                safe_float(value[3], 0.0),
            )

    return None


def get_track_distance(track: dict[str, object]) -> float | None:
    """Track sözlüğündeki kullanılabilir mesafe değerini döndürür."""
    for key in (
        "distance",
        "distance_m",
        "smoothed_distance",
        "filtered_distance",
    ):
        value = track.get(key)

        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)

    return None


def set_track_distance(
    track: dict[str, object], distance_m: float
) -> None:
    """Track sözlüğündeki mesafe alanlarını günceller.

    Args:
        track: Güncellenecek track sözlüğü.
        distance_m: Yeni mesafe değeri.
    """
    # Visualizer tarafı hangi alanı okuyorsa bozulmasın diye yaygın mesafe
    # alanları birlikte güncellenir.
    track["distance"] = distance_m
    track["distance_m"] = distance_m

    if "smoothed_distance" in track:
        track["smoothed_distance"] = distance_m

    if "filtered_distance" in track:
        track["filtered_distance"] = distance_m


def is_thermal_distance_risky(
    box: tuple[float, float, float, float],
    sensor_info: SensorRow,
) -> tuple[bool, str]:
    """Termal bbox mesafesinin riskli olup olmadığını değerlendirir.

    Args:
        box: Track bbox değeri.
        sensor_info: Mevcut frame sensör bilgisi.

    Returns:
        Risk durumu ve risk sebebi.
    """
    x1, y1, x2, y2 = box

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    area_ratio = (box_width * box_height) / float(
        PROCESS_WIDTH * PROCESS_HEIGHT
    )

    fov_h = safe_float(sensor_info.get("fov_h"), 65.7)
    fov_v = safe_float(sensor_info.get("fov_v"), 39.9)
    zoom = safe_float(sensor_info.get("zoom"), 0.0)

    narrow_fov = (
        fov_h <= THERMAL_NARROW_FOV_H_DEG
        or fov_v <= THERMAL_NARROW_FOV_V_DEG
        or zoom >= THERMAL_HIGH_ZOOM
    )

    if not narrow_fov:
        return False, "thermal_fov_not_narrow"

    if area_ratio >= THERMAL_LARGE_BBOX_AREA_RATIO:
        return True, "thermal_large_bbox_in_narrow_fov"

    if box_width <= PROCESS_WIDTH * THERMAL_PARTIAL_BBOX_WIDTH_RATIO:
        return True, "thermal_partial_bbox_in_narrow_fov"

    return False, "thermal_bbox_ok"


def apply_thermal_distance_guard(
    tracks: TrackMap,
    sensor_info: SensorRow,
) -> None:
    """Dar FOV termal track'lerde mesafe sapmasını azaltır.

    Bu fonksiyon video özelinde sabit mesafe ayarı yapmaz. Sadece termal görüntü
    dar FOV/zoomlu olduğunda ve bbox kısmi/büyük göründüğünde devreye girer.

    Mantık:
        - Horizon mesafesi varsa ve final mesafe ondan fazla uzaklaşmışsa,
          final mesafe horizon sonucuna doğru çekilir.
        - Horizon mesafesi yoksa ama bbox dar FOV'da büyük/kısmi görünüyorsa,
          mesafe sınırlı oranda aşağı çekilir.
        - Confidence düşürülür ve track içine hangi kuralın çalıştığı yazılır.

    Args:
        tracks: Güncel track sözlüğü.
        sensor_info: Mevcut frame sensör bilgisi.
    """
    for track in tracks.values():
        box = get_track_box(track)

        if box is None:
            continue

        risky, risk_reason = is_thermal_distance_risky(box, sensor_info)

        if not risky:
            continue

        current_distance = get_track_distance(track)

        if current_distance is None:
            continue

        horizon_distance = track.get("horizon_distance")
        confidence = safe_float(track.get("distance_confidence"), 0.50)

        adjusted_distance = current_distance
        adjustment_reason = risk_reason

        if isinstance(horizon_distance, int | float) and float(horizon_distance) > 1.0:
            horizon_distance_float = float(horizon_distance)

            # Termal final mesafe, horizon mesafesinden belirgin yüksekse
            # dar-FOV bbox-size sapması olabilir. Bu durumda final sonucu horizon
            # sonucuna doğru çekiyoruz ama tamamen horizon-only yapmıyoruz.
            if current_distance > horizon_distance_float * 1.12:
                adjusted_distance = weighted_log_blend(
                    current_distance,
                    0.30,
                    horizon_distance_float,
                    0.70,
                )
                adjustment_reason = f"{risk_reason}_blend_horizon"
        else:
            # Bazı track sürümlerinde horizon_distance tutulmayabilir. Bu durumda
            # sadece dar-FOV termal ve riskli bbox için sınırlı bir düzeltme
            # uygulanır. RGB veya geniş FOV bundan etkilenmez.
            adjusted_distance = current_distance * 0.78
            adjustment_reason = f"{risk_reason}_fallback_scale"

        if adjusted_distance >= current_distance:
            continue

        set_track_distance(track, adjusted_distance)

        track["thermal_distance_guard"] = True
        track["thermal_distance_guard_reason"] = adjustment_reason
        track["thermal_original_distance"] = current_distance
        track["distance_source"] = "thermal_guard"
        track["distance_confidence"] = clamp(confidence * 0.70, 0.03, 0.95)


def process_stream_frame(
    frame: np.ndarray | None,
    stream_state: StreamState,
    sensor_rows: list[SensorRow],
    model: object,
    frame_index: int,
    video_fps: float,
    distance_hl_api: DistanceHlApi,
    distance_butterfly_api: DistanceButterflyApi,
    camera_height_m: float,
) -> tuple[np.ndarray, bool]:
    """Tek bir RGB veya termal frame'i işler.

    Bu fonksiyon frame'i normalize eder, sensör bilgisini video zamanı ile
    eşleştirir, kamera hareketini hesaplar, gerekli durumda detection çalıştırır
    ve track listesini günceller.

    Args:
        frame: İşlenecek ham video frame'i.
        stream_state: İlgili akış için tutulan güncel durum sözlüğü.
        sensor_rows: CSV'den okunan sensör satırları.
        model: Detection sırasında kullanılacak model nesnesi.
        frame_index: İşlenen frame'in video içindeki sıra numarası.
        video_fps: Video FPS değeri.
        distance_hl_api: Ufuk geometrisiyle mesafe hesaplayan API.
        distance_butterfly_api: Bbox boyutuyla mesafe hesaplayan API.
        camera_height_m: Kameranın deniz seviyesinden yüksekliği.

    Returns:
        İşlenmiş frame ve kameranın hareket edip etmediğini belirten değer.
    """
    # Frame formatı BGR olacak şekilde standartlaştırılır.
    frame = ensure_bgr_frame(frame)

    # Frame okunamadıysa boş görüntü oluşturulur ve kullanıcıya bilgi yazılır.
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

    # Bütün akışlar aynı çözünürlükte işlensin diye frame resize edilir.
    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    # Optical flow ve ufuk analizi gri görüntü üzerinde yapılır.
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kanal bilgisi state içinden alınır. Eksikse RGB varsayılır.
    channel = str(stream_state.get("channel", "rgb"))

    # Frame index FPS'e bölünerek video içindeki saniye değeri bulunur.
    video_second = frame_index / video_fps

    # Video zamanına karşılık gelen sensör satırı okunur.
    sensor_raw = get_sensor_for_time(sensor_rows, video_second)

    # Sensör verileri frame'ler arasında ani sıçramaları azaltmak için
    # yumuşatılır.
    previous_sensor = cast(SensorRow | None, stream_state["sensor_smooth"])
    stream_state["sensor_smooth"] = smooth_sensor(previous_sensor, sensor_raw)
    sensor_info = cast(SensorRow, stream_state["sensor_smooth"])

    # Mevcut FOV değerlerinden piksel cinsinden focal length hesaplanır.
    fx_value, fy_value = focal_from_fov(
        float(sensor_info["fov_h"]), float(sensor_info["fov_v"])
    )

    # Önceki focal length değeri varsa FOV/zoom ölçek değişimi hesaplanır.
    previous_fx = stream_state["previous_fx"]
    previous_fy = stream_state["previous_fy"]

    if previous_fx is not None and previous_fy is not None:
        scale_x = fx_value / float(previous_fx)
        scale_y = fy_value / float(previous_fy)
    else:
        scale_x = 1.0
        scale_y = 1.0

    # Ölçek değişimi eşik üstündeyse kamera zoom yapıyor kabul edilir.
    scale_change = max(abs(scale_x - 1.0), abs(scale_y - 1.0))
    zooming = scale_change > ZOOM_ACTIVE_SCALE

    # Küçük FOV değişimlerinde mevcut track kutuları yeni ölçeğe uyarlanır.
    if scale_change > ZOOM_SCALE_EPS:
        tracks = cast(TrackMap, stream_state["tracks"])
        horizon_state = cast(HorizonState, stream_state["horizon_state"])

        apply_fov_rescale(tracks, horizon_state, scale_x, scale_y)

    # Bir sonraki frame için focal length değerleri state içinde saklanır.
    stream_state["previous_fx"] = fx_value
    stream_state["previous_fy"] = fy_value

    previous_gray = cast(np.ndarray | None, stream_state["previous_gray"])
    tracks = cast(TrackMap, stream_state["tracks"])

    # Zoom sırasında optical flow güvenilir olmayacağı için global motion
    # hesaplaması atlanır.
    if not zooming:
        gdx, gdy, gflow_ok = estimate_global_motion(
            previous_gray, current_gray, tracks
        )
    else:
        gdx, gdy, gflow_ok = 0.0, 0.0, False

    # Optical flow yeterince büyükse kamera pan hareketi yapıyor kabul edilir.
    panning = gflow_ok and math.hypot(gdx, gdy) > PAN_ACTIVE_FLOW_PX

    # Kamera hareketi zoom veya pan durumlarından biriyle belirlenir.
    camera_moving = zooming or panning

    # Kamera hareketi durduğunda detection tekrar zorlanır. Böylece hareket
    # sırasında kaçırılan objeler sonraki sabit frame'de tekrar aranır.
    if stream_state["was_moving"] and not camera_moving:
        stream_state["force_detection"] = True

    stream_state["was_moving"] = camera_moving

    horizon_state = cast(HorizonState, stream_state["horizon_state"])

    # Bu akışta ufuk çizgisi için dikey flow katkısı sıfırlanır.
    horizon_state["flow_y"] = 0.0

    # Sensör bilgisi ve mevcut frame kullanılarak ufuk durumu güncellenir.
    update_horizon(
        horizon_state, current_gray, sensor_info, frame_index, camera_moving
    )

    # Varsayılan olarak detection, kamera hareket durumunu dikkate alır.
    detection_camera_moving = camera_moving

    # İstenirse termal akışta kamera hareket ederken de detection yapılabilir.
    if channel == "thermal" and THERMAL_DETECT_WHILE_MOVING:
        detection_camera_moving = False

    # Mevcut track durumu ve kamera hareketine göre detection kararı verilir.
    run_detection, mode = should_run_detection(
        frame_index,
        tracks,
        detection_camera_moving,
        bool(stream_state["force_detection"]),
    )
    stream_state["mode"] = mode

    detections: list[dict[str, object]] = []

    # Detection sadece gerekli görülen frame'lerde çalıştırılır.
    if run_detection:
        stream_state["force_detection"] = False
        detections = detect_boats(
            frame, model, sensor_info, horizon_state, mode, channel=channel
        )

    # Tracker; detection, optical flow ve önceki track bilgisini birleştirir.
    # Mesafe API'leri yalnızca burada aşağı katmana aktarılır. Böylece bu dosya
    # mesafe formüllerini içermez ve frame işleme sorumluluğunda kalır.
    tracks, next_track_id = update_tracks(
        detections=detections,
        tracks=tracks,
        next_track_id=int(stream_state["next_track_id"]),
        previous_gray=previous_gray,
        current_gray=current_gray,
        sensor_info=sensor_info,
        detection_was_run=run_detection,
        frame_index=frame_index,
        skip_optical=zooming,
        global_flow=(gdx, gdy),
        global_flow_ok=gflow_ok,
        distance_hl_api=distance_hl_api,
        distance_butterfly_api=distance_butterfly_api,
        camera_height_m=camera_height_m,
        horizon_state=horizon_state,
        video_fps=video_fps,
    )

    # RGB tarafı aynı kalır. Sadece dar FOV / zoomlu termal görüntülerde,
    # kısmi bbox kaynaklı uzak mesafe sapması azaltılır.
    if channel == "thermal":
        apply_thermal_distance_guard(tracks, sensor_info)

    # Güncel track bilgileri bir sonraki frame için state'e yazılır.
    stream_state["tracks"] = tracks
    stream_state["next_track_id"] = next_track_id

    # Bir sonraki optical flow hesaplaması için mevcut gri frame saklanır.
    stream_state["previous_gray"] = current_gray.copy()

    return frame, camera_moving
