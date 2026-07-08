"""RGB ve termal video frame işleme yardımcıları.

Bu dosya tek bir video frame'i işlerken kullanılan ana akışı içerir. Her frame
için sensör bilgisi alınır, FOV değişimi takip edilir, kamera hareketi tahmin
edilir, ufuk çizgisi güncellenir, gerektiğinde tekne tespiti yapılır ve mevcut
track listesi güncellenir.
"""

import math
from typing import TypeAlias, cast

import cv2
import numpy as np

from detector import detect_boats
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


def process_stream_frame(
    frame: np.ndarray | None,
    stream_state: StreamState,
    sensor_rows: list[SensorRow],
    model: object,
    frame_index: int,
    video_fps: float,
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
        float(sensor_info["fov_h"]),
        float(sensor_info["fov_v"]),
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

        apply_fov_rescale(
            tracks,
            horizon_state,
            scale_x,
            scale_y,
        )

    # Bir sonraki frame için focal length değerleri state içinde saklanır.
    stream_state["previous_fx"] = fx_value
    stream_state["previous_fy"] = fy_value

    previous_gray = cast(np.ndarray | None, stream_state["previous_gray"])
    tracks = cast(TrackMap, stream_state["tracks"])

    # Zoom sırasında optical flow güvenilir olmayacağı için global motion
    # hesaplaması atlanır.
    if not zooming:
        gdx, gdy, gflow_ok = estimate_global_motion(
            previous_gray,
            current_gray,
            tracks,
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
        horizon_state,
        current_gray,
        sensor_info,
        frame_index,
        camera_moving,
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
            frame,
            model,
            sensor_info,
            horizon_state,
            mode,
            channel=channel,
        )

    # Detection sonuçları, optical flow ve önceki track bilgileriyle birleştirilir.
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
    )

    # Güncel track bilgileri bir sonraki frame için state'e yazılır.
    stream_state["tracks"] = tracks
    stream_state["next_track_id"] = next_track_id

    # Bir sonraki optical flow hesaplaması için mevcut gri frame saklanır.
    stream_state["previous_gray"] = current_gray.copy()

    return frame, camera_moving
