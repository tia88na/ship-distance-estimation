"""RGB ve termal video akışları için frame bazlı işleme yardımcıları."""

import math
from typing import Any

import cv2
import numpy as np

from detector import detect_boats
from geometry import create_horizon_state, focal_from_fov, update_horizon
from sensor_reader import get_sensor_for_time, smooth_sensor
from tracker import (
    apply_fov_rescale,
    estimate_global_motion,
    should_run_detection,
    update_tracks,
)
from visualizer import PANEL_HEIGHT, draw_text_bg, ensure_bgr_frame


PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

GLOBAL_MAX_FLOW_PX = 150.0
PAN_ACTIVE_FLOW_PX = 18.0
ZOOM_SCALE_EPS = 0.0015
ZOOM_ACTIVE_SCALE = 0.006
THERMAL_DETECT_WHILE_MOVING = False


def create_stream_state(name: str, channel: str) -> dict[str, Any]:
    """RGB veya termal video kanalı için başlangıç durumunu oluşturur.

    Her kanal kendi track listesini, sensör geçmişini, önceki frame bilgisini
    ve ufuk çizgisi durumunu ayrı tutar. Bu sayede RGB ve termal görüntüler
    birbirinden bağımsız şekilde takip edilir.

    Args:
        name: Kanalın ekranda gösterilecek adı. Örneğin "RGB" veya "THERMAL".
        channel: İşlenecek kanal tipi. Genellikle "rgb" veya "thermal" olur.

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
    stream_state: dict[str, Any],
    sensor_rows: list[dict[str, Any]],
    model: Any,
    frame_index: int,
    video_fps: float,
) -> tuple[np.ndarray, bool]:
    """Tek bir RGB veya termal frame için ana işleme akışını çalıştırır.

    Bu fonksiyon frame'i standart formata getirir, ilgili zamana ait sensör
    bilgisini alır, FOV değişimini kontrol eder, kamera hareketini tahmin eder,
    ufuk çizgisini günceller, gerekli durumlarda YOLO tespiti çalıştırır ve
    track güncellemesini yapar.

    Args:
        frame: İşlenecek RGB veya termal frame. Frame okunamadıysa None olabilir.
        stream_state: Kanalın track, sensör, önceki frame ve ufuk durumunu tutar.
        sensor_rows: CSV dosyasından okunan sensör verisi satırları.
        model: YOLO modeli.
        frame_index: İşlenen frame'in video içindeki sıra numarası.
        video_fps: Video FPS değeri.

    Returns:
        İşlenmiş frame ve kameranın hareket edip etmediğini belirten değer.
    """
    frame = ensure_bgr_frame(frame)

    # Frame okunamadığında pipeline tamamen durmasın diye boş siyah bir görüntü
    # oluşturulur. Böylece RGB veya termal kanaldan biri eksik olsa bile diğer
    # kanal işlenmeye devam edebilir.
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

    # Tüm frame'ler aynı çözünürlüğe getirilir. Böylece detection, tracking,
    # horizon ve çizim fonksiyonları aynı koordinat sistemiyle çalışır.
    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    # Optical flow ve horizon hesaplarında gri görüntü kullanılır.
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    channel = stream_state.get("channel", "rgb")

    # Frame index FPS değerine bölünerek videodaki zaman saniye cinsinden
    # hesaplanır. Bu zaman, CSV sensör verisiyle eşleştirme için kullanılır.
    video_second = frame_index / video_fps

    # Sensör verisi doğrudan kullanılmaz; önce ilgili zamana göre alınır,
    # sonra ani değişimleri azaltmak için yumuşatılır.
    sensor_raw = get_sensor_for_time(sensor_rows, video_second)
    stream_state["sensor_smooth"] = smooth_sensor(
        stream_state["sensor_smooth"],
        sensor_raw,
    )
    sensor_info = stream_state["sensor_smooth"]

    # FOV değerleri focal length değerlerine çevrilir. Önceki focal değerlerle
    # kıyaslanarak zoom değişimi olup olmadığı anlaşılır.
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"],
        sensor_info["fov_v"],
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

    # FOV değiştiğinde mevcut track kutuları ve ufuk çizgisi yeni ölçeğe göre
    # güncellenir. Bu işlem zoom sırasında tracklerin görüntü üzerinde kaymasını
    # azaltır.
    if scale_change > ZOOM_SCALE_EPS:
        apply_fov_rescale(
            stream_state["tracks"],
            stream_state["horizon_state"],
            scale_x,
            scale_y,
        )

    stream_state["previous_fx"] = fx_value
    stream_state["previous_fy"] = fy_value

    # Zoom sırasında global optical flow güvenilir olmayabilir. Bu nedenle
    # kamera hareketi optical flow ile sadece zoom aktif değilken tahmin edilir.
    if not zooming:
        gdx, gdy, gflow_ok = estimate_global_motion(
            stream_state["previous_gray"],
            current_gray,
            stream_state["tracks"],
        )
    else:
        gdx, gdy, gflow_ok = 0.0, 0.0, False

    # Pan hareketi, görüntünün genelindeki medyan hareket büyüklüğüne göre
    # anlaşılır. Zoom veya pan varsa kamera hareketli kabul edilir.
    panning = gflow_ok and math.hypot(gdx, gdy) > PAN_ACTIVE_FLOW_PX
    camera_moving = zooming or panning

    # Kamera hareketi bittikten sonra bir sonraki karede detection zorlanır.
    # Böylece hareket sırasında kaymış olabilecek trackler güncel görüntüye
    # yeniden oturtulur.
    if stream_state["was_moving"] and not camera_moving:
        stream_state["force_detection"] = True

    stream_state["was_moving"] = camera_moving

    # Ufuk çizgisinin optical-flow kaynaklı dikey kayması bu akışta sıfırlanır.
    stream_state["horizon_state"]["flow_y"] = 0.0

    # Ufuk çizgisi, sensör verisi ve mevcut frame bilgisiyle güncellenir.
    # Mesafe tahmini daha sonra bu ufuk referansına göre yapılır.
    update_horizon(
        stream_state["horizon_state"],
        current_gray,
        sensor_info,
        frame_index,
        camera_moving,
    )

    detection_camera_moving = camera_moving

    # Termal kanalda kamera hareketliyken detection çalıştırmak istenirse bu
    # flag True yapılabilir. Varsayılan durumda termal detection hareket
    # sırasında kapalı tutulur.
    if channel == "thermal" and THERMAL_DETECT_WHILE_MOVING:
        detection_camera_moving = False

    # Detection her karede çalışmaz. Tracking durumu, kamera hareketi ve zorunlu
    # detection bayrağına göre bu karede YOLO çalışıp çalışmayacağı belirlenir.
    run_detection, mode = should_run_detection(
        frame_index,
        stream_state["tracks"],
        detection_camera_moving,
        stream_state["force_detection"],
    )
    stream_state["mode"] = mode

    detections = []

    # YOLO tespiti yalnızca gerekli görülen karelerde çalıştırılır. Bu yaklaşım
    # işlem yükünü azaltır ve tracking ile detection arasında denge kurar.
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

    # Detection çıktıları mevcut tracklerle eşleştirilir. Detection çalışmadığı
    # karelerde KLT optical flow ve global hareket bilgisiyle track güncellenir.
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

    # Mevcut gri frame bir sonraki karede optical flow hesabı için saklanır.
    stream_state["previous_gray"] = current_gray.copy()

    return frame, camera_moving
