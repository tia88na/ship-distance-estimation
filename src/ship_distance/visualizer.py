"""Video çıktısı üzerine track, mesafe etiketi ve bilgi paneli çizim yardımcıları.

Bu dosya RGB ve termal akışlardan gelen işlenmiş frame'lerin görselleştirme
katmanını yönetir. Track kutuları, su hattı noktaları, mesafe yazıları, ufuk
çizgisi, üst bilgi paneli ve iki görüntünün yan yana birleştirilmesi burada
yapılır.
"""

import math
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from config import AppConfig
from detector import THERMAL_YOLO_CONF_DEEP, visible_box
from geometry import (
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    focal_from_fov,
    format_distance,
    horizon_y_at,
)
from sensor_reader import SensorRow, get_sensor_for_time
from tracker import calculate_track_distance


Rect: TypeAlias = tuple[int, int, int, int]
Point: TypeAlias = tuple[int, int]
Color: TypeAlias = tuple[int, int, int]
Track: TypeAlias = dict[str, object]
TrackMap: TypeAlias = dict[int, Track]
HorizonState: TypeAlias = dict[str, object]
StreamState: TypeAlias = dict[str, object]

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)

RECORD_NAME = CONFIG.record.name
CAMERA_HEIGHT_M = CONFIG.camera.height_m

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_FACTOR = 7.0 / 6.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * REFRACTION_FACTOR
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

DEFAULT_FOV_H_DEG = CONFIG.camera.rgb_fov_h_deg
DEFAULT_FOV_V_DEG = CONFIG.camera.rgb_fov_v_deg
DEFAULT_THERMAL_FOV_H_DEG = CONFIG.camera.thermal_fov_h_deg
DEFAULT_THERMAL_FOV_V_DEG = CONFIG.camera.thermal_fov_v_deg

SHOW_TRACK_DETAILS = False
DRAW_HORIZON_LINE = False
TRACK_MIN_AGE_TO_DISPLAY = 3
TRACK_MIN_CONFIRMED_UPDATES = 2
TRACK_DRAW_MAX_STALE_FRAMES = 25

PANEL_HEIGHT = 158


def measure_text(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    """OpenCV yazısının piksel cinsinden genişlik ve yüksekliğini ölçer.

    Args:
        text: Ölçülecek metin.
        scale: OpenCV font ölçeği.
        thickness: Yazı kalınlığı.

    Returns:
        Metnin genişliği ve baseline dahil yüksekliği.
    """
    size, base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    return size[0], size[1] + base


def rects_overlap(rect_a: Rect, rect_b: Rect) -> bool:
    """İki dikdörtgenin çakışıp çakışmadığını kontrol eder.

    Args:
        rect_a: İlk dikdörtgen. Format: x1, y1, x2, y2.
        rect_b: İkinci dikdörtgen. Format: x1, y1, x2, y2.

    Returns:
        Dikdörtgenler çakışıyorsa True, aksi halde False.
    """
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def place_label(
    occupied: list[Rect],
    x_value: int,
    y_pref: int,
    width: int,
    height: int,
) -> int:
    """Etiket için ekranda mümkün olan en uygun y konumunu seçer.

    Aynı bölgede birden fazla track etiketi varsa yazılar üst üste binebilir.
    Bu fonksiyon önce tercih edilen y konumuna yakın adayları dener, çakışma
    yoksa o konumu kullanır.

    Args:
        occupied: Daha önce yerleştirilen etiket dikdörtgenleri.
        x_value: Etiketin sol başlangıç x koordinatı.
        y_pref: Tercih edilen y koordinatı.
        width: Etiket metninin genişliği.
        height: Etiket metninin yüksekliği.

    Returns:
        Etiketin çizileceği y koordinatı.
    """
    step = height + 8

    # Önce tercih edilen konumun üst tarafı denenir.
    candidates = [y_pref - i * step for i in range(6)]

    # Üst tarafta yer yoksa alt tarafa doğru alternatifler denenir.
    candidates += [y_pref + i * step for i in range(1, 6)]

    for y_value in candidates:
        # Üst panelin üzerine yazı bindirilmez.
        if y_value - height < PANEL_HEIGHT + 6:
            continue

        # Etiket görüntünün alt sınırının dışına taşmamalıdır.
        if y_value > PROCESS_HEIGHT - 4:
            continue

        rect = (
            x_value - 4,
            y_value - height - 4,
            x_value + width + 4,
            y_value + 4,
        )

        # Daha önce çizilen etiketlerle çakışmayan ilk konum seçilir.
        if not any(rects_overlap(rect, other) for other in occupied):
            occupied.append(rect)
            return y_value

    # Hiç uygun aday yoksa güvenli bir fallback konumu kullanılır.
    fallback = min(PROCESS_HEIGHT - 6, max(PANEL_HEIGHT + height, y_pref))
    occupied.append(
        (x_value - 4, fallback - height - 4, x_value + width + 4, fallback + 4)
    )

    return fallback


def draw_text_bg(
    frame: np.ndarray,
    text: str,
    org: Point,
    scale: float = 0.48,
    color: Color = (0, 255, 255),
    bg: Color = (0, 0, 0),
    thickness: int = 1,
) -> None:
    """Frame üzerine arka plan kutulu metin çizer.

    Args:
        frame: Üzerine çizim yapılacak BGR frame.
        text: Çizilecek metin.
        org: Metnin sol-alt başlangıç koordinatı.
        scale: OpenCV font ölçeği.
        color: Yazı rengi.
        bg: Arka plan kutu rengi.
        thickness: Yazı kalınlığı.
    """
    x_value, y_value = org

    # Metnin kaplayacağı alan hesaplanır.
    size, base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    width, height = size

    # Okunabilirlik için yazının arkasına dolu dikdörtgen çizilir.
    cv2.rectangle(
        frame,
        (x_value - 4, y_value - height - 6),
        (x_value + width + 4, y_value + base + 4),
        bg,
        -1,
    )

    # Metin, arka plan kutusunun üzerine çizilir.
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


def draw_horizon_line(frame: np.ndarray, horizon_state: HorizonState) -> None:
    """İstenirse tahmini ufuk çizgisini frame üzerine çizer.

    Args:
        frame: Üzerine çizim yapılacak BGR frame.
        horizon_state: Güncel ufuk çizgisi durumu.
    """
    if not DRAW_HORIZON_LINE:
        return

    # Ufuk çizgisi eğimli olabileceği için sol ve sağ uç ayrı hesaplanır.
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


def draw_tracks(
    frame: np.ndarray,
    tracks: TrackMap,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    video_fps: float,
) -> None:
    """Aktif track kutularını, su hattı noktalarını ve mesafe etiketlerini çizer.

    Args:
        frame: Üzerine çizim yapılacak BGR frame.
        tracks: Aktif track sözlüğü.
        sensor_info: Mevcut frame'e karşılık gelen sensör bilgisi.
        horizon_state: Güncel ufuk çizgisi durumu.
        video_fps: Video FPS değeri.
    """
    occupied: list[Rect] = []

    # Yakındaki/öndeki objelerin etiketleri daha kontrollü yerleşsin diye
    # track'ler su hattı y değerine göre sıralanır.
    ordered = sorted(tracks.items(), key=lambda item: item[1]["water_y"])

    for track_id, track in ordered:
        # Çok yeni track'ler ekrana çizilmez; böylece kısa süreli false positive
        # kutuların görsel çıktıya girmesi azaltılır.
        if track["age"] < TRACK_MIN_AGE_TO_DISPLAY:
            continue

        if track["confirmed_updates"] < TRACK_MIN_CONFIRMED_UPDATES:
            continue

        if track.get("frames_since_update", 0) > TRACK_DRAW_MAX_STALE_FRAMES:
            continue

        # Termal akışta daha sıkı çizim şartı uygulanır.
        if track.get("channel") == "thermal":
            if track.get("conf", 0.0) < THERMAL_YOLO_CONF_DEEP:
                continue

            if (
                track.get("confirmed_updates", 0)
                < TRACK_MIN_CONFIRMED_UPDATES + 1
            ):
                continue

        x1, y1, x2, y2 = visible_box(track["box"])

        # Geçersiz veya sıfır alanlı kutular çizilmez.
        if x2 <= x1 or y2 <= y1:
            continue

        result = calculate_track_distance(
            track, sensor_info, horizon_state, video_fps
        )

        water_x = int(round(track["water_x"]))
        water_y = int(round(track["water_y"]))

        # Su hattı noktası görüntü sınırları içinde tutulur.
        water_x = max(0, min(PROCESS_WIDTH - 1, water_x))
        water_y = max(0, min(PROCESS_HEIGHT - 1, water_y))

        at_horizon = (
            not result["valid"] and result["reason"] == "at_or_beyond_horizon"
        )

        # Geçerli mesafe varsa yeşil kutu çizilir.
        if result["valid"]:
            color = (0, 255, 0)
            distance_text = format_distance(result["distance"])

        # Ufuk çizgisinin ötesindeki hedefler için maksimum deniz mesafesi
        # üst sınır gibi gösterilir.
        elif at_horizon:
            color = (0, 200, 255)
            distance_text = f">{format_distance(MAX_SEA_DISTANCE_M)}"

        # Diğer geçersiz mesafe durumlarında bilinmeyen mesafe yazılır.
        else:
            color = (0, 200, 255)
            distance_text = "?"

        mode_text = "KLT" if track.get("klt_ok") else "PRED"
        label = f"id={track_id} | {distance_text} | {mode_text}"

        # Track bounding box çizimi.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Mesafe hesabında kullanılan su hattı noktası işaretlenir.
        cv2.drawMarker(
            frame,
            (water_x, water_y),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

        # Etiketin ekranda çakışmadan çizileceği y konumu seçilir.
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
    frame: np.ndarray,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    fps: float,
    video_second: float,
    track_count: int,
    mode: str,
    camera_moving: bool,
) -> None:
    """Üst bilgi panelini frame üzerine çizer.

    Args:
        frame: Üzerine panel çizilecek BGR frame.
        sensor_info: Mevcut frame'e ait sensör bilgisi.
        horizon_state: Güncel ufuk çizgisi durumu.
        fps: Anlık işlem FPS değeri.
        video_second: Video içindeki saniye değeri.
        track_count: Aktif track sayısı.
        mode: Detection/tracking çalışma modu.
        camera_moving: Kameranın hareket edip etmediği.
    """
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    # Sensör alanı yoksa panelde soru işareti gösterilir.
    zoom_text = (
        "?" if sensor_info["zoom"] is None else f"{sensor_info['zoom']:.4f}"
    )
    tilt_text = (
        "?" if sensor_info["tilt"] is None else f"{sensor_info['tilt']:.3f}"
    )

    moving_text = "MOVING" if camera_moving else "stable"
    bias_deg = math.degrees(horizon_state["pitch_bias_rad"])

    # Üst panel için siyah arka plan çizilir.
    cv2.rectangle(frame, (0, 0), (PROCESS_WIDTH, PANEL_HEIGHT), (0, 0, 0), -1)

    # Panelde gösterilecek satırlar tek listede tutulur.
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


def ensure_bgr_frame(frame: np.ndarray | None) -> np.ndarray | None:
    """Frame'i BGR formatına çevirir.

    Args:
        frame: Gri, tek kanallı veya BGR formatlı frame.

    Returns:
        BGR formatlı frame veya frame yoksa None.
    """
    if frame is None:
        return None

    # Gri görüntü iki boyutlu gelir; çizim için BGR'ye çevrilir.
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Tek kanallı üç boyutlu görüntü de BGR formatına çevrilir.
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


def draw_stream_output(
    frame: np.ndarray,
    stream_state: StreamState,
    sensor_rows: list[SensorRow],
    fps: float,
    frame_index: int,
    video_fps: float,
    camera_moving: bool,
) -> None:
    """Bir akışın tüm görsel çıktı katmanlarını frame üzerine çizer.

    Args:
        frame: Üzerine çizim yapılacak BGR frame.
        stream_state: Akışa ait güncel state sözlüğü.
        sensor_rows: CSV'den okunan sensör satırları.
        fps: Anlık işlem FPS değeri.
        frame_index: İşlenen frame'in sıra numarası.
        video_fps: Video FPS değeri.
        camera_moving: Kameranın hareket edip etmediği.
    """
    video_second = frame_index / video_fps
    sensor_info = stream_state["sensor_smooth"]

    # Henüz yumuşatılmış sensör değeri yoksa doğrudan CSV'den zaman karşılığı
    # sensör bilgisi alınır.
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

    # Akış adı üst panelin hemen altına yazılır.
    draw_text_bg(
        frame,
        stream_state["name"],
        (20, PANEL_HEIGHT + 28),
        scale=0.70,
        color=(255, 255, 255),
        bg=(0, 0, 0),
        thickness=2,
    )


def make_side_by_side(
    left_frame: np.ndarray | None,
    right_frame: np.ndarray | None,
) -> np.ndarray:
    """RGB ve termal frame'leri yan yana tek çıktı görüntüsünde birleştirir.

    Args:
        left_frame: Sol tarafta gösterilecek frame.
        right_frame: Sağ tarafta gösterilecek frame.

    Returns:
        Yan yana birleştirilmiş çıktı frame'i.
    """
    # Sol frame yoksa siyah placeholder görüntü kullanılır.
    if left_frame is None:
        left_frame = np.zeros(
            (PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8
        )

    # Sağ frame yoksa siyah placeholder görüntü kullanılır.
    if right_frame is None:
        right_frame = np.zeros(
            (PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8
        )

    # İki tarafın çözünürlüğü eşitlenir.
    left_frame = cv2.resize(left_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    right_frame = cv2.resize(right_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    return np.hstack((left_frame, right_frame))import math
from pathlib import Path

from config import AppConfig
from detector import THERMAL_YOLO_CONF_DEEP
from geometry import focal_from_fov
from sensor_reader import get_sensor_for_time


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)

RECORD_NAME = CONFIG.record.name
CAMERA_HEIGHT_M = CONFIG.camera.height_m

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_FACTOR = 7.0 / 6.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * REFRACTION_FACTOR
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

DEFAULT_FOV_H_DEG = CONFIG.camera.rgb_fov_h_deg
DEFAULT_FOV_V_DEG = CONFIG.camera.rgb_fov_v_deg
DEFAULT_THERMAL_FOV_H_DEG = CONFIG.camera.thermal_fov_h_deg
DEFAULT_THERMAL_FOV_V_DEG = CONFIG.camera.thermal_fov_v_deg

"""Visualization helpers for drawing tracks, labels, panels, and output frames."""


import cv2
from detector import box_to_int, visible_box
from geometry import (
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    format_distance,
    horizon_y_at,
)
import numpy as np
from tracker import calculate_track_distance


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
