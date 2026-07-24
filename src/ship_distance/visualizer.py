"""RGB/termal çıktıların track, mesafe, ufuk ve bilgi paneli çizimleri."""

import math
from pathlib import Path
from typing import TypeAlias

from config import AppConfig
import cv2
from detector import THERMAL_YOLO_CONF_DEEP, visible_box
from geometry import (
    PROCESS_HEIGHT,
    PROCESS_WIDTH,
    focal_from_fov,
    format_distance,
    horizon_y_at,
)
import numpy as np
from sensor_reader import SensorRow, get_sensor_for_time


Rect: TypeAlias = tuple[int, int, int, int]
Point: TypeAlias = tuple[int, int]
Color: TypeAlias = tuple[int, int, int]
Track: TypeAlias = dict[str, object]
TrackMap: TypeAlias = dict[int, Track]
HorizonState: TypeAlias = dict[str, object]
StreamState: TypeAlias = dict[str, object]

# Runtime display metadata comes from the same project configuration as main.
# Keep configuration reads here limited to values that are actually rendered.
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)

RECORD_NAME = CONFIG.record.name
CAMERA_HEIGHT_M = CONFIG.camera.height_m

# Display-only horizon limit. This does not perform per-track distance estimation.
# The value is used when a target is reported at/beyond the visible horizon.
EARTH_RADIUS_M = 6_371_000.0
REFRACTION_FACTOR = 7.0 / 6.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * REFRACTION_FACTOR
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

# Visualization feature flags and minimum track quality required for drawing.
# They affect only what is rendered, not tracking or distance calculations.
SHOW_TRACK_DETAILS = False
DRAW_HORIZON_LINE = False
TRACK_MIN_AGE_TO_DISPLAY = 3
TRACK_MIN_CONFIRMED_UPDATES = 2
TRACK_DRAW_MAX_STALE_FRAMES = 25

PANEL_HEIGHT = 158


def get_display_distance_result(track: Track) -> dict[str, object]:
    """Return the cached tracker result in the exact form needed for drawing."""

    # Distance calculation belongs to tracker.py.  The visualizer only consumes
    # the already-computed result so drawing a frame can never trigger estimation.
    cached = track.get("last_result")

    if not isinstance(cached, dict):
        return {
            "valid": False,
            "reason": "distance_not_calculated",
            "distance": None,
            "raw_distance": None,
            "distance_source": "none",
        }

    # The thermal guard may adjust track["distance"] after tracker cached the raw
    # result.  Copy before overriding so visualization never mutates tracker state.
    result = cached.copy()
    current_distance = track.get("distance")

    if result.get("valid") and isinstance(current_distance, int | float):
        result["distance"] = float(current_distance)

    return result


# --- Label layout helpers ----------------------------------------------------
def measure_text(
    text: str, scale: float, thickness: int = 1
) -> tuple[int, int]:
    """OpenCV metninin baseline dahil piksel boyutunu döndürür."""
    size, base = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    return size[0], size[1] + base


def rects_overlap(rect_a: Rect, rect_b: Rect) -> bool:
    """İki ekran dikdörtgeninin çakışıp çakışmadığını döndürür."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def place_label(
    occupied: list[Rect], x_value: int, y_pref: int, width: int, height: int
) -> int:
    """Çakışmayan bir etiket y-konumu seçer; uygun yer yoksa fallback kullanır."""
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
    """Okunabilirlik için dolu arka plan üzerinde metin çizer."""
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


# --- Overlay drawing ---------------------------------------------------------
def draw_horizon_line(frame: np.ndarray, horizon_state: HorizonState) -> None:
    """Debug seçeneği açıksa tahmini ufuk çizgisini çizer."""
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


# Track rendering deliberately consumes tracker state instead of recalculating it.
# This keeps visualization side-effect free with respect to distance estimation.
def draw_tracks(
    frame: np.ndarray,
    tracks: TrackMap,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    video_fps: float,
) -> None:
    """Güvenilir track'leri bbox, water-point ve mesafe etiketiyle çizer."""
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
        # Thermal detections require one extra confirmation step before display.
        # This reduces flickering from short-lived low-confidence thermal fragments.
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

        # Distance was already calculated by tracker; drawing only reads it.
        result = get_display_distance_result(track)

        water_x = int(round(track["water_x"]))
        water_y = int(round(track["water_y"]))

        # Su hattı noktası görüntü sınırları içinde tutulur.
        water_x = max(0, min(PROCESS_WIDTH - 1, water_x))
        water_y = max(0, min(PROCESS_HEIGHT - 1, water_y))

        # "at_or_beyond_horizon" is visually distinct from a generic invalid result:
        # the target may exist, but geometry cannot return a finite sea-surface range.
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
                    f"distance_src={result.get('distance_source', 'unknown')} | "
                    f"det_src={track['source']}"
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


# --- Status panel ------------------------------------------------------------
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
    """Sensör, horizon, FPS ve tracking durumunu üst panelde gösterir."""
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
            f"HL + BUTTERFLY DISTANCE | FPS={fps:.1f} | "
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
            "HL geometry + bbox/FOV estimate | temporal tracking"
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


# --- Frame normalization and stream composition ------------------------------
def ensure_bgr_frame(frame: np.ndarray | None) -> np.ndarray | None:
    """Gri veya tek kanallı görüntüyü çizime uygun BGR biçimine çevirir."""
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
    """Bir RGB/termal akış için tüm çizim katmanlarını sırayla uygular."""
    video_second = frame_index / video_fps
    sensor_info = stream_state["sensor_smooth"]

    # Henüz yumuşatılmış sensör değeri yoksa doğrudan CSV'den zaman karşılığı
    # sensör bilgisi alınır.
    # A missing smoothed sensor value can occur before normal stream state is ready;
    # use the timestamp-matched raw row only as a rendering fallback.
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
    left_frame: np.ndarray | None, right_frame: np.ndarray | None
) -> np.ndarray:
    """RGB ve termal frame'leri aynı boyutta yan yana birleştirir."""
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

    return np.hstack((left_frame, right_frame))
