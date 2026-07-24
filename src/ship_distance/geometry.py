"""Shared camera, horizon and sea-surface geometry helpers."""

from collections import deque
import math
from pathlib import Path
from typing import Any

from config import AppConfig
import numpy as np


# --- Shared image geometry ---------------------------------------------------
# Tracker, detector and visualizer operate in this normalized 1280x720 space.
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

# Geometry must use the same camera height as main.py and DistanceHlApi.
# Keeping one runtime source prevents horizon/range limits from being calculated
# with a stale hard-coded height when config.yaml changes between recordings.
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)
CAMERA_HEIGHT_M = CONFIG.camera.height_m

TILT_ZERO_HORIZON_DEG = 90.0

# Dünya eğriliği ve atmosferik kırılma, ufuk çizgisine yakın mesafe
# hesaplarında daha gerçekçi bir üst sınır oluşturmak için kullanılır.
# --- Earth/refraction model --------------------------------------------------
# REFRACTION_K increases the effective Earth radius to approximate atmospheric
# bending close to the horizon; both horizon dip and maximum range use it.
EARTH_RADIUS_M = 6371000.0
REFRACTION_K = 0.13
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M / (1.0 - REFRACTION_K)

# Kamera yüksekliğine göre teorik ufuk çöküşü ve maksimum deniz görüş mesafesi
# hesaplanır. Bu değerler ufka çok yakın noktaların filtrelenmesinde kullanılır.
HORIZON_DIP_RAD = math.sqrt(2.0 * CAMERA_HEIGHT_M / EFFECTIVE_EARTH_RADIUS_M)
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

# Ufuk çizgisine çok yakın açılar mesafe hesabında kararsız sonuç üretir.
# Bu yüzden minimum beta açısı ve minimum geçerli mesafe sınırı kullanılır.
# --- Validity and horizon smoothing -----------------------------------------
# Very small depression angles amplify pixel-level error into large range error.
MIN_BETA_RAD = math.radians(0.015)
MIN_VALID_DISTANCE_M = 5.0

MANUAL_HORIZON_BIAS_DEG = 0.0

# Kamera sabitken ufuk çizgisi daha yavaş; kamera hareketliyken daha hızlı
# güncellenir. Böylece hem titreşim azaltılır hem de hareket sonrası uyum korunur.
HORIZON_TILT_ONLY_EMA_STABLE = 0.12
HORIZON_TILT_ONLY_EMA_MOVING = 0.28
HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX = 1.2
HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX = 9.0

HORIZON_MEDIAN_WINDOW = 21


# --- Camera projection helpers ----------------------------------------------
def focal_from_fov(fov_h_deg: float, fov_v_deg: float) -> tuple[float, float]:
    """Convert horizontal/vertical FOV to focal length in pixels."""
    # Pinhole kamera modeline göre focal length, görüntü boyutunun yarısının
    # yarım FOV açısının tanjantına bölünmesiyle hesaplanır.
    # Pinhole projection: f_px = half_image_size / tan(half_FOV).
    fx_value = (PROCESS_WIDTH / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    fy_value = (PROCESS_HEIGHT / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)

    return fx_value, fy_value


def resolve_pitch_down_from_tilt(tilt_deg: float | None) -> float:
    """Normalize PTZ tilt conventions to a downward pitch angle."""
    if tilt_deg is None:
        return 0.0

    # Bazı sistemlerde tilt 100-180 aralığında raporlanır. Bu durumda değer
    # yaklaşık merkez referansa göre yeniden ölçeklenir.
    if 100.0 <= tilt_deg <= 180.0:
        return tilt_deg - 130.0

    # 45-100 aralığında tilt değeri ufuk referansına göre yorumlanır.
    if 45.0 <= tilt_deg <= 100.0:
        return TILT_ZERO_HORIZON_DEG - tilt_deg

    # -45 ile 45 aralığı doğrudan pitch benzeri kabul edilir.
    if -45.0 <= tilt_deg <= 45.0:
        return tilt_deg

    # Negatif geniş aralıklar tekrar ufuk referansına göre çevrilir.
    if -135.0 <= tilt_deg <= -45.0:
        return TILT_ZERO_HORIZON_DEG + tilt_deg

    return 0.0


# --- Sea-surface range geometry ---------------------------------------------
def sea_distance_from_depression(alpha_rad: float) -> float | None:
    """Estimate sea-surface range from a downward viewing angle."""
    # Açı ufuk çöküşünden küçükse ışın deniz yüzeyini güvenilir şekilde kesmez.
    # A ray above/equal to the refracted horizon does not intersect the modeled sea.
    if alpha_rad <= HORIZON_DIP_RAD:
        return None

    tan_a = math.tan(alpha_rad)
    radius = EFFECTIVE_EARTH_RADIUS_M

    # Dünya eğriliği hesaba katılarak ikinci derece denklem çözülür.
    # Solve the ray/curved-Earth intersection. Camera height comes from config.
    disc = (radius * tan_a) ** 2 - 2.0 * radius * CAMERA_HEIGHT_M

    if disc <= 0.0:
        return MAX_SEA_DISTANCE_M

    distance = radius * tan_a - math.sqrt(disc)

    return min(distance, MAX_SEA_DISTANCE_M)


def pixel_row_to_angle(y_value: float, fy_value: float) -> float:
    """Convert an image row to vertical angle relative to the optical center."""
    return math.atan((y_value - CY) / fy_value)


def predict_horizon_y_from_tilt(
    sensor_info: dict[str, Any], pitch_bias_rad: float = 0.0
) -> float:
    """Predict horizon y-position from tilt, FOV and pitch bias."""
    _, fy_value = focal_from_fov(sensor_info["fov_h"], sensor_info["fov_v"])
    pitch_down = resolve_pitch_down_from_tilt(sensor_info.get("tilt"))

    # Ufuk çöküşü, kamera pitch değeri ve manuel bias birlikte kullanılarak
    # görüntüdeki ufuk açısı hesaplanır.
    angle = HORIZON_DIP_RAD - math.radians(pitch_down) + pitch_bias_rad
    angle = max(-1.2, min(1.2, angle))

    return CY + fy_value * math.tan(angle)


# --- Horizon state -----------------------------------------------------------
# Current implementation intentionally uses tilt/FOV rather than visual fitting.
def create_horizon_state() -> dict[str, Any]:
    """Create the persistent state used by tilt-based horizon tracking."""
    return {
        "y": None,
        "slope": 0.0,
        "mode": "TILT_ONLY",
        "visual_miss": 0,
        "pitch_bias_rad": math.radians(MANUAL_HORIZON_BIAS_DEG),
        "visual_history": deque(maxlen=HORIZON_MEDIAN_WINDOW),
        "slope_history": deque(maxlen=HORIZON_MEDIAN_WINDOW),
        "flow_y": 0.0,
    }


def clamp_horizon_y(y_value: float) -> float:
    """Keep the horizon inside the usable vertical image region."""
    # Ufuk çizgisinin görüntünün tamamen dışına veya en altına kayması mesafe
    # hesabını bozacağı için değer güvenli aralıkta sınırlandırılır.
    return max(PROCESS_HEIGHT * 0.02, min(PROCESS_HEIGHT * 0.90, y_value))


def limit_horizon_step(
    current_y: float, target_y: float, max_step: float
) -> float:
    """Limit the horizon's maximum per-frame vertical movement."""
    diff = target_y - current_y

    if diff > max_step:
        return current_y + max_step

    if diff < -max_step:
        return current_y - max_step

    return target_y


def update_horizon(
    horizon_state: dict[str, Any],
    gray: np.ndarray,
    sensor_info: dict[str, Any],
    frame_index: int,
    camera_moving: bool,
) -> dict[str, Any]:
    """Update the tilt/FOV horizon estimate with motion-aware smoothing."""
    # Bu parametreler mevcut fonksiyon imzasını korumak için bırakılmıştır.
    # Şu an ufuk güncellemesi doğrudan tilt/FOV tabanlı çalışmaktadır.
    _ = gray, frame_index

    # Sensor-derived target is clamped before temporal smoothing.
    target_y = clamp_horizon_y(
        predict_horizon_y_from_tilt(
            sensor_info, horizon_state["pitch_bias_rad"]
        )
    )

    # İlk frame'de geçmiş değer olmadığı için ufuk doğrudan hedef değere atanır.
    if horizon_state["y"] is None:
        horizon_state["y"] = target_y
        horizon_state["slope"] = 0.0
        horizon_state["mode"] = "TILT_ONLY"
        return horizon_state

    # Kamera hareketliyken ufuk daha hızlı güncellenir. Kamera sabitken daha
    # düşük EMA katsayısı kullanılarak titreşim ve küçük sensör oynamaları
    # azaltılır.
    if camera_moving:
        alpha = HORIZON_TILT_ONLY_EMA_MOVING
        max_step = HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX
    else:
        alpha = HORIZON_TILT_ONLY_EMA_STABLE
        max_step = HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX

    # EMA removes small tilt jitter; step limiting separately blocks sudden jumps.
    blended_y = (1.0 - alpha) * horizon_state["y"] + alpha * target_y
    horizon_state["y"] = clamp_horizon_y(
        limit_horizon_step(horizon_state["y"], blended_y, max_step)
    )

    # Bu akışta ufuk eğimi kullanılmadığı için slope sıfırda tutulur.
    horizon_state["slope"] = 0.0
    horizon_state["mode"] = "TILT_ONLY"
    horizon_state["visual_miss"] = 0
    horizon_state["flow_y"] = 0.0

    return horizon_state


def horizon_y_at(horizon_state: dict[str, Any], x_value: float) -> float:
    """Return horizon y at a requested x-position."""
    return horizon_state["y"] + horizon_state["slope"] * (x_value - CX)


# --- Image point -> sea range ------------------------------------------------
# This helper is still used by detector-side near/own-ship filtering.
def sea_distance_from_image_point(
    pixel_x: float,
    pixel_y: float,
    sensor_info: dict[str, Any],
    horizon_state: dict[str, Any],
) -> dict[str, Any]:
    """Estimate sea-surface range and components for one image point."""
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    y_horizon = horizon_y_at(horizon_state, pixel_x)

    # Beta, seçilen pikselin ufuk çizgisine göre aşağı yönde yaptığı açıdır.
    # Nesne ufka çok yakınsa beta küçük olur ve mesafe hesabı kararsızlaşır.
    # beta is positive below the horizon and approaches zero near the horizon.
    beta = math.atan((pixel_y - y_horizon) / fy_value)

    base = {
        "raw_distance": None,
        "forward": None,
        "lateral": None,
        "beta_deg": math.degrees(beta),
        "fx": fx_value,
        "fy": fy_value,
        "horizon_y": y_horizon,
    }

    # Ufuk çizgisi üzerindeki veya ufka çok yakın noktalar güvenilir değildir.
    if beta <= MIN_BETA_RAD:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

    # Toplam aşağı bakış açısı, teorik ufuk çöküşü ile pikselin ufka göre
    # ekstra aşağı açısının toplamı olarak alınır.
    # Total depression combines geometric horizon dip with pixel depression.
    alpha = HORIZON_DIP_RAD + beta
    distance = sea_distance_from_depression(alpha)

    if distance is None:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

    # X konumu yatay açıya çevrilir. Böylece toplam mesafe ileri ve yanal
    # bileşenlere ayrılabilir.
    # Horizontal bearing separates slant sea range into forward/lateral components.
    phi = math.atan((pixel_x - CX) / fx_value)
    forward_m = distance * math.cos(phi)
    lateral_m = distance * math.sin(phi)

    if distance < MIN_VALID_DISTANCE_M:
        return {
            **base,
            "valid": False,
            "reason": "distance_out_of_range",
            "distance": None,
            "raw_distance": distance,
            "forward": forward_m,
            "lateral": lateral_m,
        }

    return {
        **base,
        "valid": True,
        "reason": "OK",
        "distance": distance,
        "raw_distance": distance,
        "forward": forward_m,
        "lateral": lateral_m,
    }


def format_distance(distance_m: float | None) -> str:
    """Format a distance for the on-frame display."""
    if distance_m is None:
        return "?"

    if distance_m >= 1000.0:
        return f"{distance_m / 1000.0:.2f} km"

    return f"{distance_m:.1f} m"
