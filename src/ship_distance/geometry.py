"""Geometry and distance estimation helpers."""

from collections import deque
import math
from statistics import median

import cv2
import numpy as np

"""
Bu dosya, görüntü üzerindeki piksel konumlarından yaklaşık deniz mesafesi
hesaplamak için kullanılan matematiksel yardımcı fonksiyonları içerir.

Mesafe tahmini doğrudan görüntüdeki nesne boyutuna göre yapılmaz. Bunun yerine
kamera yüksekliği, kamera görüş açısı, tilt bilgisi, horizon konumu ve nesnenin
görüntüdeki alt noktası birlikte değerlendirilir.

Temel işlem sırası:

1. Kameranın yatay ve dikey FOV değerlerinden focal length hesaplanır.
2. Görüntüdeki bir piksel satırı, optik eksene göre açıya çevrilir.
3. Tilt ve horizon bilgisi ile bakış ışınının deniz düzlemini nerede kestiği
   tahmin edilir.
4. Kamera yüksekliği kullanılarak bu kesişim noktasına olan yaklaşık mesafe
   hesaplanır.

Bu hesaplar gerçek dünyada dalga, kamera titreşimi, lens bozulması, atmosferik
etki ve sensor hatalarından etkilenebilir. Bu yüzden sonuçlar mutlak ölçüm
değil, yaklaşık mesafe tahmini olarak değerlendirilmelidir.
"""

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

CAMERA_HEIGHT_M = 10.0

DEFAULT_FOV_H_DEG = 65.7
DEFAULT_FOV_V_DEG = 39.9
TILT_ZERO_HORIZON_DEG = 90.0

EARTH_RADIUS_M = 6371000.0
REFRACTION_K = 0.13
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M / (1.0 - REFRACTION_K)

HORIZON_DIP_RAD = math.sqrt(2.0 * CAMERA_HEIGHT_M / EFFECTIVE_EARTH_RADIUS_M)
MAX_SEA_DISTANCE_M = math.sqrt(
    2.0 * EFFECTIVE_EARTH_RADIUS_M * CAMERA_HEIGHT_M
)

MIN_BETA_RAD = math.radians(0.015)
MIN_VALID_DISTANCE_M = 5.0

MANUAL_HORIZON_BIAS_DEG = 0.0

HORIZON_TILT_ONLY_EMA_STABLE = 0.12
HORIZON_TILT_ONLY_EMA_MOVING = 0.28
HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX = 1.2
HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX = 9.0

HORIZON_MEDIAN_WINDOW = 21
"""
FOV değerini piksel cinsinden focal length değerine çevirir.
Kamera kalibrasyonu doğrudan yoksa, görüntü genişliği/yüksekliği ve FOV
kullanılarak yaklaşık odak uzaklığı hesaplanabilir.
"""
def focal_from_fov(fov_h_deg, fov_v_deg):
    fx_value = (PROCESS_WIDTH / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    fy_value = (PROCESS_HEIGHT / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)

    return fx_value, fy_value
"""
Sensor tarafından gelen tilt bilgisini, kameranın aşağı bakış açısına
dönüştürür. Mesafe hesabında kameranın denize ne kadar aşağı baktığı
kritik olduğu için tilt değeri normalize edilir.
"""
def resolve_pitch_down_from_tilt(tilt_deg):
    if tilt_deg is None:
        return 0.0

    if 100.0 <= tilt_deg <= 180.0:
        return tilt_deg - 130.0

    if 45.0 <= tilt_deg <= 100.0:
        return TILT_ZERO_HORIZON_DEG - tilt_deg

    if -45.0 <= tilt_deg <= 45.0:
        return tilt_deg

    if -135.0 <= tilt_deg <= -45.0:
        return TILT_ZERO_HORIZON_DEG + tilt_deg

    return 0.0


def sea_distance_from_depression(alpha_rad):
    if alpha_rad <= HORIZON_DIP_RAD:
        return None

    tan_a = math.tan(alpha_rad)
    radius = EFFECTIVE_EARTH_RADIUS_M

    disc = (radius * tan_a) ** 2 - 2.0 * radius * CAMERA_HEIGHT_M

    if disc <= 0.0:
        return MAX_SEA_DISTANCE_M

    distance = radius * tan_a - math.sqrt(disc)

    return min(distance, MAX_SEA_DISTANCE_M)
"""
Görüntüdeki bir y piksel konumunu kameranın optik eksenine göre açıya
çevirir. Nesnenin alt noktası görüntüde ne kadar aşağıdaysa, kamera ışını
deniz düzlemini o kadar yakında keser.
"""
def pixel_row_to_angle(y_value, fy_value):
    return math.atan((y_value - CY) / fy_value)
"""
Tilt ve FOV bilgilerini kullanarak horizon çizgisinin görüntüde hangi
y koordinatına denk geleceğini tahmin eder.
Horizon çizgisi, deniz düzlemi mesafe hesabı için referans kabul edilir.
"""
def predict_horizon_y_from_tilt(sensor_info, pitch_bias_rad=0.0):
    _, fy_value = focal_from_fov(sensor_info["fov_h"], sensor_info["fov_v"])
    pitch_down = resolve_pitch_down_from_tilt(sensor_info.get("tilt"))
    angle = HORIZON_DIP_RAD - math.radians(pitch_down) + pitch_bias_rad
    angle = max(-1.2, min(1.2, angle))

    return CY + fy_value * math.tan(angle)


def create_horizon_state():
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


def clamp_horizon_y(y_value):
    return max(PROCESS_HEIGHT * 0.02, min(PROCESS_HEIGHT * 0.90, y_value))


def limit_horizon_step(current_y, target_y, max_step):
    diff = target_y - current_y

    if diff > max_step:
        return current_y + max_step

    if diff < -max_step:
        return current_y - max_step

    return target_y


def update_horizon(
    horizon_state, gray, sensor_info, frame_index, camera_moving
):
    target_y = clamp_horizon_y(
        predict_horizon_y_from_tilt(
            sensor_info, horizon_state["pitch_bias_rad"]
        )
    )

    if horizon_state["y"] is None:
        horizon_state["y"] = target_y
        horizon_state["slope"] = 0.0
        horizon_state["mode"] = "TILT_ONLY"
        return horizon_state

    if camera_moving:
        alpha = HORIZON_TILT_ONLY_EMA_MOVING
        max_step = HORIZON_TILT_ONLY_MAX_STEP_MOVING_PX
    else:
        alpha = HORIZON_TILT_ONLY_EMA_STABLE
        max_step = HORIZON_TILT_ONLY_MAX_STEP_STABLE_PX

    blended_y = (1.0 - alpha) * horizon_state["y"] + alpha * target_y
    horizon_state["y"] = clamp_horizon_y(
        limit_horizon_step(horizon_state["y"], blended_y, max_step)
    )
    horizon_state["slope"] = 0.0
    horizon_state["mode"] = "TILT_ONLY"
    horizon_state["visual_miss"] = 0
    horizon_state["flow_y"] = 0.0

    return horizon_state


def horizon_y_at(horizon_state, x_value):
    return horizon_state["y"] + horizon_state["slope"] * (x_value - CX)
"""
Görüntüde seçilen bir noktanın deniz yüzeyi üzerinde kameraya yaklaşık
uzaklığını hesaplar.
Bu fonksiyon; piksel konumu, horizon çizgisi, FOV, tilt ve kamera yüksekliği
bilgilerini birlikte kullanır.
"""
def sea_distance_from_image_point(
    pixel_x, pixel_y, sensor_info, horizon_state
):
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    y_horizon = horizon_y_at(horizon_state, pixel_x)
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

    if beta <= MIN_BETA_RAD:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

    alpha = HORIZON_DIP_RAD + beta
    distance = sea_distance_from_depression(alpha)

    if distance is None:
        return {
            **base,
            "valid": False,
            "reason": "at_or_beyond_horizon",
            "distance": None,
        }

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
"""
Mesafe değerini ekranda okunabilir hale getirir.
Kısa mesafeler metre, daha uzun mesafeler kilometre formatında gösterilir.
"""
def format_distance(distance_m):
    if distance_m is None:
        return "?"

    if distance_m >= 1000.0:
        return f"{distance_m / 1000.0:.2f} km"

    return f"{distance_m:.1f} m"
