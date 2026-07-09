"""Kamera geometrisi, ufuk çizgisi ve deniz mesafesi hesaplama yardımcıları.

Bu dosya, görüntü üzerindeki piksel konumlarından yaklaşık deniz mesafesi
hesaplamak için kullanılan matematiksel yardımcı fonksiyonları içerir.

İlk mesafe yöntemi, kamera yüksekliği, FOV, tilt bilgisi, ufuk çizgisi ve
nesnenin görüntüdeki su hattı noktası üzerinden çalışır. Bu horizon/waterline
yaklaşımı özellikle bbox su hattı hatalıysa çok hassas sonuç üretebilir.

Bu yüzden ek olarak bbox piksel boyutu, FOV/zoom bilgisi ve geminin görüntüde
kapladığı yaklaşık fiziksel boyut aralığı kullanılarak ikinci bir mesafe
tahmini üretilir. Son aşamada horizon tabanlı mesafe ile bbox-size tabanlı
mesafe güven skorlarına göre birleştirilir.
"""

from collections import deque
import math
from typing import Any, TypeAlias

import numpy as np


Box: TypeAlias = tuple[float, float, float, float]

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

CX = PROCESS_WIDTH / 2.0
CY = PROCESS_HEIGHT / 2.0

CAMERA_HEIGHT_M = 10.0

DEFAULT_FOV_H_DEG = 65.7
DEFAULT_FOV_V_DEG = 39.9
TILT_ZERO_HORIZON_DEG = 90.0

# Dünya eğriliği ve atmosferik kırılma, ufuk çizgisine yakın mesafe
# hesaplarında daha gerçekçi bir üst sınır oluşturmak için kullanılır.
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

# Bbox-size mesafe hesabında gemi tipi kesin seçilmez. Bunun yerine büyük/orta
# deniz aracı için makul görünür uzunluk ve görünür yükseklik aralıkları
# kullanılır. Bu sayede kameraya önden bakan büyük gemi, otomatik olarak küçük
# tekne sayılmaz.
SHIP_VISIBLE_LENGTH_M = 70.0
SHIP_VISIBLE_LENGTH_MIN_M = 35.0
SHIP_VISIBLE_LENGTH_MAX_M = 180.0

SHIP_VISIBLE_HEIGHT_M = 18.0
SHIP_VISIBLE_HEIGHT_MIN_M = 8.0
SHIP_VISIBLE_HEIGHT_MAX_M = 45.0

# Dar FOV/zoom durumunda horizon-only hesap birkaç piksel hatadan çok etkilenir.
# Bu nedenle bbox-size sonucu, yeterince güvenilir olduğunda daha yüksek ağırlık
# alır.
SIZE_DISTANCE_MIN_VALID_M = 50.0
SIZE_DISTANCE_MAX_VALID_M = 30000.0

HORIZON_WEIGHT_DEFAULT = 0.35
SIZE_WEIGHT_DEFAULT = 0.65
DISAGREEMENT_RATIO_THRES = 2.2


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


def safe_float(value: object, default: float) -> float:
    """Herhangi bir değeri güvenli şekilde float'a çevirir.

    Args:
        value: Float'a çevrilecek ham değer.
        default: Dönüşüm başarısız olursa kullanılacak değer.

    Returns:
        Float değer.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def focal_from_fov(fov_h_deg: float, fov_v_deg: float) -> tuple[float, float]:
    """FOV değerlerinden piksel cinsinden focal length hesaplar.

    Kamera kalibrasyonu doğrudan yoksa, görüntü genişliği/yüksekliği ve FOV
    değerleri kullanılarak yaklaşık odak uzaklığı hesaplanabilir. Bu değerler
    daha sonra piksel konumlarını açısal değerlere çevirmek için kullanılır.

    Args:
        fov_h_deg: Kameranın yatay görüş açısı.
        fov_v_deg: Kameranın dikey görüş açısı.

    Returns:
        Yatay ve dikey focal length değerleri.
    """
    # Pinhole kamera modeline göre focal length, görüntü boyutunun yarısının
    # yarım FOV açısının tanjantına bölünmesiyle hesaplanır.
    fx_value = (PROCESS_WIDTH / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
    fy_value = (PROCESS_HEIGHT / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)

    return fx_value, fy_value


def resolve_pitch_down_from_tilt(tilt_deg: float | None) -> float:
    """Sensörden gelen tilt bilgisini kameranın aşağı bakış açısına çevirir.

    Farklı PTZ sistemleri tilt değerini farklı aralıklarda raporlayabilir.
    Bu fonksiyon, gelen tilt değerini mesafe hesabında kullanılabilecek
    normalize edilmiş aşağı bakış açısına dönüştürür.

    Args:
        tilt_deg: Sensörden okunan tilt değeri. Değer yoksa None olabilir.

    Returns:
        Kameranın yaklaşık aşağı bakış açısı.
    """
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


def sea_distance_from_depression(alpha_rad: float) -> float | None:
    """Aşağı bakış açısından deniz yüzeyi üzerindeki mesafeyi hesaplar.

    Alpha açısı, kamera ışınının ufuk referansına göre deniz yüzeyine doğru
    yaptığı toplam aşağı bakış açısıdır. Ufuk çizgisine eşit veya daha küçük
    açılar fiziksel olarak güvenilir mesafe üretmez.

    Args:
        alpha_rad: Radyan cinsinden toplam aşağı bakış açısı.

    Returns:
        Metre cinsinden yaklaşık deniz mesafesi. Açı ufukta veya ufkun
        ötesindeyse None döner.
    """
    # Açı ufuk çöküşünden küçükse ışın deniz yüzeyini güvenilir şekilde kesmez.
    if alpha_rad <= HORIZON_DIP_RAD:
        return None

    tan_a = math.tan(alpha_rad)
    radius = EFFECTIVE_EARTH_RADIUS_M

    # Dünya eğriliği hesaba katılarak ikinci derece denklem çözülür.
    disc = (radius * tan_a) ** 2 - 2.0 * radius * CAMERA_HEIGHT_M

    if disc <= 0.0:
        return MAX_SEA_DISTANCE_M

    distance = radius * tan_a - math.sqrt(disc)

    return min(distance, MAX_SEA_DISTANCE_M)


def pixel_row_to_angle(y_value: float, fy_value: float) -> float:
    """Görüntüdeki y piksel konumunu dikey açıya çevirir.

    Piksel koordinatı doğrudan metre bilgisi vermez. Bu yüzden piksel satırı,
    kameranın dikey focal length değeriyle optik eksene göre açıya çevrilir.

    Args:
        y_value: Görüntüdeki y piksel konumu.
        fy_value: Dikey focal length değeri.

    Returns:
        Radyan cinsinden dikey açı.
    """
    return math.atan((y_value - CY) / fy_value)


def predict_horizon_y_from_tilt(
    sensor_info: dict[str, Any], pitch_bias_rad: float = 0.0
) -> float:
    """Tilt ve FOV bilgileriyle ufuk çizgisinin y konumunu tahmin eder.

    Ufuk çizgisi, deniz düzlemi üzerinde mesafe hesabı için temel referanstır.
    Kamera aşağı baktıkça ufuk çizgisi görüntü içinde yukarı veya aşağı kayar.
    Bu kayma FOV ve tilt bilgisiyle yaklaşık olarak hesaplanır.

    Args:
        sensor_info: FOV ve tilt bilgilerini içeren sensör sözlüğü.
        pitch_bias_rad: Manuel veya geçmişten gelen pitch düzeltmesi.

    Returns:
        Ufuk çizgisinin görüntüdeki yaklaşık y piksel konumu.
    """
    _, fy_value = focal_from_fov(sensor_info["fov_h"], sensor_info["fov_v"])
    pitch_down = resolve_pitch_down_from_tilt(sensor_info.get("tilt"))

    # Ufuk çöküşü, kamera pitch değeri ve manuel bias birlikte kullanılarak
    # görüntüdeki ufuk açısı hesaplanır.
    angle = HORIZON_DIP_RAD - math.radians(pitch_down) + pitch_bias_rad
    angle = max(-1.2, min(1.2, angle))

    return CY + fy_value * math.tan(angle)


def create_horizon_state() -> dict[str, Any]:
    """Ufuk çizgisi takibi için başlangıç durumunu oluşturur.

    Ufuk çizgisi tek karelik bir değer olarak tutulmaz. Önceki y konumu, eğim,
    geçmiş değerler ve pitch bias bilgisi state içinde saklanır. Böylece
    frame'ler arasında daha stabil bir ufuk referansı elde edilir.

    Returns:
        Ufuk çizgisi takibi için kullanılacak state sözlüğü.
    """
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
    """Ufuk çizgisinin görüntü içinde makul bir aralıkta kalmasını sağlar.

    Args:
        y_value: Hesaplanan ufuk çizgisi y konumu.

    Returns:
        Görüntü sınırları içinde kısıtlanmış y konumu.
    """
    # Ufuk çizgisinin görüntünün tamamen dışına veya en altına kayması mesafe
    # hesabını bozacağı için değer güvenli aralıkta sınırlandırılır.
    return max(PROCESS_HEIGHT * 0.02, min(PROCESS_HEIGHT * 0.90, y_value))


def limit_horizon_step(
    current_y: float, target_y: float, max_step: float
) -> float:
    """Ufuk çizgisinin tek karede yapabileceği maksimum değişimi sınırlar.

    Args:
        current_y: Mevcut ufuk çizgisi y konumu.
        target_y: Yeni hedef ufuk çizgisi y konumu.
        max_step: Tek karede izin verilen maksimum piksel değişimi.

    Returns:
        Sınırlandırılmış yeni ufuk çizgisi y konumu.
    """
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
    """Ufuk çizgisi durumunu günceller.

    Bu sürümde ufuk çizgisi temel olarak tilt/FOV bilgisine göre güncellenir.
    Kamera hareketliyken daha hızlı, sabitken daha yavaş yumuşatma uygulanır.
    Böylece ufuk çizgisi hem ani sıçramalardan korunur hem de kamera hareketine
    uyum sağlayabilir.

    Args:
        horizon_state: Önceki ufuk çizgisi bilgilerini tutan state sözlüğü.
        gray: Mevcut gri frame. Bu akışta imza uyumluluğu için korunur.
        sensor_info: Mevcut frame'e ait sensör bilgisi.
        frame_index: İşlenen frame index değeri. İmza uyumluluğu için korunur.
        camera_moving: Kameranın pan/zoom hareketinde olup olmadığı.

    Returns:
        Güncellenmiş ufuk state sözlüğü.
    """
    # Bu parametreler mevcut fonksiyon imzasını korumak için bırakılmıştır.
    # Şu an ufuk güncellemesi doğrudan tilt/FOV tabanlı çalışmaktadır.
    _ = gray, frame_index

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
    """Belirli bir x konumunda ufuk çizgisinin y değerini hesaplar.

    Args:
        horizon_state: Ufuk çizgisi y konumu ve eğimini içeren state.
        x_value: Ufuk çizgisinin hesaplanacağı x piksel konumu.

    Returns:
        Verilen x konumundaki ufuk y değeri.
    """
    return horizon_state["y"] + horizon_state["slope"] * (x_value - CX)


def sea_distance_from_image_point(
    pixel_x: float,
    pixel_y: float,
    sensor_info: dict[str, Any],
    horizon_state: dict[str, Any],
) -> dict[str, Any]:
    """Görüntüdeki bir noktanın deniz yüzeyi üzerindeki mesafesini hesaplar.

    Mesafe hesabı için noktanın ufuk çizgisine göre dikey açısı bulunur.
    Daha sonra bu açı, kamera yüksekliği ve Dünya eğriliği modeliyle birlikte
    kullanılarak yaklaşık ileri mesafe ve yanal mesafe hesaplanır.

    Args:
        pixel_x: Noktanın x piksel konumu.
        pixel_y: Noktanın y piksel konumu.
        sensor_info: FOV ve tilt bilgilerini içeren sensör sözlüğü.
        horizon_state: Güncel ufuk çizgisi state bilgisi.

    Returns:
        Mesafe sonucunu, geçerlilik bilgisini, ileri/yanal bileşenleri ve
        hesaplama yardımcı değerlerini içeren sözlük.
    """
    fx_value, fy_value = focal_from_fov(
        sensor_info["fov_h"], sensor_info["fov_v"]
    )

    y_horizon = horizon_y_at(horizon_state, pixel_x)

    # Beta, seçilen pikselin ufuk çizgisine göre aşağı yönde yaptığı açıdır.
    # Nesne ufka çok yakınsa beta küçük olur ve mesafe hesabı kararsızlaşır.
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


def orientation_scores_from_box(box: Box) -> dict[str, float]:
    """Bbox şekline göre yan/ön görünüm güven skorlarını hesaplar.

    Bu fonksiyon gemi tipini kesin sınıflandırmaz. Sadece bbox oranından
    width-based mesafe hesabının ne kadar güvenilir olabileceğini çıkarır.

    Args:
        box: Detection veya track bounding box değeri.

    Returns:
        Yan görünüm, ön/diyagonal görünüm ve genel bbox güven skorları.
    """
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / height
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)

    # Uzun yatay kutular geminin yandan göründüğüne dair daha güçlü sinyal verir.
    side_score = clamp((aspect - 1.2) / 2.8, 0.0, 1.0)

    # Kompakt kutular önden/diyagonal görünüm olabilir. Bu durumda gemi uzunluğu
    # görüntüde tam görünmeyebilir ve width-based mesafe daha az güvenilir olur.
    bow_score = clamp((1.8 - aspect) / 1.2, 0.0, 1.0)

    # Çok küçük, aşırı büyük veya görüntü kenarına yapışık kutuların boyut
    # bilgisinden mesafe çıkarmak daha risklidir.
    size_score = clamp((area_ratio - 0.00012) / 0.014, 0.0, 1.0)

    edge_penalty = 0.0

    if x1 <= 2.0 or y1 <= 2.0:
        edge_penalty += 0.25

    if x2 >= PROCESS_WIDTH - 3.0 or y2 >= PROCESS_HEIGHT - 3.0:
        edge_penalty += 0.25

    bbox_score = clamp(0.30 + 0.70 * size_score - edge_penalty, 0.05, 1.0)

    return {
        "aspect": aspect,
        "side_score": side_score,
        "bow_score": bow_score,
        "bbox_score": bbox_score,
        "area_ratio": area_ratio,
    }


def estimate_distance_from_bbox_size(
    box: Box, sensor_info: dict[str, Any]
) -> dict[str, Any]:
    """BBox piksel boyutu ve FOV değerlerinden mesafe tahmini yapar.

    Burada gemi tipi kesin olarak seçilmez. Bunun yerine görünür gemi uzunluğu
    ve görünür gemi yüksekliği için makul aralıklar kullanılır. Yan görünüm
    güvenliyse width-based mesafe daha fazla ağırlık alır; kompakt/önden görünüm
    varsa height-based mesafe daha fazla korunur.

    Args:
        box: Detection veya track bounding box değeri.
        sensor_info: FOV/zoom bilgilerini içeren sensör sözlüğü.

    Returns:
        Bbox-size tabanlı mesafe tahmini ve güven yardımcı değerleri.
    """
    x1, y1, x2, y2 = box

    width_px = max(1.0, x2 - x1)
    height_px = max(1.0, y2 - y1)

    fov_h = safe_float(sensor_info.get("fov_h"), DEFAULT_FOV_H_DEG)
    fov_v = safe_float(sensor_info.get("fov_v"), DEFAULT_FOV_V_DEG)
    fx_value, fy_value = focal_from_fov(fov_h, fov_v)

    scores = orientation_scores_from_box(box)

    # Yan görünümde görünen genişlik geminin gerçek uzunluğuna daha yakındır.
    # Önden/diyagonal görünümde ise gemi uzunluğu görüntüye yansımaz; bu yüzden
    # width-based ağırlık düşürülür.
    width_confidence = clamp(0.10 + 0.75 * scores["side_score"], 0.10, 0.85)

    # Height-based hesap tek başına kesin değildir ama önden/diyagonal görünümde
    # width-based hesaba göre daha stabil olabilir.
    height_confidence = clamp(0.25 + 0.35 * scores["bow_score"], 0.20, 0.60)

    width_distance = SHIP_VISIBLE_LENGTH_M * fx_value / width_px
    width_min = SHIP_VISIBLE_LENGTH_MIN_M * fx_value / width_px
    width_max = SHIP_VISIBLE_LENGTH_MAX_M * fx_value / width_px

    height_distance = SHIP_VISIBLE_HEIGHT_M * fy_value / height_px
    height_min = SHIP_VISIBLE_HEIGHT_MIN_M * fy_value / height_px
    height_max = SHIP_VISIBLE_HEIGHT_MAX_M * fy_value / height_px

    total_weight = width_confidence + height_confidence

    if total_weight <= 0.0:
        return {
            "valid": False,
            "reason": "bbox_size_weight_zero",
            "distance": None,
            "confidence": 0.0,
        }

    size_distance = (
        width_distance * width_confidence + height_distance * height_confidence
    ) / total_weight

    min_distance = min(width_min, height_min)
    max_distance = max(width_max, height_max)

    confidence = clamp(
        scores["bbox_score"]
        * (0.35 + 0.65 * max(width_confidence, height_confidence)),
        0.05,
        0.95,
    )

    if (
        not SIZE_DISTANCE_MIN_VALID_M
        <= size_distance
        <= SIZE_DISTANCE_MAX_VALID_M
    ):
        return {
            "valid": False,
            "reason": "bbox_size_distance_out_of_range",
            "distance": None,
            "confidence": confidence,
            "width_distance": width_distance,
            "height_distance": height_distance,
        }

    return {
        "valid": True,
        "reason": "bbox_size_ok",
        "distance": size_distance,
        "confidence": confidence,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "width_distance": width_distance,
        "height_distance": height_distance,
        "width_confidence": width_confidence,
        "height_confidence": height_confidence,
        **scores,
    }


def horizon_confidence_from_result(
    horizon_result: dict[str, Any], sensor_info: dict[str, Any]
) -> float:
    """Horizon/waterline sonucunun güven skorunu hesaplar.

    Args:
        horizon_result: sea_distance_from_image_point çıktısı.
        sensor_info: FOV/zoom bilgilerini içeren sensör sözlüğü.

    Returns:
        0-1 arası güven skoru.
    """
    if not horizon_result.get("valid"):
        return 0.0

    beta_deg = abs(safe_float(horizon_result.get("beta_deg"), 0.0))
    fov_h = safe_float(sensor_info.get("fov_h"), DEFAULT_FOV_H_DEG)

    # Dar FOV'da birkaç piksel su hattı hatası mesafeyi çok değiştirdiği için
    # horizon-only güveni düşük tutulur.
    if fov_h < 5.0:
        base_confidence = 0.20
    elif fov_h < 15.0:
        base_confidence = 0.30
    else:
        base_confidence = 0.45

    # Ufka çok yakın beta da, çok büyük beta da hassasiyet açısından risklidir.
    if beta_deg < 0.03:
        beta_factor = 0.55
    elif beta_deg < 0.10:
        beta_factor = 0.90
    elif beta_deg < 0.35:
        beta_factor = 0.75
    else:
        beta_factor = 0.45

    return clamp(base_confidence * beta_factor, 0.05, 0.55)


def weighted_log_average(
    value_a: float, weight_a: float, value_b: float, weight_b: float
) -> float:
    """İki pozitif mesafeyi logaritmik ağırlıklı ortalama ile birleştirir.

    Mesafeler kilometre ölçeğinde çarpansal hata ürettiği için aritmetik ortalama
    bazen kötü davranır. Logaritmik ortalama oran farklarını daha dengeli işler.

    Args:
        value_a: İlk pozitif mesafe.
        weight_a: İlk mesafe ağırlığı.
        value_b: İkinci pozitif mesafe.
        weight_b: İkinci mesafe ağırlığı.

    Returns:
        Birleştirilmiş pozitif mesafe.
    """
    total_weight = max(weight_a + weight_b, 1e-6)

    return math.exp(
        (math.log(value_a) * weight_a + math.log(value_b) * weight_b)
        / total_weight
    )


def fuse_horizon_and_size_distance(
    horizon_result: dict[str, Any],
    size_result: dict[str, Any],
    sensor_info: dict[str, Any],
) -> dict[str, Any]:
    """Horizon ve bbox-size mesafe sonuçlarını tek tahmine indirger.

    Args:
        horizon_result: Horizon/waterline tabanlı mesafe çıktısı.
        size_result: Bbox-size tabanlı mesafe çıktısı.
        sensor_info: FOV/zoom bilgilerini içeren sensör sözlüğü.

    Returns:
        Hibrit mesafe sonucu.
    """
    horizon_valid = bool(horizon_result.get("valid"))
    size_valid = bool(size_result.get("valid"))

    horizon_distance = horizon_result.get("distance")
    size_distance = size_result.get("distance")

    horizon_confidence = horizon_confidence_from_result(
        horizon_result, sensor_info
    )
    size_confidence = safe_float(size_result.get("confidence"), 0.0)

    if size_valid and not horizon_valid:
        return {
            **horizon_result,
            "valid": True,
            "reason": "size_only",
            "distance": size_distance,
            "raw_distance": size_distance,
            "horizon_distance": None,
            "size_distance": size_distance,
            "distance_source": "size_only",
            "distance_confidence": size_confidence,
            "size_result": size_result,
        }

    if horizon_valid and not size_valid:
        return {
            **horizon_result,
            "horizon_distance": horizon_distance,
            "size_distance": None,
            "distance_source": "horizon_only",
            "distance_confidence": horizon_confidence,
            "size_result": size_result,
        }

    if not horizon_valid or not size_valid:
        return {
            **horizon_result,
            "valid": False,
            "reason": "no_valid_distance",
            "distance": None,
            "horizon_distance": None,
            "size_distance": None,
            "distance_source": "none",
            "distance_confidence": 0.0,
            "size_result": size_result,
        }

    horizon_distance_float = float(horizon_distance)
    size_distance_float = float(size_distance)

    ratio = max(horizon_distance_float, size_distance_float) / max(
        min(horizon_distance_float, size_distance_float), 1.0
    )

    if ratio > DISAGREEMENT_RATIO_THRES:
        # Bbox-size sonucu makul güvene sahipse ve horizon ile ciddi çelişiyorsa
        # dar FOV senaryosunda size sonucu baskın alınır. Bu, yanlış waterline
        # noktasının mesafeyi 6-10 km yerine 1 km seviyesine düşürmesini engeller.
        if size_confidence >= 0.30:
            size_weight = 0.82
            horizon_weight = 0.18
            reason = "size_dominant_disagreement"
        else:
            size_weight = SIZE_WEIGHT_DEFAULT
            horizon_weight = HORIZON_WEIGHT_DEFAULT
            reason = "hybrid_disagreement_low_confidence"
    else:
        size_weight = SIZE_WEIGHT_DEFAULT * size_confidence
        horizon_weight = HORIZON_WEIGHT_DEFAULT * horizon_confidence
        reason = "hybrid_agree"

    fused_distance = weighted_log_average(
        horizon_distance_float,
        horizon_weight,
        size_distance_float,
        size_weight,
    )

    confidence = clamp(
        0.5 * size_confidence + 0.5 * horizon_confidence, 0.05, 0.95
    )

    return {
        **horizon_result,
        "valid": True,
        "reason": reason,
        "distance": fused_distance,
        "raw_distance": fused_distance,
        "horizon_distance": horizon_distance_float,
        "size_distance": size_distance_float,
        "distance_source": "hybrid",
        "distance_confidence": confidence,
        "horizon_confidence": horizon_confidence,
        "size_confidence": size_confidence,
        "size_result": size_result,
    }


def water_point_from_box_for_geometry(
    box: Box, sensor_info: dict[str, Any]
) -> tuple[float, float]:
    """Geometry içinde bbox su hattı noktasını hesaplar.

    detector.py içinde de benzer hesap vardır. Bu fonksiyon geometry.py içinde
    dış bağımlılık oluşturmadan hibrit mesafe tahmini yapabilmek için tutulur.

    Args:
        box: Detection veya track bounding box değeri.
        sensor_info: FOV/zoom bilgilerini içeren sensör sözlüğü.

    Returns:
        Su hattı x ve y piksel koordinatı.
    """
    x1, y1, x2, y2 = box
    height = max(1.0, y2 - y1)
    fov_h = safe_float(sensor_info.get("fov_h"), DEFAULT_FOV_H_DEG)

    if fov_h < 15.0:
        ratio = 0.86
    else:
        ratio = 0.90

    return (x1 + x2) / 2.0, y1 + ratio * height


def estimate_hybrid_distance_from_box(
    box: Box, sensor_info: dict[str, Any], horizon_state: dict[str, Any]
) -> dict[str, Any]:
    """BBox, FOV/zoom ve horizon bilgisinden hibrit mesafe hesaplar.

    Args:
        box: Detection veya track bounding box değeri.
        sensor_info: FOV/zoom bilgilerini içeren sensör sözlüğü.
        horizon_state: Güncel ufuk çizgisi state bilgisi.

    Returns:
        Horizon-only ve bbox-size sonuçlarını içeren hibrit mesafe sözlüğü.
    """
    water_x, water_y = water_point_from_box_for_geometry(box, sensor_info)

    horizon_result = sea_distance_from_image_point(
        water_x, water_y, sensor_info, horizon_state
    )

    size_result = estimate_distance_from_bbox_size(box, sensor_info)

    return fuse_horizon_and_size_distance(
        horizon_result, size_result, sensor_info
    )


def format_distance(distance_m: float | None) -> str:
    """Mesafe değerini ekranda okunabilir metin formatına çevirir.

    Args:
        distance_m: Metre cinsinden mesafe değeri.

    Returns:
        Metre veya kilometre formatında okunabilir mesafe metni.
    """
    if distance_m is None:
        return "?"

    if distance_m >= 1000.0:
        return f"{distance_m / 1000.0:.2f} km"

    return f"{distance_m:.1f} m"
