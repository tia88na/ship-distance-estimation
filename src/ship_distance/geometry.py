"""Kamera geometrisi, ufuk çizgisi ve deniz mesafesi hesaplama yardımcıları.

Bu dosya, görüntü üzerindeki piksel konumlarından yaklaşık deniz mesafesi
hesaplamak için kullanılan matematiksel yardımcı fonksiyonları içerir.

Mesafe tahmini doğrudan görüntüdeki nesne boyutuna göre yapılmaz. Bunun yerine
kamera yüksekliği, kamera görüş açısı, tilt bilgisi, ufuk çizgisi konumu ve
nesnenin görüntüdeki su hattı noktası birlikte değerlendirilir.
"""

from collections import deque
import math
from typing import Any


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
    sensor_info: dict[str, Any],
    pitch_bias_rad: float = 0.0,
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
    current_y: float,
    target_y: float,
    max_step: float,
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
    gray: Any,
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
        predict_horizon_y_from_tilt(sensor_info, horizon_state["pitch_bias_rad"])
    )

    # İlk frame'de geçmiş değer olmadığı için ufuk doğrudan hedef değere atanır.
    if horizon_state["y"] is None:
        horizon_state["y"] = target_y
        horizon_state["slope"] = 0.0
        horizon_state["mode"] = "TILT_ONLY"
        return horizon_state

    # Kamera hareketliyken ufuk daha hızlı güncellenir. Kamera sabitken daha
    # düşük EMA katsayısı kullanılarak titreşim ve küçük sensör oynamaları azaltılır.
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
    fx_value, fy_value = focal_from_fov(sensor_info["fov_h"], sensor_info["fov_v"])

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
