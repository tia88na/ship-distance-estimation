"""Horizontal-line (HL) tabanlı bağımsız mesafe hesaplama API'si.

Bu dosya yalnızca ufuk çizgisi ile geminin yaklaşık su hattı arasındaki
geometriden mesafe üretir. Butterfly/bbox-size hesabı ve iki yöntemi
birleştiren fusion mantığı bu modülde bulunmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


# Bbox değerleri (x1, y1, x2, y2) formatında tutulur.
Box: TypeAlias = tuple[float, float, float, float]

# Dünya eğriliği ve atmosferik kırılma hesabında kullanılan sabitler.
EARTH_RADIUS_M = 6_371_000.0
DEFAULT_REFRACTION_K = 0.13

# Kullanılabilir mesafe sonuçlarının varsayılan sınırları.
DEFAULT_MIN_DISTANCE_M = 5.0
DEFAULT_MAX_DISTANCE_M = 30_000.0

# Su hattı ufka bundan daha yakınsa hesap kararsız kabul edilir.
MIN_BETA_DEG = 0.015


@dataclass(frozen=True)
class DistanceHlResult:
    """Horizontal-line hesabının dışarıya döndürdüğü sonuç.

    Attributes:
        track_id: Mesafesi hesaplanan detection veya track kimliği.
        distance_m: Geçerli sonuç varsa metre cinsinden mesafe.
        confidence: Hesabın 0 ile 1 arasındaki güven değeri.
        valid: Sonucun kullanılabilir olup olmadığı.
        reason: Sonucun geçerli veya geçersiz olma nedeni.
    """

    track_id: int
    distance_m: float | None
    confidence: float
    valid: bool
    reason: str


class DistanceHlApi:
    """Ufuk çizgisi ve bbox su hattından deniz mesafesi hesaplar."""

    def __init__(
        self,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
        refraction_k: float = DEFAULT_REFRACTION_K,
    ) -> None:
        """Mesafe sınırlarını ve atmosferik kırılma katsayısını ayarlar."""
        if min_distance_m < 0.0:
            raise ValueError("min_distance_m negatif olamaz.")

        if max_distance_m <= min_distance_m:
            raise ValueError(
                "max_distance_m, min_distance_m değerinden büyük olmalıdır."
            )

        if not 0.0 <= refraction_k < 1.0:
            raise ValueError("refraction_k 0 ile 1 arasında olmalıdır.")

        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.refraction_k = float(refraction_k)

    def calc_distance(
        self,
        track_id: int,
        box: Box,
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
        tilt_deg: float,
        camera_height_m: float,
        zoom: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        horizon_y: float | None = None,
        horizon_slope: float = 0.0,
    ) -> DistanceHlResult:
        """Tek bir bbox için horizontal-line tabanlı mesafe hesaplar.

        Args:
            track_id: Detection veya track kimliği.
            box: (x1, y1, x2, y2) formatındaki bbox.
            image_width: İşlenen görüntünün piksel genişliği.
            image_height: İşlenen görüntünün piksel yüksekliği.
            fov_h_deg: Yatay görüş açısı.
            fov_v_deg: Dikey görüş açısı.
            tilt_deg: Kameradan gelen tilt değeri.
            camera_height_m: Kameranın deniz seviyesinden yüksekliği.
            zoom: Normalize edilmiş zoom değeri.
            roll_deg: Kameranın roll açısı.
            pitch_deg: Kameranın ek pitch düzeltmesi.
            horizon_y: Biliniyorsa görüntü merkezindeki ufuk y konumu.
            horizon_slope: Ufuk çizgisinin piksel tabanlı eğimi.

        Returns:
            Mesafe, güven ve geçerlilik bilgilerini içeren sonuç.
        """
        x1, y1, x2, y2 = box

        # Trigonometrik hesaplara geçmeden önce bütün sayısal girişlerin
        # kullanılabilir ve sonlu olduğu doğrulanır.
        finite_values = (
            x1,
            y1,
            x2,
            y2,
            fov_h_deg,
            fov_v_deg,
            tilt_deg,
            camera_height_m,
            zoom,
            roll_deg,
            pitch_deg,
            horizon_slope,
        )

        if not all(math.isfinite(value) for value in finite_values):
            return self._invalid_result(track_id, "non_finite_input")

        if horizon_y is not None and not math.isfinite(horizon_y):
            return self._invalid_result(track_id, "non_finite_horizon")

        if image_width <= 0 or image_height <= 0:
            return self._invalid_result(track_id, "invalid_image_size")

        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            return self._invalid_result(track_id, "invalid_bbox_size")

        if camera_height_m <= 0.0:
            return self._invalid_result(track_id, "invalid_camera_height")

        if not self._is_valid_fov(fov_h_deg):
            return self._invalid_result(track_id, "invalid_horizontal_fov")

        if not self._is_valid_fov(fov_v_deg):
            return self._invalid_result(track_id, "invalid_vertical_fov")

        # Detector bbox'ı her zaman geminin gerçek su seviyesinde bitmez.
        # Dar FOV ve zoom altında su hattı bbox'ın altına daha yakın seçilir.
        water_x, water_y = self._get_water_point(
            box=box,
            image_width=image_width,
            image_height=image_height,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            zoom=zoom,
        )

        # Kamera roll yaptığında görüntü ekseni de döner. Su hattı noktası
        # mesafe açısı hesaplanmadan önce görüntü merkezine göre düzeltilir.
        water_x, water_y = self._rotate_pixel(
            x=water_x,
            y=water_y,
            roll_deg=roll_deg,
            image_width=image_width,
            image_height=image_height,
        )

        # Dikey FOV değeri piksel cinsinden focal length değerine çevrilir.
        focal_y = (image_height / 2.0) / math.tan(
            math.radians(fov_v_deg) / 2.0
        )

        # Güncel bir ufuk değeri dışarıdan verildiyse doğrudan kullanılır.
        # Verilmediyse tilt, pitch ve kamera yüksekliğiyle yaklaşık değer üretilir.
        if horizon_y is None:
            resolved_horizon_y = self._predict_horizon_y(
                image_height=image_height,
                focal_y=focal_y,
                tilt_deg=tilt_deg,
                pitch_deg=pitch_deg,
                camera_height_m=camera_height_m,
            )
        else:
            resolved_horizon_y = float(horizon_y)

        # Ufuk çizgisi eğimliyse su hattının x konumundaki gerçek ufuk y
        # koordinatı hesaplanır. Eğim sıfırsa horizon_y değişmeden kullanılır.
        center_x = image_width / 2.0
        horizon_at_water = resolved_horizon_y + horizon_slope * (
            water_x - center_x
        )

        # Beta, geminin su hattının ufuk çizgisinin ne kadar altında kaldığını
        # gösteren açıdır.
        beta_rad = math.atan((water_y - horizon_at_water) / focal_y)
        beta_deg = math.degrees(beta_rad)

        # Ufka aşırı yakın noktalarda birkaç piksellik hata kilometre seviyesinde
        # sapma oluşturabildiği için bu sonuçlar geçersiz kabul edilir.
        if beta_deg <= MIN_BETA_DEG:
            return self._invalid_result(track_id, "at_or_beyond_horizon")

        # Dünya eğriliğinin oluşturduğu teorik ufuk çöküşü beta açısına eklenir.
        # Böylece kameranın deniz yüzeyine toplam aşağı bakış açısı bulunur.
        depression_rad = self._horizon_dip_rad(camera_height_m) + beta_rad

        distance_m = self._sea_distance_from_depression(
            depression_rad=depression_rad,
            camera_height_m=camera_height_m,
        )

        if distance_m is None:
            return self._invalid_result(track_id, "distance_unavailable")

        if not self.min_distance_m <= distance_m <= self.max_distance_m:
            return self._invalid_result(track_id, "distance_out_of_range")

        confidence = self._calculate_confidence(
            beta_deg=beta_deg,
            fov_h_deg=fov_h_deg,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
        )

        return DistanceHlResult(
            track_id=track_id,
            distance_m=distance_m,
            confidence=confidence,
            valid=True,
            reason="horizontal_line_ok",
        )

    @staticmethod
    def _invalid_result(
        track_id: int,
        reason: str,
    ) -> DistanceHlResult:
        """Geçersiz hesaplar için ortak sonuç oluşturur."""
        return DistanceHlResult(
            track_id=track_id,
            distance_m=None,
            confidence=0.0,
            valid=False,
            reason=reason,
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Bir değeri verilen alt ve üst sınırlar içinde tutar."""
        return max(minimum, min(maximum, value))

    @staticmethod
    def _is_valid_fov(fov_deg: float) -> bool:
        """FOV değerinin trigonometrik hesap için geçerli olduğunu doğrular."""
        return 0.0 < fov_deg < 180.0

    @staticmethod
    def _rotate_pixel(
        x: float,
        y: float,
        roll_deg: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """Pikseli görüntü merkezi çevresinde roll açısına göre döndürür."""
        if abs(roll_deg) < 1e-9:
            return x, y

        angle_rad = math.radians(roll_deg)

        center_x = image_width / 2.0
        center_y = image_height / 2.0

        offset_x = x - center_x
        offset_y = y - center_y

        rotated_x = (
            math.cos(angle_rad) * offset_x
            - math.sin(angle_rad) * offset_y
            + center_x
        )

        rotated_y = (
            math.sin(angle_rad) * offset_x
            + math.cos(angle_rad) * offset_y
            + center_y
        )

        return rotated_x, rotated_y

    @staticmethod
    def _resolve_pitch_down_from_tilt(
        tilt_deg: float,
    ) -> float:
        """Farklı tilt aralıklarını ortak aşağı bakış açısına dönüştürür."""
        if 100.0 <= tilt_deg <= 180.0:
            return tilt_deg - 130.0

        if 45.0 <= tilt_deg <= 100.0:
            return 90.0 - tilt_deg

        if -45.0 <= tilt_deg <= 45.0:
            return tilt_deg

        if -135.0 <= tilt_deg <= -45.0:
            return 90.0 + tilt_deg

        return 0.0

    def _effective_earth_radius(self) -> float:
        """Atmosferik kırılma düzeltilmiş Dünya yarıçapını döndürür."""
        return EARTH_RADIUS_M / (1.0 - self.refraction_k)

    def _horizon_dip_rad(
        self,
        camera_height_m: float,
    ) -> float:
        """Kamera yüksekliğine göre teorik ufuk çöküşünü hesaplar."""
        return math.sqrt(
            2.0 * camera_height_m / self._effective_earth_radius()
        )

    def _maximum_sea_distance_m(
        self,
        camera_height_m: float,
    ) -> float:
        """Kameranın görebileceği teorik en uzak deniz mesafesini hesaplar."""
        return math.sqrt(
            2.0 * self._effective_earth_radius() * camera_height_m
        )

    def _sea_distance_from_depression(
        self,
        depression_rad: float,
        camera_height_m: float,
    ) -> float | None:
        """Aşağı bakış açısının deniz yüzeyiyle kesiştiği mesafeyi hesaplar."""
        horizon_dip = self._horizon_dip_rad(camera_height_m)

        if depression_rad <= horizon_dip:
            return None

        effective_radius = self._effective_earth_radius()
        tangent_value = math.tan(depression_rad)

        # Küresel Dünya modelindeki görüş ışını ile deniz yüzeyi kesişiminin
        # diskriminant değeri hesaplanır.
        discriminant = (
            effective_radius * tangent_value
        ) ** 2 - 2.0 * effective_radius * camera_height_m

        maximum_distance = self._maximum_sea_distance_m(camera_height_m)

        # Kesişim teorik sınıra çok yakınsa kamera yüksekliğine göre hesaplanan
        # maksimum ufuk mesafesi güvenli üst sınır olarak kullanılır.
        if discriminant <= 0.0:
            return min(maximum_distance, self.max_distance_m)

        distance_m = (
            effective_radius * tangent_value
            - math.sqrt(discriminant)
        )

        if not math.isfinite(distance_m) or distance_m <= 0.0:
            return None

        return min(
            distance_m,
            maximum_distance,
            self.max_distance_m,
        )

    def _predict_horizon_y(
        self,
        image_height: int,
        focal_y: float,
        tilt_deg: float,
        pitch_deg: float,
        camera_height_m: float,
    ) -> float:
        """Tilt, pitch ve kamera geometrisinden yaklaşık ufuk değeri üretir."""
        pitch_down_deg = (
            self._resolve_pitch_down_from_tilt(tilt_deg)
            + pitch_deg
        )

        # Pozitif aşağı bakış açısı ufuk çizgisini görüntü içinde yukarı taşır.
        horizon_angle_rad = (
            self._horizon_dip_rad(camera_height_m)
            - math.radians(pitch_down_deg)
        )

        # Aşırı sensör değerlerinin tanjant hesabını bozması engellenir.
        horizon_angle_rad = self._clamp(
            horizon_angle_rad,
            -1.2,
            1.2,
        )

        horizon_y = (
            image_height / 2.0
            + focal_y * math.tan(horizon_angle_rad)
        )

        # Hesaplanan ufuk görüntünün tamamen dışına taşmamalıdır.
        return self._clamp(
            horizon_y,
            image_height * 0.02,
            image_height * 0.90,
        )

    def _get_water_point(
        self,
        box: Box,
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
        zoom: float,
    ) -> tuple[float, float]:
        """Bbox içinde geminin yaklaşık su hattı noktasını seçer."""
        x1, y1, x2, y2 = box

        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)

        frame_area = float(image_width * image_height)
        box_area_ratio = (
            box_width * box_height
        ) / frame_area

        narrow_fov = (
            fov_h_deg <= 12.0
            or fov_v_deg <= 9.0
            or zoom >= 0.85
        )

        large_close_box = box_area_ratio >= 0.040

        # Dar FOV veya yakın termal görüntüde detector kutusu geminin yalnızca
        # üst gövdesini kapsayabilir. Su hattı bu nedenle alta yaklaştırılır.
        if narrow_fov and large_close_box:
            waterline_ratio = 0.96
        elif narrow_fov:
            waterline_ratio = 0.92
        elif fov_h_deg < 15.0:
            waterline_ratio = 0.88
        else:
            waterline_ratio = 0.90

        water_x = (x1 + x2) / 2.0
        water_y = y1 + waterline_ratio * box_height

        return water_x, water_y

    def _calculate_confidence(
        self,
        beta_deg: float,
        fov_h_deg: float,
        roll_deg: float,
        pitch_deg: float,
    ) -> float:
        """FOV, beta, roll ve pitch değerlerinden sonuç güvenini hesaplar."""
        # Dar FOV altında birkaç piksellik su hattı hatası mesafeyi ciddi biçimde
        # değiştirebildiği için temel güven daha düşük başlatılır.
        if fov_h_deg < 5.0:
            base_confidence = 0.20
        elif fov_h_deg < 15.0:
            base_confidence = 0.30
        else:
            base_confidence = 0.45

        absolute_beta = abs(beta_deg)

        # Ufka aşırı yakın veya görüntüde çok aşağıda kalan noktalar orta açı
        # bölgesine göre daha az güvenilir kabul edilir.
        if absolute_beta < 0.03:
            beta_factor = 0.55
        elif absolute_beta < 0.10:
            beta_factor = 0.90
        elif absolute_beta < 0.35:
            beta_factor = 0.75
        else:
            beta_factor = 0.45

        confidence = base_confidence * beta_factor

        # Büyük roll ve pitch değerleri ufuk-su hattı ilişkisini belirsizleştirir.
        if abs(roll_deg) > 10.0:
            confidence *= 0.80

        if abs(pitch_deg) > 10.0:
            confidence *= 0.85

        return self._clamp(
            confidence,
            0.05,
            0.55,
        )
