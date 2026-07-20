"""Butterfly/bbox-size tabanlı bağımsız mesafe hesaplama API'si.

Bu modül bir detection kutusunun piksel genişliği ve yüksekliğini, kameranın
yatay/dikey FOV değerlerini ve varsayılan gemi boyutlarını kullanarak yaklaşık
mesafe hesaplar.

Horizontal-line hesabı, ufuk çizgisi işlemleri ve iki yöntemi birleştiren
fusion mantığı bu dosyada bulunmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


# Bounding box değerleri (x1, y1, x2, y2) formatında kullanılır.
Box: TypeAlias = tuple[float, float, float, float]

# Gemi tipi kesin olarak bilinmediğinde kullanılan yaklaşık fiziksel boyutlar.
DEFAULT_SHIP_LENGTH_M = 130.0
DEFAULT_SHIP_HEIGHT_M = 26.0

# Butterfly yönteminin kabul edeceği varsayılan mesafe aralığı.
DEFAULT_MIN_DISTANCE_M = 50.0
DEFAULT_MAX_DISTANCE_M = 30_000.0


@dataclass(frozen=True)
class DistanceButterflyResult:
    """Butterfly mesafe hesabının dışarıya döndürdüğü sonuç.

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


class DistanceButterflyApi:
    """BBox piksel boyutu ve FOV değerlerinden yaklaşık mesafe hesaplar."""

    def __init__(
    self: DistanceButterflyApi,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
        ship_length_m: float = DEFAULT_SHIP_LENGTH_M,
        ship_height_m: float = DEFAULT_SHIP_HEIGHT_M,
    ) -> None:
        """Butterfly hesabının sınırlarını ve gemi boyutlarını ayarlar.

        Args:
            min_distance_m: Kabul edilecek minimum mesafe.
            max_distance_m: Kabul edilecek maksimum mesafe.
            ship_length_m: Genişlik hesabında kullanılacak gemi uzunluğu.
            ship_height_m: Yükseklik hesabında kullanılacak gemi yüksekliği.
        """
        if min_distance_m < 0.0:
            raise ValueError("min_distance_m negatif olamaz.")

        if max_distance_m <= min_distance_m:
            raise ValueError(
                "max_distance_m, min_distance_m değerinden büyük olmalıdır."
            )

        if ship_length_m <= 0.0:
            raise ValueError("ship_length_m sıfırdan büyük olmalıdır.")

        if ship_height_m <= 0.0:
            raise ValueError("ship_height_m sıfırdan büyük olmalıdır.")

        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.ship_length_m = float(ship_length_m)
        self.ship_height_m = float(ship_height_m)

    def calc_distance(
    self: DistanceButterflyApi,
        track_id: int,
        box: Box,
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
        zoom: float = 0.0,
    ) -> DistanceButterflyResult:
        """Tek bir bbox için butterfly tabanlı mesafe hesaplar.

        Args:
            track_id: Detection veya track kimliği.
            box: (x1, y1, x2, y2) formatındaki bbox.
            image_width: İşlenen görüntünün piksel genişliği.
            image_height: İşlenen görüntünün piksel yüksekliği.
            fov_h_deg: Kameranın yatay görüş açısı.
            fov_v_deg: Kameranın dikey görüş açısı.
            zoom: Normalize edilmiş zoom değeri.

        Returns:
            Mesafe, güven ve geçerlilik bilgilerini içeren sonuç.
        """
        x1, y1, x2, y2 = box

        # Mesafe hesabına giren bütün sayısal değerlerin sonlu olması gerekir.
        # NaN veya sonsuz değerler trigonometrik hesabı bozacağı için reddedilir.
        finite_values = (
            x1,
            y1,
            x2,
            y2,
            fov_h_deg,
            fov_v_deg,
            zoom,
        )

        if not all(math.isfinite(value) for value in finite_values):
            return self._invalid_result(
                track_id=track_id,
                reason="non_finite_input",
            )

        if image_width <= 0 or image_height <= 0:
            return self._invalid_result(
                track_id=track_id,
                reason="invalid_image_size",
            )

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 1.0 or box_height <= 1.0:
            return self._invalid_result(
                track_id=track_id,
                reason="invalid_bbox_size",
            )

        if not self._is_valid_fov(fov_h_deg):
            return self._invalid_result(
                track_id=track_id,
                reason="invalid_horizontal_fov",
            )

        if not self._is_valid_fov(fov_v_deg):
            return self._invalid_result(
                track_id=track_id,
                reason="invalid_vertical_fov",
            )

        # Kamera FOV değerleri piksel cinsinden yatay ve dikey focal length
        # değerlerine çevrilir.
        focal_x, focal_y = self._focal_from_fov(
            image_width=image_width,
            image_height=image_height,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
        )

        # Bbox en-boy oranı geminin yandan mı yoksa önden/diyagonal mi
        # göründüğünü yaklaşık olarak belirlemek için kullanılır.
        aspect_ratio = box_width / box_height

        # Uzun ve yatay bbox, geminin yandan görünme ihtimalini artırır.
        # Bu durumda gerçek gemi uzunluğuna dayanan genişlik hesabı daha güvenilir
        # kabul edilir.
        side_score = self._clamp(
            (aspect_ratio - 1.2) / 2.8,
            0.0,
            1.0,
        )

        # Daha kompakt bbox, geminin önden veya diyagonal görünme ihtimalini
        # artırır. Bu durumda yükseklik tabanlı hesap korunur.
        bow_score = self._clamp(
            (1.8 - aspect_ratio) / 1.2,
            0.0,
            1.0,
        )

        # Yan görünüm skoru yükseldikçe width-based hesabın ağırlığı artırılır.
        width_confidence = self._clamp(
            0.10 + 0.75 * side_score,
            0.10,
            0.85,
        )

        # Önden/diyagonal görünümde yükseklik hesabı tamamen atılmaz.
        # Ancak gemi yüksekliği türlere göre değiştiği için üst güven sınırı
        # genişlik hesabından daha düşük tutulur.
        height_confidence = self._clamp(
            0.25 + 0.35 * bow_score,
            0.20,
            0.60,
        )

        # Pinhole kamera modelinde gerçek boyut ile focal length çarpımı,
        # görüntüdeki piksel boyutuna bölünerek yaklaşık mesafe elde edilir.
        width_distance = (
            self.ship_length_m * focal_x / box_width
        )

        height_distance = (
            self.ship_height_m * focal_y / box_height
        )

        total_weight = width_confidence + height_confidence

        if total_weight <= 0.0:
            return self._invalid_result(
                track_id=track_id,
                reason="butterfly_weight_zero",
            )

        # Genişlik ve yükseklik tahminleri, görünüm güvenlerine göre
        # ağırlıklı ortalamayla tek mesafeye dönüştürülür.
        distance_m = (
            width_distance * width_confidence
            + height_distance * height_confidence
        ) / total_weight

        frame_area = float(image_width * image_height)
        box_area_ratio = (
            box_width * box_height
        ) / frame_area

        # Çok küçük bbox'larda birkaç piksellik detection hatası mesafeyi ciddi
        # biçimde değiştirebilir. Bbox büyüdükçe boyut bilgisinin güveni artar.
        size_score = self._clamp(
            (box_area_ratio - 0.00012) / 0.014,
            0.0,
            1.0,
        )

        touches_left_or_top = x1 <= 2.0 or y1 <= 2.0

        touches_right_or_bottom = (
            x2 >= image_width - 3.0
            or y2 >= image_height - 3.0
        )

        # Görüntü kenarına değen bbox geminin tamamını içermeyebilir.
        # Kutunun iki farklı kenar grubuna temas etmesi cezayı artırır.
        edge_penalty = 0.0

        if touches_left_or_top:
            edge_penalty += 0.25

        if touches_right_or_bottom:
            edge_penalty += 0.25

        bbox_confidence = self._clamp(
            0.30 + 0.70 * size_score - edge_penalty,
            0.05,
            1.0,
        )

        # Nihai güven, bbox boyut güveni ile en güvenilir fiziksel boyut
        # tahmininin birlikte değerlendirilmesiyle elde edilir.
        confidence = bbox_confidence * (
            0.35
            + 0.65 * max(
                width_confidence,
                height_confidence,
            )
        )

        narrow_fov = (
            fov_h_deg <= 12.0
            or fov_v_deg <= 9.0
            or zoom >= 0.85
        )

        very_narrow_fov = (
            fov_h_deg <= 9.0
            or fov_v_deg <= 7.0
        )

        large_close_box = box_area_ratio >= 0.040

        # Dar FOV altında bbox görüntünün yalnızca bir bölümünü kapsayabilir.
        # Bu kontrol eski hesap davranışını korumak için genişlik oranını kullanır.
        partial_box = box_width < image_width * 0.50

        touches_edge = (
            touches_left_or_top
            or touches_right_or_bottom
        )

        penalty_factor = 1.0

        # Çok dar FOV ve büyük bbox birlikteyse kutunun geminin tamamını
        # göstermeme ihtimali yüksektir. Bu nedenle en güçlü güven cezası verilir.
        if very_narrow_fov and large_close_box:
            penalty_factor = 0.35

        # Dar FOV ve yakın/büyük bbox kombinasyonu da butterfly tahminini
        # güvensizleştirir.
        elif narrow_fov and large_close_box:
            penalty_factor = 0.45

        # Dar FOV altında görüntünün yarısından küçük kalan bbox için daha hafif
        # bir güven cezası uygulanır.
        elif narrow_fov and partial_box:
            penalty_factor = 0.60

        # Dar FOV ile birlikte görüntü kenarına değen bbox'ın tam olmadığı
        # varsayılarak güven üst sınırı düşürülür.
        if narrow_fov and touches_edge:
            penalty_factor = min(
                penalty_factor,
                0.55,
            )

        confidence = self._clamp(
            confidence * penalty_factor,
            0.02,
            0.95,
        )

        # Fiziksel olarak belirlenen kullanım aralığının dışındaki sonuçlar
        # döndürülmez. Güven değeri tanı amaçlı korunur.
        if not self.min_distance_m <= distance_m <= self.max_distance_m:
            return DistanceButterflyResult(
                track_id=track_id,
                distance_m=None,
                confidence=confidence,
                valid=False,
                reason="butterfly_distance_out_of_range",
            )

        return DistanceButterflyResult(
            track_id=track_id,
            distance_m=distance_m,
            confidence=confidence,
            valid=True,
            reason="butterfly_ok",
        )

    @staticmethod
    def _invalid_result(
        track_id: int,
        reason: str,
    ) -> DistanceButterflyResult:
        """Geçersiz hesaplar için ortak sonuç oluşturur."""
        return DistanceButterflyResult(
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
        return max(
            minimum,
            min(maximum, value),
        )

    @staticmethod
    def _is_valid_fov(fov_deg: float) -> bool:
        """FOV değerinin kamera hesabı için geçerli olduğunu doğrular."""
        return 0.0 < fov_deg < 180.0

    @staticmethod
    def _focal_from_fov(
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
    ) -> tuple[float, float]:
        """FOV ve görüntü boyutundan piksel focal length değerlerini hesaplar."""
        focal_x = (image_width / 2.0) / math.tan(
            math.radians(fov_h_deg) / 2.0
        )

        focal_y = (image_height / 2.0) / math.tan(
            math.radians(fov_v_deg) / 2.0
        )

        return focal_x, focal_y
