"""Dosya ve görselleştirmeden bağımsız mesafe hesaplama API'si.

Bu modül video okumaz, CSV açmaz, YOLO çalıştırmaz, track oluşturmaz ve
görüntü çizmez. Dışarıdan verilen detection, kamera ve sensör bilgileriyle
mesafe hesaplar.

API üç aday sonuç üretir:

1. Horizontal-line / deniz düzlemi tabanlı mesafe,
2. Bbox-size / kelebek geometrisi tabanlı mesafe,
3. Bu iki sonucun güven skorlarına göre birleştirilmiş nihai mesafesi.

Bu yapı başka projelerden doğrudan import edilip kullanılabilir.
"""

from dataclasses import dataclass
import math
from typing import Literal


StreamType = Literal["rgb", "thermal"]


@dataclass(frozen=True)
class DistanceInput:
    """Tek bir detection için mesafe hesaplama girdileri.

    x ve y değerleri bbox merkez koordinatlarıdır.

    Attributes:
        track_id: Detection veya track kimliği.
        x: Bbox merkez x koordinatı.
        y: Bbox merkez y koordinatı.
        w: Bbox genişliği.
        h: Bbox yüksekliği.
        image_width: Görüntünün piksel genişliği.
        image_height: Görüntünün piksel yüksekliği.
        fov_h_deg: Yatay görüş açısı.
        fov_v_deg: Dikey görüş açısı.
        zoom: Normalize edilmiş zoom değeri.
        tilt_deg: Kamera tilt değeri.
        camera_height_m: Kameranın deniz seviyesinden yüksekliği.
        roll_deg: Kamera roll açısı.
        pitch_deg: Kamera pitch açısı.
        stream: RGB veya thermal akış tipi.
    """

    track_id: int

    x: float
    y: float
    w: float
    h: float

    image_width: int
    image_height: int

    fov_h_deg: float
    fov_v_deg: float
    zoom: float

    tilt_deg: float
    camera_height_m: float

    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    stream: StreamType = "rgb"


@dataclass(frozen=True)
class DistanceOutput:
    """Mesafe API çıktısı.

    Attributes:
        track_id: Girdiyle aynı track kimliği.
        distance_m: Birleştirilmiş nihai mesafe.
        horizontal_line_distance_m: Horizontal-line aday mesafesi.
        butterfly_distance_m: Bbox-size aday mesafesi.
        horizontal_confidence: Horizontal-line sonucu güven skoru.
        butterfly_confidence: Bbox-size sonucu güven skoru.
        confidence: Nihai sonuç güven skoru.
        valid: Sonucun geçerli olup olmadığı.
        reason: Kullanılan kararın açıklaması.
    """

    track_id: int

    distance_m: float

    horizontal_line_distance_m: float | None
    butterfly_distance_m: float | None

    horizontal_confidence: float
    butterfly_confidence: float
    confidence: float

    valid: bool
    reason: str


class DistanceAPI:
    """Dış projelerden çağrılabilen modüler mesafe hesaplama API'si."""

    def __init__(
        self,
        max_distance_m: float = 30000.0,
        min_distance_m: float = 5.0,
        default_ship_length_m: float = 130.0,
        default_ship_height_m: float = 26.0,
    ) -> None:
        """Mesafe hesaplama API'sini oluşturur.

        Args:
            max_distance_m: Kabul edilen maksimum mesafe.
            min_distance_m: Kabul edilen minimum mesafe.
            default_ship_length_m: Kelebek hesabında kullanılan görünür uzunluk.
            default_ship_height_m: Kelebek hesabında kullanılan görünür yükseklik.
        """
        if max_distance_m <= 0:
            raise ValueError("max_distance_m sıfırdan büyük olmalıdır.")

        if min_distance_m < 0:
            raise ValueError("min_distance_m negatif olamaz.")

        if default_ship_length_m <= 0:
            raise ValueError("default_ship_length_m sıfırdan büyük olmalıdır.")

        if default_ship_height_m <= 0:
            raise ValueError("default_ship_height_m sıfırdan büyük olmalıdır.")

        self.max_distance_m = float(max_distance_m)
        self.min_distance_m = float(min_distance_m)
        self.default_ship_length_m = float(default_ship_length_m)
        self.default_ship_height_m = float(default_ship_height_m)

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        """Bir değeri verilen aralıkta sınırlar."""
        return max(minimum, min(maximum, value))

    @staticmethod
    def rotate_pixel(
        x: float,
        y: float,
        roll_deg: float,
        image_width: float,
        image_height: float,
    ) -> tuple[float, float]:
        """Pikseli görüntü merkezi çevresinde roll açısına göre döndürür."""
        if abs(roll_deg) < 1e-9:
            return x, y

        roll_rad = math.radians(roll_deg)

        center_x = image_width / 2.0
        center_y = image_height / 2.0

        x_rotated = (
            math.cos(roll_rad) * (x - center_x)
            - math.sin(roll_rad) * (y - center_y)
            + center_x
        )

        y_rotated = (
            math.sin(roll_rad) * (x - center_x)
            + math.cos(roll_rad) * (y - center_y)
            + center_y
        )

        return x_rotated, y_rotated

    @staticmethod
    def focal_from_fov(
        image_width: float,
        image_height: float,
        fov_h_deg: float,
        fov_v_deg: float,
    ) -> tuple[float, float]:
        """FOV değerlerinden piksel cinsinden odak uzaklığı hesaplar."""
        if not 0.0 < fov_h_deg < 180.0:
            raise ValueError("fov_h_deg 0 ile 180 derece arasında olmalıdır.")

        if not 0.0 < fov_v_deg < 180.0:
            raise ValueError("fov_v_deg 0 ile 180 derece arasında olmalıdır.")

        fx = (image_width / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
        fy = (image_height / 2.0) / math.tan(math.radians(fov_v_deg) / 2.0)

        return fx, fy

    @staticmethod
    def bbox_to_xyxy(
        distance_input: DistanceInput,
    ) -> tuple[float, float, float, float]:
        """Merkez formatındaki bbox değerini x1, y1, x2, y2 formatına çevirir."""
        half_width = distance_input.w / 2.0
        half_height = distance_input.h / 2.0

        return (
            distance_input.x - half_width,
            distance_input.y - half_height,
            distance_input.x + half_width,
            distance_input.y + half_height,
        )

    @staticmethod
    def get_water_point(
        distance_input: DistanceInput,
    ) -> tuple[float, float]:
        """Mesafe hesabı için bbox içindeki yaklaşık su hattı noktasını seçer."""
        x1, y1, x2, y2 = DistanceAPI.bbox_to_xyxy(distance_input)

        bbox_height = max(1.0, y2 - y1)

        is_narrow_fov = (
            distance_input.fov_h_deg <= 12.0
            or distance_input.fov_v_deg <= 9.0
            or distance_input.zoom >= 0.85
        )

        if is_narrow_fov:
            waterline_ratio = 0.92
        else:
            waterline_ratio = 0.90

        water_x = (x1 + x2) / 2.0
        water_y = y1 + waterline_ratio * bbox_height

        return water_x, water_y

    def validate_input(self, distance_input: DistanceInput) -> str | None:
        """API girdilerinin kullanılabilir olup olmadığını kontrol eder."""
        if distance_input.image_width <= 0:
            return "invalid_image_width"

        if distance_input.image_height <= 0:
            return "invalid_image_height"

        if distance_input.w <= 1.0 or distance_input.h <= 1.0:
            return "invalid_bbox_size"

        if distance_input.camera_height_m <= 0.0:
            return "invalid_camera_height"

        if not 0.0 < distance_input.fov_h_deg < 180.0:
            return "invalid_horizontal_fov"

        if not 0.0 < distance_input.fov_v_deg < 180.0:
            return "invalid_vertical_fov"

        return None

    def calculate_horizontal_line_distance(
        self,
        distance_input: DistanceInput,
    ) -> tuple[float | None, float, str]:
        """Horizontal-line ve kamera açılarıyla yaklaşık mesafe hesaplar.

        Bu yöntem detection'ın bbox alt merkezine yakın su hattı noktasını,
        FOV, tilt, roll, pitch ve kamera yüksekliğiyle birlikte kullanır.

        Returns:
            Mesafe, güven skoru ve açıklama.
        """
        water_x, water_y = self.get_water_point(distance_input)

        water_x, water_y = self.rotate_pixel(
            water_x,
            water_y,
            distance_input.roll_deg,
            distance_input.image_width,
            distance_input.image_height,
        )

        # Alperen'in mevcut hesaplama yapısındaki açı düzeni korunur.
        alpha_base = (
            distance_input.tilt_deg
            - distance_input.fov_v_deg / 2.0
            + distance_input.pitch_deg
        )

        alpha_pixel = distance_input.fov_v_deg * (
            1.0 - water_y / distance_input.image_height
        )

        vertical_angle_deg = alpha_base + alpha_pixel

        if vertical_angle_deg <= 0.0 or vertical_angle_deg >= 90.0:
            return None, 0.0, "horizontal_invalid_vertical_angle"

        forward_distance = (
            math.tan(math.radians(vertical_angle_deg))
            * distance_input.camera_height_m
        )

        horizontal_angle_deg = abs(
            distance_input.fov_h_deg
            * (0.5 - water_x / distance_input.image_width)
        )

        lateral_distance = (
            math.tan(math.radians(horizontal_angle_deg)) * forward_distance
        )

        distance = math.hypot(forward_distance, lateral_distance)

        if not self.min_distance_m <= distance <= self.max_distance_m:
            return None, 0.0, "horizontal_distance_out_of_range"

        # Dar FOV'da birkaç piksel su hattı farkı büyük mesafe değişimi
        # oluşturabildiği için güven düşürülür.
        if distance_input.fov_h_deg < 5.0:
            confidence = 0.20
        elif distance_input.fov_h_deg < 15.0:
            confidence = 0.30
        else:
            confidence = 0.45

        if abs(distance_input.roll_deg) > 10.0:
            confidence *= 0.80

        if abs(distance_input.pitch_deg) > 10.0:
            confidence *= 0.85

        return (
            distance,
            self.clamp(confidence, 0.05, 0.60),
            "horizontal_ok",
        )

    def calculate_butterfly_distance(
        self,
        distance_input: DistanceInput,
    ) -> tuple[float | None, float, str]:
        """BBox-size ve FOV kullanarak kelebek geometrisi mesafesi hesaplar.

        Gerçek gemi boyu bilinmediği için varsayılan görünür gemi uzunluğu ve
        yüksekliği kullanılır. Bbox oranına göre width ve height sonuçlarının
        güven ağırlıkları değiştirilir.

        Returns:
            Mesafe, güven skoru ve açıklama.
        """
        fx, fy = self.focal_from_fov(
            distance_input.image_width,
            distance_input.image_height,
            distance_input.fov_h_deg,
            distance_input.fov_v_deg,
        )

        aspect_ratio = distance_input.w / max(distance_input.h, 1.0)

        side_score = self.clamp((aspect_ratio - 1.2) / 2.8, 0.0, 1.0)
        bow_score = self.clamp((1.8 - aspect_ratio) / 1.2, 0.0, 1.0)

        width_confidence = self.clamp(
            0.10 + 0.75 * side_score,
            0.10,
            0.85,
        )

        height_confidence = self.clamp(
            0.25 + 0.35 * bow_score,
            0.20,
            0.60,
        )

        width_distance = (
            self.default_ship_length_m * fx / distance_input.w
        )

        height_distance = (
            self.default_ship_height_m * fy / distance_input.h
        )

        total_weight = width_confidence + height_confidence

        if total_weight <= 0.0:
            return None, 0.0, "butterfly_weight_zero"

        distance = (
            width_distance * width_confidence
            + height_distance * height_confidence
        ) / total_weight

        if not self.min_distance_m <= distance <= self.max_distance_m:
            return None, 0.0, "butterfly_distance_out_of_range"

        bbox_area_ratio = (
            distance_input.w * distance_input.h
        ) / float(distance_input.image_width * distance_input.image_height)

        bbox_score = self.clamp(
            0.30 + 0.70 * self.clamp(
                (bbox_area_ratio - 0.00012) / 0.014,
                0.0,
                1.0,
            ),
            0.05,
            1.0,
        )

        confidence = bbox_score * (
            0.35 + 0.65 * max(width_confidence, height_confidence)
        )

        is_narrow_fov = (
            distance_input.fov_h_deg <= 12.0
            or distance_input.fov_v_deg <= 9.0
            or distance_input.zoom >= 0.85
        )

        if is_narrow_fov:
            # Dar FOV'da bbox geminin tamamı yerine yalnızca bir bölümünü
            # kapsayabilir. Bu nedenle kelebek geometrisi güveni azaltılır.
            confidence *= 0.60

        x1, y1, x2, y2 = self.bbox_to_xyxy(distance_input)

        touches_edge = (
            x1 <= 2.0
            or y1 <= 2.0
            or x2 >= distance_input.image_width - 3.0
            or y2 >= distance_input.image_height - 3.0
        )

        if touches_edge:
            confidence *= 0.60

        return (
            distance,
            self.clamp(confidence, 0.03, 0.95),
            "butterfly_ok",
        )

    @staticmethod
    def weighted_log_average(
        first_distance: float,
        first_weight: float,
        second_distance: float,
        second_weight: float,
    ) -> float:
        """İki pozitif mesafeyi logaritmik ağırlıklı ortalama ile birleştirir."""
        total_weight = max(first_weight + second_weight, 1e-6)

        return math.exp(
            (
                math.log(max(first_distance, 1.0)) * first_weight
                + math.log(max(second_distance, 1.0)) * second_weight
            )
            / total_weight
        )

    def fuse_distances(
        self,
        horizontal_distance: float | None,
        horizontal_confidence: float,
        butterfly_distance: float | None,
        butterfly_confidence: float,
    ) -> tuple[float | None, float, str]:
        """Horizontal-line ve kelebek sonuçlarını tek mesafeye indirger."""
        if horizontal_distance is None and butterfly_distance is None:
            return None, 0.0, "no_valid_distance"

        if horizontal_distance is not None and butterfly_distance is None:
            return (
                horizontal_distance,
                horizontal_confidence,
                "horizontal_only",
            )

        if horizontal_distance is None and butterfly_distance is not None:
            return (
                butterfly_distance,
                butterfly_confidence,
                "butterfly_only",
            )

        assert horizontal_distance is not None
        assert butterfly_distance is not None

        ratio = max(horizontal_distance, butterfly_distance) / max(
            min(horizontal_distance, butterfly_distance),
            1.0,
        )

        if ratio > 2.2:
            # İki yöntem çok ayrışıyorsa güven skoruna göre daha güçlü yöntem
            # baskın tutulur; ikisinin kör şekilde ortalaması alınmaz.
            if butterfly_confidence >= 0.55:
                horizontal_weight = 0.25
                butterfly_weight = 0.75
                reason = "butterfly_dominant_disagreement"
            elif horizontal_confidence >= butterfly_confidence:
                horizontal_weight = 0.70
                butterfly_weight = 0.30
                reason = "horizontal_dominant_disagreement"
            else:
                horizontal_weight = 0.45
                butterfly_weight = 0.55
                reason = "balanced_disagreement"
        else:
            horizontal_weight = max(horizontal_confidence, 0.05)
            butterfly_weight = max(butterfly_confidence, 0.05)
            reason = "distance_methods_agree"

        final_distance = self.weighted_log_average(
            horizontal_distance,
            horizontal_weight,
            butterfly_distance,
            butterfly_weight,
        )

        final_confidence = self.clamp(
            (
                horizontal_confidence * horizontal_weight
                + butterfly_confidence * butterfly_weight
            )
            / max(horizontal_weight + butterfly_weight, 1e-6),
            0.03,
            0.95,
        )

        return final_distance, final_confidence, reason

    def calc_distance(
        self,
        distance_input: DistanceInput,
    ) -> DistanceOutput:
        """Tek detection için bütün mesafe hesaplarını çalıştırır."""
        validation_error = self.validate_input(distance_input)

        if validation_error is not None:
            return DistanceOutput(
                track_id=distance_input.track_id,
                distance_m=-1.0,
                horizontal_line_distance_m=None,
                butterfly_distance_m=None,
                horizontal_confidence=0.0,
                butterfly_confidence=0.0,
                confidence=0.0,
                valid=False,
                reason=validation_error,
            )

        (
            horizontal_distance,
            horizontal_confidence,
            horizontal_reason,
        ) = self.calculate_horizontal_line_distance(distance_input)

        (
            butterfly_distance,
            butterfly_confidence,
            butterfly_reason,
        ) = self.calculate_butterfly_distance(distance_input)

        final_distance, final_confidence, fuse_reason = self.fuse_distances(
            horizontal_distance,
            horizontal_confidence,
            butterfly_distance,
            butterfly_confidence,
        )

        if final_distance is None:
            return DistanceOutput(
                track_id=distance_input.track_id,
                distance_m=-1.0,
                horizontal_line_distance_m=horizontal_distance,
                butterfly_distance_m=butterfly_distance,
                horizontal_confidence=horizontal_confidence,
                butterfly_confidence=butterfly_confidence,
                confidence=0.0,
                valid=False,
                reason=(
                    f"{fuse_reason};"
                    f"{horizontal_reason};"
                    f"{butterfly_reason}"
                ),
            )

        if not self.min_distance_m <= final_distance <= self.max_distance_m:
            return DistanceOutput(
                track_id=distance_input.track_id,
                distance_m=-1.0,
                horizontal_line_distance_m=horizontal_distance,
                butterfly_distance_m=butterfly_distance,
                horizontal_confidence=horizontal_confidence,
                butterfly_confidence=butterfly_confidence,
                confidence=0.0,
                valid=False,
                reason="final_distance_out_of_range",
            )

        return DistanceOutput(
            track_id=distance_input.track_id,
            distance_m=final_distance,
            horizontal_line_distance_m=horizontal_distance,
            butterfly_distance_m=butterfly_distance,
            horizontal_confidence=horizontal_confidence,
            butterfly_confidence=butterfly_confidence,
            confidence=final_confidence,
            valid=True,
            reason=fuse_reason,
        )
