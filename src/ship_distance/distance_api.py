"""Dosya ve görselleştirmeden bağımsız mesafe hesaplama API'si.

Bu modül video okumaz, CSV açmaz, YOLO çalıştırmaz, track oluşturmaz ve
görüntü çizmez. Dışarıdan verilen detection, kamera ve sensör bilgilerini
kullanarak yaklaşık mesafe hesaplar.

API üç temel aşamadan oluşur:

1. Horizontal-line / horizon tabanlı deniz mesafesi,
2. Bbox-size / kelebek geometrisi tabanlı mesafe,
3. İki yöntemin güven skorlarına göre birleştirilmesi.

Görüntü çözünürlüğü, kamera yüksekliği, FOV, zoom, tilt, roll ve pitch
değerleri dışarıdan alınır. Bu nedenle API hem RGB hem de termal kamera
akışlarında bağımsız olarak kullanılabilir.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias


StreamType: TypeAlias = Literal["rgb", "thermal"]
Box: TypeAlias = tuple[float, float, float, float]

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_K = 0.13

DEFAULT_SHIP_LENGTH_M = 130.0
DEFAULT_SHIP_HEIGHT_M = 26.0

DEFAULT_MIN_DISTANCE_M = 5.0
DEFAULT_MAX_DISTANCE_M = 30_000.0

DEFAULT_SIZE_MIN_DISTANCE_M = 50.0
DEFAULT_SIZE_MAX_DISTANCE_M = 30_000.0

MIN_BETA_DEG = 0.015

HORIZON_WEIGHT_DEFAULT = 0.35
SIZE_WEIGHT_DEFAULT = 0.65
DISAGREEMENT_RATIO_THRESHOLD = 2.2


@dataclass(frozen=True)
class DistanceInput:
    """Tek detection için mesafe hesaplama girdileri.

    ``x`` ve ``y`` değerleri bbox merkez koordinatlarıdır.

    Attributes:
        track_id: Detection veya track kimliği.
        x: Bbox merkez x koordinatı.
        y: Bbox merkez y koordinatı.
        w: Bbox genişliği.
        h: Bbox yüksekliği.
        image_width: Görüntü genişliği.
        image_height: Görüntü yüksekliği.
        fov_h_deg: Yatay FOV değeri.
        fov_v_deg: Dikey FOV değeri.
        tilt_deg: Kamera tilt değeri.
        camera_height_m: Kameranın deniz seviyesinden yüksekliği.
        zoom: Normalize edilmiş zoom değeri.
        roll_deg: Kamera roll açısı.
        pitch_deg: Kamera pitch açısı.
        stream: RGB veya thermal akış tipi.
        horizon_y: Dış sistem tarafından bilinen ufuk y koordinatı.
        horizon_slope: Ufuk çizgisinin piksel tabanlı eğimi.
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

    tilt_deg: float
    camera_height_m: float

    zoom: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    stream: StreamType = "rgb"

    horizon_y: float | None = None
    horizon_slope: float = 0.0


@dataclass(frozen=True)
class DistanceOutput:
    """Mesafe API çıktısı.

    Attributes:
        track_id: Girdiyle aynı track kimliği.
        distance_m: Nihai birleştirilmiş mesafe.
        horizontal_line_distance_m: Horizon tabanlı aday mesafe.
        butterfly_distance_m: Bbox-size tabanlı aday mesafe.
        horizontal_confidence: Horizon sonucu güven skoru.
        butterfly_confidence: Bbox-size sonucu güven skoru.
        confidence: Nihai sonuç güven skoru.
        valid: Sonucun kullanılabilir olup olmadığı.
        reason: Nihai kararın açıklaması.
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


@dataclass(frozen=True)
class CandidateDistance:
    """Tek bir mesafe yönteminin ara sonucu."""

    distance_m: float | None
    confidence: float
    valid: bool
    reason: str


class DistanceAPI:
    """Dış projelerden çağrılabilen modüler mesafe hesaplama API'si."""

    def __init__(
        self: DistanceAPI,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
        min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
        size_min_distance_m: float = DEFAULT_SIZE_MIN_DISTANCE_M,
        size_max_distance_m: float = DEFAULT_SIZE_MAX_DISTANCE_M,
        default_ship_length_m: float = DEFAULT_SHIP_LENGTH_M,
        default_ship_height_m: float = DEFAULT_SHIP_HEIGHT_M,
        refraction_k: float = REFRACTION_K,
    ) -> None:
        """Mesafe API nesnesini oluşturur.

        Args:
            max_distance_m: Kabul edilecek maksimum nihai mesafe.
            min_distance_m: Kabul edilecek minimum nihai mesafe.
            size_min_distance_m: Bbox-size hesabının minimum mesafesi.
            size_max_distance_m: Bbox-size hesabının maksimum mesafesi.
            default_ship_length_m: Kelebek hesabındaki gemi uzunluk varsayımı.
            default_ship_height_m: Kelebek hesabındaki gemi yükseklik varsayımı.
            refraction_k: Atmosferik kırılma katsayısı.
        """
        if max_distance_m <= 0.0:
            raise ValueError("max_distance_m sıfırdan büyük olmalıdır.")

        if min_distance_m < 0.0:
            raise ValueError("min_distance_m negatif olamaz.")

        if min_distance_m >= max_distance_m:
            raise ValueError(
                "min_distance_m, max_distance_m değerinden küçük olmalıdır."
            )

        if size_min_distance_m < 0.0:
            raise ValueError("size_min_distance_m negatif olamaz.")

        if size_max_distance_m <= 0.0:
            raise ValueError("size_max_distance_m sıfırdan büyük olmalıdır.")

        if size_min_distance_m >= size_max_distance_m:
            raise ValueError(
                "size_min_distance_m, size_max_distance_m değerinden "
                "küçük olmalıdır."
            )

        if size_min_distance_m >= max_distance_m:
            raise ValueError(
                "size_min_distance_m, max_distance_m değerinden küçük olmalıdır."
            )

        if default_ship_length_m <= 0.0:
            raise ValueError("default_ship_length_m sıfırdan büyük olmalıdır.")

        if default_ship_height_m <= 0.0:
            raise ValueError("default_ship_height_m sıfırdan büyük olmalıdır.")

        if not 0.0 <= refraction_k < 1.0:
            raise ValueError("refraction_k 0 ile 1 arasında olmalıdır.")

        self.max_distance_m = float(max_distance_m)
        self.min_distance_m = float(min_distance_m)

        self.size_min_distance_m = float(size_min_distance_m)
        self.size_max_distance_m = float(
            min(size_max_distance_m, max_distance_m)
        )

        self.default_ship_length_m = float(default_ship_length_m)
        self.default_ship_height_m = float(default_ship_height_m)

        self.refraction_k = float(refraction_k)

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        """Bir değeri verilen aralıkta sınırlar."""
        return max(minimum, min(maximum, value))

    @staticmethod
    def validate_angle(angle_deg: float) -> bool:
        """FOV açısının geçerli aralıkta olup olmadığını kontrol eder."""
        return math.isfinite(angle_deg) and 0.0 < angle_deg < 180.0

    @staticmethod
    def bbox_to_xyxy(distance_input: DistanceInput) -> Box:
        """Merkez formatındaki bbox değerini xyxy formatına dönüştürür."""
        half_width = distance_input.w / 2.0
        half_height = distance_input.h / 2.0

        return (
            distance_input.x - half_width,
            distance_input.y - half_height,
            distance_input.x + half_width,
            distance_input.y + half_height,
        )

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

        rotated_x = (
            math.cos(roll_rad) * (x - center_x)
            - math.sin(roll_rad) * (y - center_y)
            + center_x
        )

        rotated_y = (
            math.sin(roll_rad) * (x - center_x)
            + math.cos(roll_rad) * (y - center_y)
            + center_y
        )

        return rotated_x, rotated_y

    @staticmethod
    def focal_from_fov(
        image_width: float,
        image_height: float,
        fov_h_deg: float,
        fov_v_deg: float,
    ) -> tuple[float, float]:
        """FOV ve görüntü boyutundan piksel focal length hesaplar."""
        fx_value = (image_width / 2.0) / math.tan(
            math.radians(fov_h_deg) / 2.0
        )

        fy_value = (image_height / 2.0) / math.tan(
            math.radians(fov_v_deg) / 2.0
        )

        return fx_value, fy_value

    @staticmethod
    def resolve_pitch_down_from_tilt(tilt_deg: float) -> float:
        """Sensör tilt değerini yaklaşık aşağı bakış açısına dönüştürür."""
        if 100.0 <= tilt_deg <= 180.0:
            return tilt_deg - 130.0

        if 45.0 <= tilt_deg <= 100.0:
            return 90.0 - tilt_deg

        if -45.0 <= tilt_deg <= 45.0:
            return tilt_deg

        if -135.0 <= tilt_deg <= -45.0:
            return 90.0 + tilt_deg

        return 0.0

    def validate_input(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> str | None:
        """Mesafe girdilerinin kullanılabilir olup olmadığını kontrol eder."""
        if distance_input.image_width <= 0:
            return "invalid_image_width"

        if distance_input.image_height <= 0:
            return "invalid_image_height"

        if distance_input.w <= 1.0 or distance_input.h <= 1.0:
            return "invalid_bbox_size"

        if distance_input.camera_height_m <= 0.0:
            return "invalid_camera_height"

        if not self.validate_angle(distance_input.fov_h_deg):
            return "invalid_horizontal_fov"

        if not self.validate_angle(distance_input.fov_v_deg):
            return "invalid_vertical_fov"

        finite_values = (
            distance_input.x,
            distance_input.y,
            distance_input.w,
            distance_input.h,
            distance_input.zoom,
            distance_input.tilt_deg,
            distance_input.camera_height_m,
            distance_input.roll_deg,
            distance_input.pitch_deg,
            distance_input.horizon_slope,
        )

        if not all(math.isfinite(value) for value in finite_values):
            return "non_finite_input"

        if distance_input.horizon_y is not None and not math.isfinite(
            distance_input.horizon_y
        ):
            return "non_finite_horizon"

        return None

    def effective_earth_radius(self: DistanceAPI) -> float:
        """Atmosferik kırılma düzeltilmiş Dünya yarıçapını döndürür."""
        return EARTH_RADIUS_M / (1.0 - self.refraction_k)

    def horizon_dip_rad(self: DistanceAPI, camera_height_m: float) -> float:
        """Kamera yüksekliğine göre teorik ufuk çöküşünü hesaplar."""
        return math.sqrt(2.0 * camera_height_m / self.effective_earth_radius())

    def maximum_sea_distance_m(
        self: DistanceAPI, camera_height_m: float
    ) -> float:
        """Kamera yüksekliğine göre teorik maksimum deniz mesafesini hesaplar."""
        return math.sqrt(2.0 * self.effective_earth_radius() * camera_height_m)

    def sea_distance_from_depression(
        self: DistanceAPI, alpha_rad: float, camera_height_m: float
    ) -> float | None:
        """Aşağı bakış açısıyla deniz yüzeyi kesişim mesafesini hesaplar."""
        horizon_dip = self.horizon_dip_rad(camera_height_m)

        if alpha_rad <= horizon_dip:
            return None

        effective_radius = self.effective_earth_radius()
        tangent_value = math.tan(alpha_rad)

        discriminant = (
            effective_radius * tangent_value
        ) ** 2 - 2.0 * effective_radius * camera_height_m

        maximum_distance = self.maximum_sea_distance_m(camera_height_m)

        if discriminant <= 0.0:
            return min(maximum_distance, self.max_distance_m)

        distance = effective_radius * tangent_value - math.sqrt(discriminant)

        if not math.isfinite(distance) or distance <= 0.0:
            return None

        return min(distance, maximum_distance, self.max_distance_m)

    def predict_horizon_y(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> float:
        """Tilt, pitch ve FOV değerleriyle yaklaşık ufuk y konumunu hesaplar."""
        if distance_input.horizon_y is not None:
            return float(distance_input.horizon_y)

        _, fy_value = self.focal_from_fov(
            distance_input.image_width,
            distance_input.image_height,
            distance_input.fov_h_deg,
            distance_input.fov_v_deg,
        )

        center_y = distance_input.image_height / 2.0

        pitch_down_deg = self.resolve_pitch_down_from_tilt(
            distance_input.tilt_deg
        )

        pitch_down_deg += distance_input.pitch_deg

        angle_rad = self.horizon_dip_rad(
            distance_input.camera_height_m
        ) - math.radians(pitch_down_deg)

        angle_rad = self.clamp(angle_rad, -1.2, 1.2)

        horizon_y = center_y + fy_value * math.tan(angle_rad)

        return self.clamp(
            horizon_y,
            distance_input.image_height * 0.02,
            distance_input.image_height * 0.90,
        )

    def get_water_point(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> tuple[float, float]:
        """BBox içinden yaklaşık su hattı noktasını seçer."""
        x1, y1, x2, y2 = self.bbox_to_xyxy(distance_input)

        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)

        frame_area = float(
            distance_input.image_width * distance_input.image_height
        )

        area_ratio = (box_width * box_height) / frame_area

        narrow_fov = (
            distance_input.fov_h_deg <= 12.0
            or distance_input.fov_v_deg <= 9.0
            or distance_input.zoom >= 0.85
        )

        large_close_box = area_ratio >= 0.040

        if narrow_fov and large_close_box:
            waterline_ratio = 0.96
        elif narrow_fov:
            waterline_ratio = 0.92
        elif distance_input.fov_h_deg < 15.0:
            waterline_ratio = 0.88
        else:
            waterline_ratio = 0.90

        water_x = (x1 + x2) / 2.0
        water_y = y1 + waterline_ratio * box_height

        return water_x, water_y

    def calculate_horizontal_line_distance(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> CandidateDistance:
        """Horizon ve bbox su hattından deniz mesafesi hesaplar."""
        _, fy_value = self.focal_from_fov(
            distance_input.image_width,
            distance_input.image_height,
            distance_input.fov_h_deg,
            distance_input.fov_v_deg,
        )

        water_x, water_y = self.get_water_point(distance_input)

        water_x, water_y = self.rotate_pixel(
            water_x,
            water_y,
            distance_input.roll_deg,
            distance_input.image_width,
            distance_input.image_height,
        )

        center_x = distance_input.image_width / 2.0

        horizon_y = self.predict_horizon_y(distance_input)

        horizon_y_at_x = horizon_y + distance_input.horizon_slope * (
            water_x - center_x
        )

        beta_rad = math.atan((water_y - horizon_y_at_x) / fy_value)

        beta_deg = math.degrees(beta_rad)

        if beta_deg <= MIN_BETA_DEG:
            return CandidateDistance(
                distance_m=None,
                confidence=0.0,
                valid=False,
                reason="at_or_beyond_horizon",
            )

        alpha_rad = (
            self.horizon_dip_rad(distance_input.camera_height_m) + beta_rad
        )

        surface_distance = self.sea_distance_from_depression(
            alpha_rad, distance_input.camera_height_m
        )

        if surface_distance is None:
            return CandidateDistance(
                distance_m=None,
                confidence=0.0,
                valid=False,
                reason="horizontal_distance_unavailable",
            )

        if not self.min_distance_m <= surface_distance <= self.max_distance_m:
            return CandidateDistance(
                distance_m=None,
                confidence=0.0,
                valid=False,
                reason="horizontal_distance_out_of_range",
            )

        if distance_input.fov_h_deg < 5.0:
            base_confidence = 0.20
        elif distance_input.fov_h_deg < 15.0:
            base_confidence = 0.30
        else:
            base_confidence = 0.45

        absolute_beta = abs(beta_deg)

        if absolute_beta < 0.03:
            beta_factor = 0.55
        elif absolute_beta < 0.10:
            beta_factor = 0.90
        elif absolute_beta < 0.35:
            beta_factor = 0.75
        else:
            beta_factor = 0.45

        confidence = base_confidence * beta_factor

        if abs(distance_input.roll_deg) > 10.0:
            confidence *= 0.80

        if abs(distance_input.pitch_deg) > 10.0:
            confidence *= 0.85

        return CandidateDistance(
            distance_m=surface_distance,
            confidence=self.clamp(confidence, 0.05, 0.55),
            valid=True,
            reason="horizontal_ok",
        )

    def calculate_butterfly_distance(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> CandidateDistance:
        """BBox piksel boyutu ve FOV ile kelebek geometrisi hesabı yapar."""
        fx_value, fy_value = self.focal_from_fov(
            distance_input.image_width,
            distance_input.image_height,
            distance_input.fov_h_deg,
            distance_input.fov_v_deg,
        )

        aspect_ratio = distance_input.w / max(distance_input.h, 1.0)

        side_score = self.clamp((aspect_ratio - 1.2) / 2.8, 0.0, 1.0)

        bow_score = self.clamp((1.8 - aspect_ratio) / 1.2, 0.0, 1.0)

        width_confidence = self.clamp(0.10 + 0.75 * side_score, 0.10, 0.85)

        height_confidence = self.clamp(0.25 + 0.35 * bow_score, 0.20, 0.60)

        width_distance = (
            self.default_ship_length_m * fx_value / distance_input.w
        )

        height_distance = (
            self.default_ship_height_m * fy_value / distance_input.h
        )

        total_weight = width_confidence + height_confidence

        if total_weight <= 0.0:
            return CandidateDistance(
                distance_m=None,
                confidence=0.0,
                valid=False,
                reason="butterfly_weight_zero",
            )

        distance = (
            width_distance * width_confidence
            + height_distance * height_confidence
        ) / total_weight

        frame_area = float(
            distance_input.image_width * distance_input.image_height
        )

        box_area_ratio = (distance_input.w * distance_input.h) / frame_area

        size_score = self.clamp((box_area_ratio - 0.00012) / 0.014, 0.0, 1.0)

        x1, y1, x2, y2 = self.bbox_to_xyxy(distance_input)

        edge_penalty = 0.0

        if x1 <= 2.0 or y1 <= 2.0:
            edge_penalty += 0.25

        if (
            x2 >= distance_input.image_width - 3.0
            or y2 >= distance_input.image_height - 3.0
        ):
            edge_penalty += 0.25

        bbox_confidence = self.clamp(
            0.30 + 0.70 * size_score - edge_penalty, 0.05, 1.0
        )

        confidence = bbox_confidence * (
            0.35 + 0.65 * max(width_confidence, height_confidence)
        )

        narrow_fov = (
            distance_input.fov_h_deg <= 12.0
            or distance_input.fov_v_deg <= 9.0
            or distance_input.zoom >= 0.85
        )

        very_narrow_fov = (
            distance_input.fov_h_deg <= 9.0 or distance_input.fov_v_deg <= 7.0
        )

        large_close_box = box_area_ratio >= 0.040

        partial_box = distance_input.w < distance_input.image_width * 0.50

        touches_edge = (
            x1 <= 2.0
            or y1 <= 2.0
            or x2 >= distance_input.image_width - 3.0
            or y2 >= distance_input.image_height - 3.0
        )

        penalty_factor = 1.0

        if very_narrow_fov and large_close_box:
            penalty_factor = 0.35
        elif narrow_fov and large_close_box:
            penalty_factor = 0.45
        elif narrow_fov and partial_box:
            penalty_factor = 0.60

        if narrow_fov and touches_edge:
            penalty_factor = min(penalty_factor, 0.55)

        confidence = self.clamp(confidence * penalty_factor, 0.02, 0.95)

        if not (
            self.size_min_distance_m <= distance <= self.size_max_distance_m
        ):
            return CandidateDistance(
                distance_m=None,
                confidence=confidence,
                valid=False,
                reason="butterfly_distance_out_of_range",
            )

        return CandidateDistance(
            distance_m=distance,
            confidence=confidence,
            valid=True,
            reason="butterfly_ok",
        )

    @staticmethod
    def weighted_log_average(
        first_distance: float,
        first_weight: float,
        second_distance: float,
        second_weight: float,
    ) -> float:
        """İki pozitif mesafeyi logaritmik ağırlıkla birleştirir."""
        total_weight = max(first_weight + second_weight, 1e-6)

        return math.exp(
            (
                math.log(max(first_distance, 1.0)) * first_weight
                + math.log(max(second_distance, 1.0)) * second_weight
            )
            / total_weight
        )

    def fuse_distances(
        self: DistanceAPI,
        horizontal_result: CandidateDistance,
        butterfly_result: CandidateDistance,
    ) -> tuple[float | None, float, str]:
        """Horizontal-line ve kelebek sonuçlarını tek mesafeye indirger."""
        if not horizontal_result.valid and not butterfly_result.valid:
            return None, 0.0, "no_valid_distance"

        if horizontal_result.valid and not butterfly_result.valid:
            return (
                horizontal_result.distance_m,
                horizontal_result.confidence,
                "horizon_only",
            )

        if butterfly_result.valid and not horizontal_result.valid:
            return (
                butterfly_result.distance_m,
                butterfly_result.confidence,
                "size_only",
            )

        assert horizontal_result.distance_m is not None
        assert butterfly_result.distance_m is not None

        horizontal_distance = horizontal_result.distance_m
        butterfly_distance = butterfly_result.distance_m

        ratio = max(horizontal_distance, butterfly_distance) / max(
            min(horizontal_distance, butterfly_distance), 1.0
        )

        if ratio > DISAGREEMENT_RATIO_THRESHOLD:
            if butterfly_result.confidence >= 0.60:
                horizontal_weight = 0.18
                butterfly_weight = 0.82
                reason = "size_dominant_high_confidence"
            elif butterfly_result.confidence >= 0.42:
                horizontal_weight = 0.28
                butterfly_weight = 0.72
                reason = "size_dominant_medium_confidence"
            elif butterfly_result.confidence >= 0.30:
                horizontal_weight = 0.38
                butterfly_weight = 0.62
                reason = "size_dominant_low_confidence"
            else:
                horizontal_weight = HORIZON_WEIGHT_DEFAULT
                butterfly_weight = SIZE_WEIGHT_DEFAULT
                reason = "hybrid_disagreement_low_confidence"
        else:
            horizontal_weight = (
                HORIZON_WEIGHT_DEFAULT * horizontal_result.confidence
            )

            butterfly_weight = (
                SIZE_WEIGHT_DEFAULT * butterfly_result.confidence
            )

            reason = "hybrid_agree"

        final_distance = self.weighted_log_average(
            horizontal_distance,
            horizontal_weight,
            butterfly_distance,
            butterfly_weight,
        )

        final_confidence = self.clamp(
            0.5 * butterfly_result.confidence
            + 0.5 * horizontal_result.confidence,
            0.05,
            0.95,
        )

        return final_distance, final_confidence, reason

    def calc_distance(
        self: DistanceAPI, distance_input: DistanceInput
    ) -> DistanceOutput:
        """Tek detection için nihai mesafe sonucunu üretir."""
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

        horizontal_result = self.calculate_horizontal_line_distance(
            distance_input
        )

        butterfly_result = self.calculate_butterfly_distance(distance_input)

        final_distance, final_confidence, reason = self.fuse_distances(
            horizontal_result, butterfly_result
        )

        if final_distance is None:
            return DistanceOutput(
                track_id=distance_input.track_id,
                distance_m=-1.0,
                horizontal_line_distance_m=horizontal_result.distance_m,
                butterfly_distance_m=butterfly_result.distance_m,
                horizontal_confidence=horizontal_result.confidence,
                butterfly_confidence=butterfly_result.confidence,
                confidence=0.0,
                valid=False,
                reason=(
                    f"{reason};"
                    f"{horizontal_result.reason};"
                    f"{butterfly_result.reason}"
                ),
            )

        if not (self.min_distance_m <= final_distance <= self.max_distance_m):
            return DistanceOutput(
                track_id=distance_input.track_id,
                distance_m=-1.0,
                horizontal_line_distance_m=horizontal_result.distance_m,
                butterfly_distance_m=butterfly_result.distance_m,
                horizontal_confidence=horizontal_result.confidence,
                butterfly_confidence=butterfly_result.confidence,
                confidence=0.0,
                valid=False,
                reason="final_distance_out_of_range",
            )

        return DistanceOutput(
            track_id=distance_input.track_id,
            distance_m=final_distance,
            horizontal_line_distance_m=horizontal_result.distance_m,
            butterfly_distance_m=butterfly_result.distance_m,
            horizontal_confidence=horizontal_result.confidence,
            butterfly_confidence=butterfly_result.confidence,
            confidence=final_confidence,
            valid=True,
            reason=reason,
        )
