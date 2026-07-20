"""Detection sonuçlarını iki bağımsız mesafe API'sine bağlayan sınıf.

RGB ve termal detection kutuları ortak ``xyxy`` bbox formatına dönüştürülür.
Horizontal-line hesabı ``DistanceHlApi``, bbox boyutu hesabı ise
``DistanceButterflyApi`` tarafından yapılır.

İki API birbirinden bağımsızdır. Sonuç seçimi ve gerektiğinde birleştirme bu
entegrasyon katmanında yapılır; mesafe formülleri burada tekrar edilmez.
"""

from __future__ import annotations

import math
from typing import Literal


try:
    from ship_distance.distance_butterfly_api import (
        DistanceButterflyApi,
        DistanceButterflyResult,
    )
    from ship_distance.distance_hl_api import DistanceHlApi, DistanceHlResult
except ImportError:
    from distance_butterfly_api import (
        DistanceButterflyApi,
        DistanceButterflyResult,
    )
    from distance_hl_api import DistanceHlApi, DistanceHlResult

from object_track.data_types import (
    ImageDetectionResult,
    ThermalObjectDetectionResult,
)
from params import PARAM_CAMERA_DEPTH_DIST_MAX_KM, PARAM_CAMERAS


StreamType = Literal["rgb", "thermal"]
Box = tuple[float, float, float, float]


class DistanceCalculator:
    """Detection sonuçlarının kameraya olan yaklaşık mesafesini hesaplar.

    Mevcut çağrı arayüzü korunur. Her detection için HL ve Butterfly API'leri
    ayrı çalıştırılır. Geçerli sonuç detection nesnesinin ``dist`` alanına metre
    cinsinden yazılır; geçerli sonuç üretilemezse ``-1`` kullanılır.
    """

    RGB_IMAGE_WIDTH = 1920
    RGB_IMAGE_HEIGHT = 1080

    THERMAL_IMAGE_WIDTH = 1280
    THERMAL_IMAGE_HEIGHT = 1024

    def __init__(self: DistanceCalculator) -> None:
        """Mesafe API nesnelerini ve üst mesafe sınırını oluşturur."""
        self.distance_hl_api = DistanceHlApi()
        self.distance_butterfly_api = DistanceButterflyApi()
        self.max_distance_m = PARAM_CAMERA_DEPTH_DIST_MAX_KM * 1000.0

    @staticmethod
    def update_roll_pitch_due_to_eis(
        camera_type: str, roll: float, pitch: float, stream: str
    ) -> tuple[float, float]:
        """EIS etkisine göre roll ve pitch değerlerini düzeltir.

        EIS açıkken beş derecenin altındaki küçük IMU hareketleri sıfırlanır.
        Daha büyük açılarda EIS tarafından telafi edildiği varsayılan beş
        derecelik bölüm çıkarılır.
        """
        eis_key = f"CAMERA_{stream}_EIS_ON"

        if not PARAM_CAMERAS[camera_type][eis_key]:
            return roll, pitch

        if abs(roll) < 5.0:
            roll = 0.0
        elif roll < -5.0:
            roll += 5.0
        else:
            roll -= 5.0

        if abs(pitch) < 5.0:
            pitch = 0.0
        elif pitch < -5.0:
            pitch += 5.0
        else:
            pitch -= 5.0

        return roll, pitch

    @staticmethod
    def find_thermal_detection(
        track_id: int, tors: list[ThermalObjectDetectionResult]
    ) -> ThermalObjectDetectionResult | None:
        """Track ID ile eşleşen termal detection sonucunu döndürür."""
        for thermal_detection in tors:
            if thermal_detection.track_id == track_id:
                return thermal_detection

        return None

    @staticmethod
    def create_box(*, x: float, y: float, width: float, height: float) -> Box:
        """Merkez koordinatı ve boyuttan ``xyxy`` bbox üretir.

        Mevcut entegrasyonda ``x`` ve ``y`` bbox merkezidir. Alt noktanın önceki
        kodda ``y + h / 2`` ile bulunması da bu koordinat düzenini doğrular.
        """
        half_width = float(width) / 2.0
        half_height = float(height) / 2.0

        return (
            float(x) - half_width,
            float(y) - half_height,
            float(x) + half_width,
            float(y) + half_height,
        )

    @staticmethod
    def combine_distance_results(
        hl_result: DistanceHlResult, butterfly_result: DistanceButterflyResult
    ) -> float | None:
        """İki bağımsız API sonucundan tek bir mesafe seçer.

        Tek yöntem geçerliyse o sonuç kullanılır. İki yöntem de geçerliyse,
        çarpansal mesafe hatalarını daha dengeli ele almak için güven skorlarıyla
        ağırlıklandırılmış logaritmik ortalama alınır. Bu birleştirme iki API'nin
        dışında tutulur.
        """
        hl_valid = hl_result.valid and hl_result.distance_m is not None
        butterfly_valid = (
            butterfly_result.valid and butterfly_result.distance_m is not None
        )

        if hl_valid and not butterfly_valid:
            return float(hl_result.distance_m)

        if butterfly_valid and not hl_valid:
            return float(butterfly_result.distance_m)

        if not hl_valid or not butterfly_valid:
            return None

        hl_distance = max(float(hl_result.distance_m), 1.0)
        butterfly_distance = max(float(butterfly_result.distance_m), 1.0)

        hl_weight = max(float(hl_result.confidence), 0.05)
        butterfly_weight = max(float(butterfly_result.confidence), 0.05)
        total_weight = hl_weight + butterfly_weight

        return math.exp(
            (
                math.log(hl_distance) * hl_weight
                + math.log(butterfly_distance) * butterfly_weight
            )
            / total_weight
        )

    def calculate_box_distance(
        self: DistanceCalculator,
        *,
        track_id: int,
        box: Box,
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
        tilt_deg: float,
        camera_height_m: float,
        zoom: float,
        roll_deg: float,
        pitch_deg: float,
    ) -> float | None:
        """Tek bbox için iki API'yi çalıştırıp son mesafeyi döndürür."""
        hl_result = self.distance_hl_api.calc_distance(
            track_id=track_id,
            box=box,
            image_width=image_width,
            image_height=image_height,
            fov_h_deg=float(fov_h_deg),
            fov_v_deg=float(fov_v_deg),
            tilt_deg=float(tilt_deg),
            camera_height_m=float(camera_height_m),
            zoom=float(zoom),
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
        )

        butterfly_result = self.distance_butterfly_api.calc_distance(
            track_id=track_id,
            box=box,
            image_width=image_width,
            image_height=image_height,
            fov_h_deg=float(fov_h_deg),
            fov_v_deg=float(fov_v_deg),
            zoom=float(zoom),
        )

        distance_m = self.combine_distance_results(hl_result, butterfly_result)

        # Projenin tanımlı azami mesafe sınırı entegrasyon çıkışında uygulanır.
        if distance_m is None or not 0.0 < distance_m <= self.max_distance_m:
            return None

        return distance_m

    def calculate_distance(
        self: DistanceCalculator,
        camera_type: str,
        detections: list[ImageDetectionResult],
        image_detection_results_night: list[ImageDetectionResult],
        tors: list[ThermalObjectDetectionResult],
        fov_h_rgb: float,
        fov_v_rgb: float,
        fov_h_thr: float,
        fov_v_thr: float,
        tilt: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        zoom_rgb: float = 0.0,
        zoom_thr: float = 0.0,
    ) -> None:
        """RGB ve termal detection sonuçlarının mesafesini günceller.

        ``Obstacle`` sınıfı için aynı track ID'ye ait termal kutu kullanılır.
        Diğer detection'lar RGB kutusu üzerinden hesaplanır. Gece detection
        sonuçları aynı track ID'ye ait ana detection mesafesini paylaşır.
        """
        camera_parameters = PARAM_CAMERAS[camera_type]
        camera_height_m = float(camera_parameters["CAMERA_HEIGHT"])

        roll_rgb, pitch_rgb = self.update_roll_pitch_due_to_eis(
            camera_type, roll, pitch, "RGB"
        )
        roll_thr, pitch_thr = self.update_roll_pitch_due_to_eis(
            camera_type, roll, pitch, "THR"
        )

        for detection in detections:
            if detection.name == "Obstacle":
                thermal_detection = self.find_thermal_detection(
                    detection.track_id, tors
                )

                if thermal_detection is None:
                    detection.dist = -1
                    continue

                box = self.create_box(
                    x=thermal_detection.x,
                    y=thermal_detection.y,
                    width=thermal_detection.w,
                    height=thermal_detection.h,
                )
                distance_m = self.calculate_box_distance(
                    track_id=detection.track_id,
                    box=box,
                    image_width=self.THERMAL_IMAGE_WIDTH,
                    image_height=self.THERMAL_IMAGE_HEIGHT,
                    fov_h_deg=fov_h_thr,
                    fov_v_deg=fov_v_thr,
                    tilt_deg=tilt,
                    camera_height_m=camera_height_m,
                    zoom=zoom_thr,
                    roll_deg=roll_thr,
                    pitch_deg=pitch_thr,
                )
            else:
                box = self.create_box(
                    x=detection.x,
                    y=detection.y,
                    width=detection.w,
                    height=detection.h,
                )
                distance_m = self.calculate_box_distance(
                    track_id=detection.track_id,
                    box=box,
                    image_width=self.RGB_IMAGE_WIDTH,
                    image_height=self.RGB_IMAGE_HEIGHT,
                    fov_h_deg=fov_h_rgb,
                    fov_v_deg=fov_v_rgb,
                    tilt_deg=tilt,
                    camera_height_m=camera_height_m,
                    zoom=zoom_rgb,
                    roll_deg=roll_rgb,
                    pitch_deg=pitch_rgb,
                )

            detection.dist = distance_m if distance_m is not None else -1

        # Gece sonuçları, aynı track ID için hesaplanan ana detection mesafesini
        # kullanır. Böylece aynı hedef için ikinci kez API çağrısı yapılmaz.
        distance_by_track_id = {
            detection.track_id: detection.dist for detection in detections
        }

        for night_detection in image_detection_results_night:
            if night_detection.track_id in distance_by_track_id:
                night_detection.dist = distance_by_track_id[
                    night_detection.track_id
                ]
