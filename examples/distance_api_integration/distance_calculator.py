"""Detection sonuçlarını modüler mesafe API'sine bağlayan hesaplama sınıfı.

Bu modül, RGB ve termal detection sonuçlarını DistanceInput formatına çevirir
ve bağımsız DistanceAPI backend'inden mesafe sonucu alır.

Mesafe matematiği bu dosyada tekrar edilmez. Horizontal-line, bbox-size ve
hibrit birleştirme işlemleri distance_api.py içinde gerçekleştirilir.
"""

from __future__ import annotations

from typing import Literal


try:
    from .distance_api import DistanceAPI, DistanceInput
except ImportError:
    from distance_api import DistanceAPI, DistanceInput

from object_track.data_types import (
    ImageDetectionResult,
    ThermalObjectDetectionResult,
)
from params import PARAM_CAMERA_DEPTH_DIST_MAX_KM, PARAM_CAMERAS


StreamType = Literal["rgb", "thermal"]


class DistanceCalculator:
    """Detection sonuçlarının kameraya olan yaklaşık mesafesini hesaplar.

    RGB ve termal detection bilgileri, kamera parametreleriyle birlikte
    DistanceAPI sınıfına gönderilir. Hesaplanan mesafe metre cinsinden
    detection nesnesinin ``dist`` alanına yazılır.

    Bütün açı değerleri derece cinsindendir.
    """

    RGB_IMAGE_WIDTH = 1920
    RGB_IMAGE_HEIGHT = 1080

    THERMAL_IMAGE_WIDTH = 1280
    THERMAL_IMAGE_HEIGHT = 1024

    def __init__(self: DistanceCalculator) -> None:
        """Mesafe hesaplama API nesnesini oluşturur."""
        self.distance_api = DistanceAPI(
            max_distance_m=PARAM_CAMERA_DEPTH_DIST_MAX_KM * 1000.0
        )

    @staticmethod
    def update_roll_pitch_due_to_eis(
        camera_type: str, roll: float, pitch: float, stream: str
    ) -> tuple[float, float]:
        """EIS etkisine göre roll ve pitch değerlerini düzeltir.

        EIS açıkken beş derecenin altındaki küçük IMU hareketleri sıfırlanır.
        Daha büyük açılarda ise EIS tarafından telafi edildiği varsayılan
        beş derecelik bölüm çıkarılır.

        Args:
            camera_type: PARAM_CAMERAS içindeki kamera tipi.
            roll: Ham roll açısı.
            pitch: Ham pitch açısı.
            stream: ``RGB`` veya ``THR`` kamera akışı.

        Returns:
            Düzeltilmiş roll ve pitch değerleri.
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
        """Track ID ile eşleşen termal detection sonucunu bulur.

        Args:
            track_id: Aranan detection track kimliği.
            tors: Termal detection sonuçları.

        Returns:
            Eşleşen termal detection veya eşleşme yoksa ``None``.
        """
        for thermal_detection in tors:
            if thermal_detection.track_id == track_id:
                return thermal_detection

        return None

    @staticmethod
    def create_distance_input(
        *,
        track_id: int,
        x: float,
        y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        fov_h_deg: float,
        fov_v_deg: float,
        zoom: float,
        tilt_deg: float,
        camera_height_m: float,
        roll_deg: float,
        pitch_deg: float,
        stream: StreamType,
    ) -> DistanceInput:
        """Detection ve kamera bilgilerini DistanceInput nesnesine dönüştürür.

        ``x`` ve ``y`` değerlerinin bbox merkez koordinatları olduğu kabul
        edilir. Bu kabul, Alperen'in mevcut kodunda x değerinin doğrudan,
        alt noktanın ise ``y + h / 2`` ile hesaplanmasına dayanır.

        Returns:
            Mesafe API'sine gönderilecek girdi nesnesi.
        """
        return DistanceInput(
            track_id=track_id,
            x=float(x),
            y=float(y),
            w=float(width),
            h=float(height),
            image_width=image_width,
            image_height=image_height,
            fov_h_deg=float(fov_h_deg),
            fov_v_deg=float(fov_v_deg),
            tilt_deg=float(tilt_deg),
            camera_height_m=float(camera_height_m),
            zoom=float(zoom),
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            stream=stream,
        )

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
        """Detection sonuçları için mesafe hesaplar.

        Mevcut fonksiyon imzasının temel parametreleri korunmuştur. RGB ve
        termal zoom değerleri sona isteğe bağlı parametre olarak eklenmiştir;
        bu nedenle eski çağrılar değişmeden çalışmaya devam eder.

        Args:
            camera_type: PARAM_CAMERAS içindeki kamera tipi.
            detections: Ana detection sonuçları.
            image_detection_results_night: Gece detection sonuçları.
            tors: Termal detection sonuçları.
            fov_h_rgb: RGB yatay FOV değeri.
            fov_v_rgb: RGB dikey FOV değeri.
            fov_h_thr: Termal yatay FOV değeri.
            fov_v_thr: Termal dikey FOV değeri.
            tilt: Kamera tilt değeri.
            roll: Kamera roll değeri.
            pitch: Kamera pitch değeri.
            zoom_rgb: RGB akışının normalize zoom değeri.
            zoom_thr: Termal akışın normalize zoom değeri.
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

                distance_input = self.create_distance_input(
                    track_id=detection.track_id,
                    x=thermal_detection.x,
                    y=thermal_detection.y,
                    width=thermal_detection.w,
                    height=thermal_detection.h,
                    image_width=self.THERMAL_IMAGE_WIDTH,
                    image_height=self.THERMAL_IMAGE_HEIGHT,
                    fov_h_deg=fov_h_thr,
                    fov_v_deg=fov_v_thr,
                    zoom=zoom_thr,
                    tilt_deg=tilt,
                    camera_height_m=camera_height_m,
                    roll_deg=roll_thr,
                    pitch_deg=pitch_thr,
                    stream="thermal",
                )
            else:
                distance_input = self.create_distance_input(
                    track_id=detection.track_id,
                    x=detection.x,
                    y=detection.y,
                    width=detection.w,
                    height=detection.h,
                    image_width=self.RGB_IMAGE_WIDTH,
                    image_height=self.RGB_IMAGE_HEIGHT,
                    fov_h_deg=fov_h_rgb,
                    fov_v_deg=fov_v_rgb,
                    zoom=zoom_rgb,
                    tilt_deg=tilt,
                    camera_height_m=camera_height_m,
                    roll_deg=roll_rgb,
                    pitch_deg=pitch_rgb,
                    stream="rgb",
                )

            distance_result = self.distance_api.calc_distance(distance_input)

            if distance_result.valid:
                detection.dist = distance_result.distance_m
            else:
                detection.dist = -1

        distance_by_track_id = {
            detection.track_id: detection.dist for detection in detections
        }

        for night_detection in image_detection_results_night:
            if night_detection.track_id in distance_by_track_id:
                night_detection.dist = distance_by_track_id[
                    night_detection.track_id
                ]
