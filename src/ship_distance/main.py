"""RGB ve termal video akışlarını çalıştıran uygulama giriş noktası."""

from pathlib import Path

import cv2

from config import AppConfig
from distance_butterfly_api import DistanceButterflyApi
from distance_hl_api import DistanceHlApi
from sensor_reader import load_sensor_csv
from video_processor import create_stream_state, process_stream_frame
from visualizer import draw_stream_output, make_side_by_side


try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except (ImportError, RuntimeError):
    CUDA_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
CONFIG = AppConfig.from_yaml(CONFIG_PATH)

RGB_VIDEO_PATH = CONFIG.paths.rgb_video
THERMAL_VIDEO_PATH = CONFIG.paths.thermal_video
CSV_PATH = CONFIG.paths.sensor_csv
OUTPUT_DIR = CONFIG.paths.output_dir
OUTPUT_VIDEO_NAME = (
    f"{CONFIG.record.name}_rgb_thermal_clean_independent_distance.mp4"
)

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
CAMERA_HEIGHT_M = CONFIG.camera.height_m

SAVE_OUTPUT_VIDEO = True
SHOW_WINDOW = True

YOLO_MODEL_PATH = CONFIG.model.yolo_path
YOLO_DEVICE = 0 if CUDA_AVAILABLE else "cpu"
YOLO_HALF = CUDA_AVAILABLE


def get_video_fps(capture: cv2.VideoCapture, fallback: float = 25.0) -> float:
    """Geçerli video FPS değerini döndürür."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if fps > 1.0 else fallback


def main() -> None:
    """RGB ve termal tespit, takip ve mesafe hattını çalıştırır."""
    if not YOLO_AVAILABLE:
        print("Ultralytics kurulu degil.")
        print("Kurulum: pip install ultralytics")
        return

    rgb_file = Path(RGB_VIDEO_PATH)
    thermal_file = Path(THERMAL_VIDEO_PATH)

    if not rgb_file.exists():
        print(f"RGB video bulunamadi: {RGB_VIDEO_PATH}")
        return

    if not thermal_file.exists():
        print(f"Thermal video bulunamadi: {THERMAL_VIDEO_PATH}")
        return

    # Her kanal için kendi sensör satırları kullanılır.
    rgb_sensor_rows = load_sensor_csv(CSV_PATH, channel="rgb")
    thermal_sensor_rows = load_sensor_csv(CSV_PATH, channel="thermal")

    print("YOLO modeli yukleniyor...")
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO modeli hazir.")
    print(f"Device: {YOLO_DEVICE} | Half: {YOLO_HALF}")

    # İki mesafe yöntemi bir kez oluşturulur ve bütün karelerde yeniden kullanılır.
    distance_hl_api = DistanceHlApi()
    distance_butterfly_api = DistanceButterflyApi()

    rgb_cap = cv2.VideoCapture(str(rgb_file))
    thermal_cap = cv2.VideoCapture(str(thermal_file))

    if not rgb_cap.isOpened():
        print(f"RGB video acilamadi: {RGB_VIDEO_PATH}")
        return

    if not thermal_cap.isOpened():
        print(f"Thermal video acilamadi: {THERMAL_VIDEO_PATH}")
        rgb_cap.release()
        return

    video_fps = get_video_fps(rgb_cap)

    output_video_path = Path(OUTPUT_DIR) / OUTPUT_VIDEO_NAME
    writer: cv2.VideoWriter | None = None

    if SAVE_OUTPUT_VIDEO:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (PROCESS_WIDTH * 2, PROCESS_HEIGHT),
        )

    window_name = "RGB + THERMAL DISTANCE"

    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # RGB ve termal akışların takip geçmişleri birbirinden bağımsız tutulur.
    rgb_state = create_stream_state("RGB", "rgb")
    thermal_state = create_stream_state("THERMAL", "thermal")

    frame_index = 0
    previous_tick = cv2.getTickCount()

    print("Video isleniyor...")
    print(f"RGB     : {RGB_VIDEO_PATH}")
    print(f"Thermal : {THERMAL_VIDEO_PATH}")
    print(f"CSV     : {CSV_PATH}")

    if SAVE_OUTPUT_VIDEO:
        print(f"Output  : {output_video_path}")

    print("Cikis: q")

    try:
        while True:
            rgb_ret, rgb_frame = rgb_cap.read()
            thermal_ret, thermal_frame = thermal_cap.read()

            if not rgb_ret and not thermal_ret:
                break

            if not rgb_ret:
                rgb_frame = None

            if not thermal_ret:
                thermal_frame = None

            current_tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(
                current_tick - previous_tick,
                1,
            )
            previous_tick = current_tick

            # API nesneleri ve kamera yüksekliği takip katmanına iletilir.
            rgb_output, rgb_moving = process_stream_frame(
                frame=rgb_frame,
                stream_state=rgb_state,
                sensor_rows=rgb_sensor_rows,
                model=model,
                frame_index=frame_index,
                video_fps=video_fps,
                distance_hl_api=distance_hl_api,
                distance_butterfly_api=distance_butterfly_api,
                camera_height_m=CAMERA_HEIGHT_M,
            )
            thermal_output, thermal_moving = process_stream_frame(
                frame=thermal_frame,
                stream_state=thermal_state,
                sensor_rows=thermal_sensor_rows,
                model=model,
                frame_index=frame_index,
                video_fps=video_fps,
                distance_hl_api=distance_hl_api,
                distance_butterfly_api=distance_butterfly_api,
                camera_height_m=CAMERA_HEIGHT_M,
            )

            draw_stream_output(
                rgb_output,
                rgb_state,
                rgb_sensor_rows,
                fps,
                frame_index,
                video_fps,
                rgb_moving,
            )
            draw_stream_output(
                thermal_output,
                thermal_state,
                thermal_sensor_rows,
                fps,
                frame_index,
                video_fps,
                thermal_moving,
            )

            combined = make_side_by_side(rgb_output, thermal_output)

            if writer is not None:
                writer.write(combined)

            if SHOW_WINDOW:
                cv2.imshow(window_name, combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_index % 200 == 0:
                print(
                    f"frame={frame_index} | "
                    f"rgb_tracks={len(rgb_state['tracks'])} | "
                    f"thermal_tracks={len(thermal_state['tracks'])} | "
                    f"rgb_mode={rgb_state['mode']} | "
                    f"thermal_mode={thermal_state['mode']}"
                )

            frame_index += 1
    finally:
        rgb_cap.release()
        thermal_cap.release()

        if writer is not None:
            writer.release()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    print("Bitti.")

    if SAVE_OUTPUT_VIDEO:
        print(f"Kaydedilen video: {output_video_path}")


if __name__ == "__main__":
    main()
