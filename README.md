# Gemi Tespiti, Takibi ve Mesafe Tahmini

Bu proje, RGB ve termal kamera videoları üzerinden gemi/tekne tespiti, takip işlemi ve kameraya olan mesafe tahmini yapmak için geliştirilmiştir.

Sistem; YOLO tabanlı nesne tespiti, KLT optical flow tabanlı takip, kamera FOV değerleri, tilt bilgisi, horizon geometrisi ve mesafe yumuşatma mantığını birlikte kullanır.

Amaç, RGB ve termal görüntüleri ayrı ayrı işleyerek gemileri tespit etmek, takip etmek ve her gemi için yaklaşık mesafe değerini hesaplamaktır.

## Proje Yapısı

```text
ship-distance-estimation/
├── configs/
│   └── config.yaml
├── precommit/
│   └── check_comments.py
├── src/
│   └── ship_distance/
│       ├── config.py
│       ├── detector.py
│       ├── geometry.py
│       ├── main.py
│       ├── sensor_reader.py
│       ├── tracker.py
│       ├── video_processor.py
│       └── visualizer.py
├── pyproject.toml
├── uv.lock
└── README.md

