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
Dosyaların Görevleri
config.py

configs/config.yaml dosyasını okur.

Kullanıcıdan kullanıcıya değişebilecek ayarlar burada yönetilir:

kayıt adı
RGB video yolu
termal video yolu
sensor CSV yolu
çıktı klasörü
kamera yüksekliği
RGB kamera FOV değerleri
termal kamera FOV değerleri
YOLO model yolu
sensor_reader.py

Sensor CSV dosyasını okur ve video zamanına göre gerekli değerleri hazırlar.

Bu dosyada şu işlemler yapılır:

zaman bilgisini saniyeye çevirme
FOV kolonlarını bulma
zoom değerlerini okuma
tilt, roll ve pan değerlerini okuma
RGB ve termal kamera için ayrı sensor bilgisi hazırlama
iki zaman arası sensor değerlerini interpolate etme
geometry.py

Mesafe hesabı için kullanılan matematiksel işlemleri içerir.

Bu dosyada şu işlemler yapılır:

FOV değerlerinden pixel cinsinden focal length hesaplama
tilt bilgisine göre horizon konumu tahmini
pixel satırını açıya çevirme
kamera yüksekliği ve horizon geometrisi ile mesafe hesaplama
deniz düzlemi üzerinde yaklaşık uzaklık hesaplama
mesafe formatlama

Bu proje için en kritik matematiksel işlemler bu dosyadadır.

detector.py

Gemi/tekne tespit işlemlerini içerir.

Bu dosyada şu işlemler yapılır:

YOLO ile gemi tespiti
RGB görüntü üzerinde tespit
termal görüntü için ön işleme
termal aday bölgeleri filtreleme
hatalı bounding box’ları eleme
aynı gemiye ait kutuları birleştirme
tracker.py

Takip ve mesafe stabilizasyonu işlemlerini içerir.

Bu dosyada şu işlemler yapılır:

KLT optical flow ile takip
kamera hareketini tahmin etme
tespit edilen nesneleri mevcut track’lerle eşleştirme
yeni track oluşturma
kaybolan track’leri silme
ani mesafe sıçramalarını engelleme
range-lock mantığı ile mesafeyi yumuşatma
visualizer.py

Görüntü üzerine çizim yapan fonksiyonları içerir.

Bu dosyada şu işlemler yapılır:

bounding box çizme
track ID yazma
mesafe bilgisini görüntüye yazma
bilgi paneli çizme
RGB ve termal görüntüleri yan yana birleştirme
video_processor.py

RGB ve termal görüntülerin frame bazlı işlenmesini yönetir.

Bu dosyada şu işlemler yapılır:

her stream için state oluşturma
frame bazlı sensor bilgisi alma
detection zamanlamasını yönetme
tracking güncelleme
RGB ve termal tarafı ayrı ayrı işleme
main.py

Programın ana giriş dosyasıdır.

Bu dosya:

config dosyasını yükler
RGB ve termal videoları açar
sensor CSV dosyasını okur
YOLO modelini yükler
RGB ve termal görüntüleri işler
çıktı videosunu kaydeder veya ekranda gösterir
Config Kullanımı

Kullanıcıdan kullanıcıya değişen değerler kodun içinde tutulmaz. Bu değerler şu dosyada tutulur:

configs/config.yaml

Örnek config yapısı:

record:
  name: "2025_05_25-21_38_27"
  root: "/home/tuana/records_work/Records_all"

paths:
  rgb_video: "/home/tuana/records_work/Records_all/2025_05_25-21_38_27/rgb.mp4"
  thermal_video: "/home/tuana/records_work/Records_all/2025_05_25-21_38_27/thermal.mp4"
  sensor_csv: "/home/tuana/records_work/Records_all/2025_05_25-21_38_27/sensor_data.csv"
  output_dir: "/home/tuana/video_distance_outputs"

camera:
  height_m: 10.0
  rgb_fov_h_deg: 65.7
  rgb_fov_v_deg: 39.9
  thermal_fov_h_deg: 32.4
  thermal_fov_v_deg: 24.6

model:
  yolo_path: "yolov8x.pt"

Farklı bir kayıt üzerinde çalışmak için configs/config.yaml içindeki video yolları ve kayıt adı değiştirilir.

Kurulum

Bu projede Python paket yönetimi için uv kullanılmaktadır.

Bağımlılıkları kurmak için:

uv sync
Pre-commit Kullanımı

Pre-commit, kod commit edilmeden önce otomatik kontrol yapar.

Pre-commit hook kurmak için:

uv run pre-commit install

Tüm dosyalar için kontrol çalıştırmak için:

uv run pre-commit run --all-files

Projede kullanılan kontroller:

yorum yerleşimi kontrolü
Black format kontrolü
Ruff lint kontrolü
Çalıştırma

Programı çalıştırmak için:

PYTHONPATH=src uv run python src/ship_distance/main.py
Çıktı

İşlenen video, configs/config.yaml içinde belirtilen çıktı klasörüne kaydedilir.

Çıktı videosunda RGB ve termal görüntüler yan yana gösterilir. Tespit edilen gemiler için bounding box, track ID ve mesafe bilgisi görüntü üzerine yazılır.

Notlar

Aşağıdaki dosyalar GitHub’a yüklenmez:

sanal ortam dosyaları
video çıktıları
büyük model dosyaları
.pt, .onnx, .engine model dosyaları
cache dosyaları

Bu dosyalar .gitignore içinde hariç tutulmuştur.
