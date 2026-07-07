# Gemi Tespiti, Takibi ve Mesafe Tahmini

Bu proje, RGB ve termal kamera videoları üzerinden gemi/tekne tespiti, takip işlemi ve kameraya olan yaklaşık mesafe tahmini yapmak için geliştirilmiştir.

Sistem; YOLO tabanlı nesne tespiti, KLT optical flow tabanlı takip, kamera FOV değerleri, tilt bilgisi, horizon geometrisi ve mesafe yumuşatma mantığını birlikte kullanır.

Amaç, RGB ve termal görüntüleri ayrı ayrı işleyerek gemileri tespit etmek, takip etmek ve her gemi için yaklaşık mesafe değerini hesaplamaktır.

## Proje Yapısı

    ship-distance-estimation/
    ├── configs/
    │   └── config.yaml
    ├── precommit/
    │   └── check_comments.py
    ├── src/
    │   └── ship_distance/
    │       ├── __init__.py
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

## Dosyaların Görevleri

### config.py

`configs/config.yaml` dosyasını okur ve projede kullanılan ayarları merkezi olarak yönetir.

Kullanıcıya, bilgisayara veya kayıt dosyasına göre değişebilecek değerler doğrudan kodun içine yazılmaz. Bunun yerine bu değerler config dosyasından alınır.

Bu dosya şu ayarları yönetir:

- kayıt adı
- kayıt klasörü
- RGB video yolu
- termal video yolu
- sensor CSV yolu
- çıktı klasörü
- kamera yüksekliği
- RGB kamera FOV değerleri
- termal kamera FOV değerleri
- YOLO model yolu

Bu yapı sayesinde farklı bir kayıt üzerinde çalışmak için Python kodunu değiştirmek gerekmez. Sadece `configs/config.yaml` dosyası güncellenir.

### sensor_reader.py

Sensor CSV dosyasını okumak ve video zamanına göre gerekli sensor değerlerini hazırlamak için kullanılır.

Bu dosyada şu işlemler yapılır:

- CSV dosyasını okuma
- zaman bilgisini saniyeye çevirme
- RGB ve termal kamera için ilgili kolonları bulma
- FOV değerlerini okuma
- zoom değerlerini okuma
- tilt, roll ve pan değerlerini okuma
- eksik veya hatalı değerleri güvenli şekilde yönetme
- video zamanına göre sensor değerlerini interpolate etme

Video kareleri belirli zamanlara karşılık geldiği için sensor verisinin de o zamana uygun şekilde alınması gerekir. Bu dosya, her frame için en uygun sensor bilgisinin hazırlanmasını sağlar.

### geometry.py

Mesafe tahmini için kullanılan matematiksel işlemleri içerir.

Bu dosya projedeki en kritik matematiksel bölümlerden biridir.

Bu dosyada şu işlemler yapılır:

- kamera FOV değerlerinden focal length hesaplama
- pixel konumunu görüntü açısına çevirme
- tilt bilgisine göre horizon konumunu tahmin etme
- kameranın yüksekliğini kullanarak deniz yüzeyindeki yaklaşık mesafeyi hesaplama
- horizon çizgisini güncelleme
- mesafe değerini okunabilir formata çevirme

Mesafe tahmini yapılırken görüntüdeki nesnenin alt noktası, kamera açısı, kamera yüksekliği ve horizon geometrisi birlikte değerlendirilir.

### detector.py

RGB ve termal görüntüler üzerinde gemi/tekne tespiti yapmak için kullanılır.

Bu dosyada şu işlemler yapılır:

- YOLO modeli ile nesne tespiti
- RGB görüntü üzerinde gemi/tekne adaylarını bulma
- termal görüntü üzerinde aday bölgeleri çıkarma
- gereksiz veya hatalı bounding box’ları filtreleme
- aynı gemiye ait olabilecek kutuları birleştirme
- tespit kalitesini değerlendirme

Bu dosya, görüntüde hangi bölgelerin gemi/tekne olabileceğini belirleyen ana tespit mantığını içerir.

### tracker.py

Tespit edilen gemilerin video boyunca takip edilmesini sağlar.

Bu dosyada şu işlemler yapılır:

- KLT optical flow ile nesne takibi
- kamera hareketini tahmin etme
- eski track’ler ile yeni detection sonuçlarını eşleştirme
- yeni track oluşturma
- kaybolan track’leri pasifleştirme
- takip edilen nesnelerin kutularını güncelleme
- mesafe değerlerindeki ani sıçramaları azaltma
- range-lock mantığı ile mesafe bilgisini daha stabil hale getirme

Tracking işlemi sayesinde her frame’de yeniden tespit yapılamasa bile geminin konumu ve mesafesi takip edilmeye devam eder.

### visualizer.py

İşlenen görüntülerin üzerine çizim yapmak için kullanılır.

Bu dosyada şu işlemler yapılır:

- bounding box çizme
- track ID yazma
- mesafe bilgisini görüntü üzerine ekleme
- horizon çizgisini gösterme
- bilgi paneli oluşturma
- RGB ve termal görüntüleri yan yana birleştirme

Bu dosya doğrudan mesafe hesabı yapmaz. Hesaplanan sonuçların kullanıcıya görsel olarak gösterilmesini sağlar.

### video_processor.py

RGB ve termal videoların frame bazlı işlenmesini yönetir.

Bu dosyada şu işlemler yapılır:

- her video akışı için state oluşturma
- frame bazında sensor bilgisini alma
- tespit işleminin ne zaman çalışacağını belirleme
- tracker güncellemelerini yönetme
- RGB ve termal görüntüleri ayrı ayrı işleme
- her frame için sonuçları hazırlama

Bu dosya; detector, tracker, geometry ve visualizer modülleri arasında bağlantı kurar.

### main.py

Programın ana giriş dosyasıdır.

Bu dosya şu işlemleri yapar:

- config dosyasını yükler
- RGB ve termal video dosyalarını açar
- sensor CSV dosyasını okur
- YOLO modelini yükler
- RGB ve termal stream’leri işler
- çıktı videosunu oluşturur
- sonuçları ekranda gösterir veya dosyaya kaydeder

`main.py` içinde kullanıcıya özel path veya kamera ayarı doğrudan tutulmaz. Bu bilgiler `configs/config.yaml` üzerinden alınır.

## Config Kullanımı

Kullanıcıya veya kayıt dosyasına göre değişen değerler kodun içinde tutulmaz. Bu değerler şu dosyada tutulur:

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

Farklı bir kayıt üzerinde çalışmak için `configs/config.yaml` dosyasında ilgili video yolları, kayıt adı ve sensor CSV yolu değiştirilir.

## Kurulum

Bu projede Python paket yönetimi için `uv` kullanılmaktadır.

Önce repository bilgisayara klonlanır:

    git clone https://github.com/tia88na/ship-distance-estimation.git
    cd ship-distance-estimation

Bağımlılıkları kurmak için:

    uv sync

Bu komut, `pyproject.toml` ve `uv.lock` dosyalarını kullanarak proje için gerekli Python ortamını hazırlar.

## Pre-commit Kullanımı

Projede commit öncesi otomatik kontroller için `pre-commit` kullanılmaktadır.

Pre-commit hook kurmak için:

    uv run pre-commit install

Tüm dosyalar için kontrolleri manuel çalıştırmak için:

    uv run pre-commit run --all-files

Projede kullanılan kontroller:

- yorum yerleşimi kontrolü
- Black format kontrolü
- Ruff lint kontrolü

Bu kontroller sayesinde kod formatı, temel lint kuralları ve yorum yerleşimi commit öncesinde kontrol edilir.

## Çalıştırma

Programı çalıştırmak için:

    PYTHONPATH=src uv run python src/ship_distance/main.py

Program çalışmadan önce `configs/config.yaml` dosyasındaki video yolları, sensor CSV yolu, çıktı klasörü ve model yolu kontrol edilmelidir.

## Çıktı

İşlenen video, `configs/config.yaml` içinde belirtilen çıktı klasörüne kaydedilir.

Çıktı videosunda şu bilgiler görüntülenir:

- RGB görüntü
- termal görüntü
- tespit edilen gemiler
- bounding box bilgileri
- track ID değerleri
- yaklaşık mesafe bilgileri
- horizon çizgisi
- işlem bilgileri

RGB ve termal görüntüler yan yana gösterilerek sonuçların aynı anda incelenmesi sağlanır.

## Notlar

Aşağıdaki dosyalar GitHub’a yüklenmez:

- sanal ortam dosyaları
- cache dosyaları
- video çıktıları
- büyük model dosyaları
- `.pt`, `.onnx`, `.engine` model dosyaları
- geçici çıktı klasörleri

Bu dosyalar `.gitignore` içinde hariç tutulmuştur.

## Genel Özet

Bu proje, RGB ve termal kamera kayıtlarını kullanarak deniz üzerindeki gemi/tekne nesnelerini tespit eder, takip eder ve kameraya olan yaklaşık mesafelerini hesaplar.

Kod yapısı modüllere ayrılmıştır. Kullanıcıya özel ayarlar config dosyasında tutulur. Paket yönetimi için `uv`, commit öncesi kontroller için `pre-commit` kullanılmaktadır.
