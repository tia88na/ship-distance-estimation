"""Tekne tespiti, termal ön işleme ve detection birleştirme yardımcıları.

Bu dosya RGB ve termal görüntülerde tekne/gemi tespiti için kullanılan ana
yardımcı fonksiyonları içerir. YOLO bölgesel inference işlemi, bounding box
filtreleme, su hattı noktası hesaplama ve aynı hedefe ait detection kutularını
birleştirme adımları burada yönetilir.

Bu sürümde ana hedef detection kalitesini iyileştirmektir. Mesafe hesabı
değiştirilmeden önce yanlış bbox kaynaklı hatalar azaltılır. Özellikle dar
zoom kayıtlarında alt kısımdaki bina/çatı/sahil parçalarının ve termal
görüntüde gemi parçası gibi görünen kenar yapılarının gemi olarak algılanması
engellenmeye çalışılır.
"""

# -----------------------------------------------------------------------------
# İçe aktarımlar
# Bu modül yalnızca detection, görüntü işleme ve gerekli geometri yardımcılarını kullanır.
# -----------------------------------------------------------------------------
from typing import TypeAlias, cast

import cv2
from geometry import horizon_y_at, sea_distance_from_image_point
import numpy as np
from sensor_reader import SensorRow


# -----------------------------------------------------------------------------
# Ortak tip takma adları
# Tuple ve sözlük tabanlı veri yapılarının anlamını fonksiyon imzalarında açık tutar.
# -----------------------------------------------------------------------------
Box: TypeAlias = tuple[float, float, float, float]
IntBox: TypeAlias = tuple[int, int, int, int]
Region: TypeAlias = tuple[str, int, int, int, int]
Detection: TypeAlias = dict[str, object]
HorizonState: TypeAlias = dict[str, object]

# -----------------------------------------------------------------------------
# Sabit işlem çözünürlüğü
# Detector, tracker ve geometri katmanları aynı koordinat sistemini kullanır.
# -----------------------------------------------------------------------------
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

# -----------------------------------------------------------------------------
# YOLO inference temel ayarları
# Bu değerler model çağrısında tüm region türleri için ortak tabanı oluşturur.
# -----------------------------------------------------------------------------
YOLO_IOU_THRES = 0.50
YOLO_DEVICE = 0

# Full frame detection daha yüksek confidence ile çalışır. Full frame içinde
# bina/çatı gibi false positive riski daha yüksektir.
# Full-frame inference daha fazla arka plan içerdiği için confidence eşiği daha sıkıdır.
# Bu eşik modelin sınıf confidence filtresidir; son geometrik filtreler ayrıca uygulanır.
YOLO_CONF_FULL = 0.42

# Ufuk bandı ve tile bölgeleri küçük/uzak hedefler için kullanıldığı için daha
# düşük confidence ile ikinci seviyede taranır.
# Deep region'larda hedef ekranda daha küçük olabileceğinden daha düşük confidence kabul edilir.
# Düşük eşik tek başına sonucu kabul ettirmez; filter_detection son kontrolü yine yapar.
YOLO_CONF_DEEP = 0.20

# Full ve deep inference giriş boyutları ayrı tutulur.
# Deep aramada daha büyük imgsz, küçük hedef detaylarının korunmasına yardımcı olur.
YOLO_IMGSZ_FULL = 960
YOLO_IMGSZ_DEEP = 1536

# Termal full-frame eşiği RGB'ye göre biraz daha düşük tutulur ama önceki
# sürüme göre sertleştirildi. Dar FOV termalde gemi olmayan sıcak/soğuk
# parçalar kolayca boat sanılabildiği için full-frame eşiği fazla düşük
# olmamalıdır.
# Termal görüntü için confidence/imgsz ayarları RGB'den bağımsız tutulur.
# Termal kontrast ve false-positive davranışı farklı olduğu için ayrı tuning değerleri kullanılır.
THERMAL_YOLO_CONF_FULL = 0.40
THERMAL_YOLO_CONF_DEEP = 0.24
THERMAL_YOLO_IMGSZ_FULL = 1280
THERMAL_YOLO_IMGSZ_DEEP = 1536


# -----------------------------------------------------------------------------
# Su hattı tahmini
# BBox tabanından biraz yukarıda seçilen nokta deniz-temas bölgesini yaklaşık temsil eder.
# -----------------------------------------------------------------------------
WATERLINE_RATIO_NORMAL = 0.90
WATERLINE_RATIO_ZOOM = 0.86

# -----------------------------------------------------------------------------
# Kamerayı taşıyan geminin kendi yapısını elemek için kullanılan geometrik eşikler
# Bu kontroller özellikle görüntünün altındaki çok büyük ve çok yakın kutulara yöneliktir.
# -----------------------------------------------------------------------------
OWN_SHIP_BOTTOM_RATIO = 0.90
OWN_SHIP_MIN_HEIGHT_RATIO = 0.30
OWN_SHIP_MAX_AREA_RATIO = 0.40
OWN_SHIP_NEAR_DISTANCE_M = 12.0
OWN_SHIP_NEAR_BOTTOM_RATIO = 0.82

# -----------------------------------------------------------------------------
# Detection birleştirme eşikleri
# Aynı hedefin farklı region'lardan tekrar gelmesi durumunda geometrik yakınlık değerlendirilir.
# -----------------------------------------------------------------------------
MERGE_IOU_THRES = 0.22
MERGE_INSIDE_THRES = 0.55
MERGE_HORIZONTAL_OVERLAP_THRES = 0.35
MERGE_VERTICAL_GAP_PX = 180
MERGE_CENTER_DISTANCE_RATIO = 0.82

# -----------------------------------------------------------------------------
# FOV sınıfları
# Search region derinliği ve minimum detection boyutu gibi kararlar bu seviyelere göre değişir.
# -----------------------------------------------------------------------------
STRONG_ZOOM_FOV_H_DEG = 5.0
ZOOM_FOV_H_DEG = 15.0
MID_FOV_H_DEG = 30.0

# Dar FOV test videolarında hedefler genellikle ufuk çizgisine yakın deniz
# bandında olur. Çok aşağıdaki kutular çoğunlukla sahil, bina veya çatı
# false positive üretir.
STRONG_ZOOM_SEARCH_DEPTH_PX = 230
ZOOM_SEARCH_DEPTH_PX = 300
MID_FOV_SEARCH_DEPTH_PX = 390

# Detection filtreleme için search depth'ten biraz daha toleranslı sınır.
STRONG_ZOOM_VALID_DEPTH_PX = 260
ZOOM_VALID_DEPTH_PX = 340
MID_FOV_VALID_DEPTH_PX = 440

# -----------------------------------------------------------------------------
# Alt bölge false-positive filtreleri
# Özellikle kıyı, bina ve çatı parçalarının boat olarak kabul edilmesini azaltmak için kullanılır.
# -----------------------------------------------------------------------------
BOTTOM_STRUCTURE_Y1_RATIO = 0.62
BOTTOM_STRUCTURE_WATER_RATIO = 0.76
BOTTOM_STRUCTURE_AREA_RATIO = 0.012

# Termal dar FOV görüntülerde ekran kenarında çıkan küçük/orta kutular çoğu
# zaman gerçek gemi değil, yakın geminin/kıyı yapısının parçalarıdır. Gerçek
# gemi ekrana kenardan giriyorsa birkaç frame sonra daha tam görüneceği için
# ilk fragment kutularını elemek daha güvenlidir.
# -----------------------------------------------------------------------------
# Termal fragment filtreleri
# Dar FOV'da ekran kenarındaki kısmi yapıların yanlış gemi detection'ı olma riski ayrıca kontrol edilir.
# -----------------------------------------------------------------------------
THERMAL_EDGE_MARGIN_RATIO = 0.025
THERMAL_EDGE_FRAGMENT_MAX_WIDTH_RATIO = 0.30
THERMAL_EDGE_FRAGMENT_MAX_AREA_RATIO = 0.070
THERMAL_EDGE_FRAGMENT_MAX_ASPECT = 4.2
THERMAL_FRAGMENT_LOW_CONF = 0.78

# Termal dar FOV'da full-frame kaynaklı düşük güvenli küçük parçalar false
# positive üretmeye çok yatkındır.
THERMAL_NARROW_FULL_MIN_CONF = 0.55
THERMAL_NARROW_DEEP_MIN_CONF = 0.30



# =============================================================================
# get_sensor_fov_h
# Yatay FOV, detection bölgelerinin ve zoom seviyesinin yorumlanmasında temel girdidir.
# Sensör satırında eksik ya da hatalı veri bulunması ihtimaline karşı güvenli okuma yapılır.
# Fallback değerinin amacı akışı durdurmak yerine geniş açı varsayımıyla devam etmektir.
# =============================================================================
def get_sensor_fov_h(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden yatay FOV değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Yatay FOV değeri. Okunamazsa geniş açı varsayımı döner.
    """
    # Sensör verisinin beklenmeyen tipte olma ihtimaline karşı dönüşüm kontrollü şekilde denenir.
    try:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return float(sensor_info.get("fov_h", 65.7))
    # Dönüşüm başarısız olduğunda güvenli varsayılan değerle akışın devam etmesi sağlanır.
    except (TypeError, ValueError):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return 65.7



# =============================================================================
# get_sensor_fov_v
# Dikey FOV özellikle dar görüş açısının belirlenmesinde yardımcı bir sensör parametresidir.
# Bu yardımcı fonksiyon dönüşüm hatalarını tek noktada toplar.
# Böylece çağıran fonksiyonların ayrı ayrı hata yönetimi yapmasına gerek kalmaz.
# =============================================================================
def get_sensor_fov_v(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden dikey FOV değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Dikey FOV değeri. Okunamazsa geniş açı varsayımı döner.
    """
    # Sensör verisinin beklenmeyen tipte olma ihtimaline karşı dönüşüm kontrollü şekilde denenir.
    try:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return float(sensor_info.get("fov_v", 39.9))
    # Dönüşüm başarısız olduğunda güvenli varsayılan değerle akışın devam etmesi sağlanır.
    except (TypeError, ValueError):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return 39.9



# =============================================================================
# get_sensor_zoom
# Zoom değeri, FOV ile birlikte dar görüntü koşullarını değerlendirmek için kullanılır.
# Geçersiz veri durumunda nötr kabul edilen 0.0 değeri kullanılır.
# Bu fonksiyon yalnızca sensör verisini normalize eder; detection kararı burada verilmez.
# =============================================================================
def get_sensor_zoom(sensor_info: SensorRow) -> float:
    """Sensör bilgisinden zoom değerini güvenli şekilde okur.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Zoom değeri. Okunamazsa 0.0 döner.
    """
    # Sensör verisinin beklenmeyen tipte olma ihtimaline karşı dönüşüm kontrollü şekilde denenir.
    try:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return float(sensor_info.get("zoom", 0.0))
    # Dönüşüm başarısız olduğunda güvenli varsayılan değerle akışın devam etmesi sağlanır.
    except (TypeError, ValueError):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return 0.0



# =============================================================================
# is_narrow_fov
# Dar FOV kararı tek bir değere değil yatay FOV, dikey FOV ve zoom kombinasyonuna dayanır.
# Bu karar özellikle termal detection filtrelerinin ne kadar sıkı uygulanacağını etkiler.
# Fonksiyon yalnızca boolean durum üretir ve asıl filtreleme başka fonksiyonlarda yapılır.
# =============================================================================
def is_narrow_fov(sensor_info: SensorRow) -> bool:
    """Mevcut sensör bilgisinin dar FOV / zoomlu olup olmadığını kontrol eder.

    Args:
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        FOV dar veya zoom yüksekse True.
    """
    fov_h = get_sensor_fov_h(sensor_info)
    fov_v = get_sensor_fov_v(sensor_info)
    zoom = get_sensor_zoom(sensor_info)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return fov_h < ZOOM_FOV_H_DEG or fov_v < 9.0 or zoom >= 0.85



# =============================================================================
# get_class_name
# Ultralytics model sınıf isimleri farklı sürümlerde dict veya list biçiminde gelebilir.
# Bu yardımcı iki veri biçimini de destekleyerek sınıf adını tek biçime çevirir.
# Bulunamayan sınıflarda ID string olarak döndürülerek akışın kırılması engellenir.
# =============================================================================
def get_class_name(model: object, cls_id: int) -> str:
    """YOLO modelindeki class id değerini okunabilir sınıf adına çevirir.

    Args:
        model: Ultralytics YOLO model nesnesi.
        cls_id: Model sonucundan gelen sınıf id değeri.

    Returns:
        Sınıf adı bulunursa sınıf adı, bulunamazsa id değerinin string hali.
    """
    names = model.names

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if isinstance(names, dict):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return str(names.get(cls_id, str(cls_id)))

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if isinstance(names, list) and 0 <= cls_id < len(names):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return str(names[cls_id])

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return str(cls_id)



# =============================================================================
# calculate_iou
# IoU, iki bounding box'ın ne kadar aynı alanı temsil ettiğini ölçen temel eşleşme metriğidir.
# Bu değer hem detection birleştirmede hem tracker eşleştirmesinde kullanılabilir.
# Alan hesaplarında minimum 1.0 kullanılması sıfıra bölünme riskini önler.
# =============================================================================
def calculate_iou(box_a: Box, box_b: Box) -> float:
    """İki bounding box arasındaki IoU değerini hesaplar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return inter_area / max(area_a + area_b - inter_area, 1.0)



# =============================================================================
# overlap_ratio_small_inside_large
# Bu metrik klasik IoU'dan farklı olarak küçük kutunun büyük kutu içinde kalma oranını ölçer.
# Aynı geminin farklı crop/region inference sonuçlarında farklı boyutta kutulanmasını yakalamaya yardım eder.
# Özellikle biri diğerinin içinde kalan detection'ların tek hedef olarak birleştirilmesinde kullanılır.
# =============================================================================
def overlap_ratio_small_inside_large(box_a: Box, box_b: Box) -> float:
    """Küçük kutunun büyük kutu içinde ne kadar kaldığını hesaplar."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return inter_area / min(area_a, area_b)



# =============================================================================
# horizontal_overlap_ratio
# Gemiler yatay doğrultuda uzun görünebildiği için yatay örtüşme ayrıca değerlendirilir.
# Bu metrik dikey fark olsa bile aynı hedefe ait parçalı kutuları ilişkilendirmeye yardımcı olur.
# Normalize edilmiş sonuç, farklı kutu genişliklerinde karşılaştırılabilir kalır.
# =============================================================================
def horizontal_overlap_ratio(box_a: Box, box_b: Box) -> float:
    """İki kutunun yatay eksende ne kadar örtüştüğünü hesaplar."""
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))

    width_a = max(1.0, ax2 - ax1)
    width_b = max(1.0, bx2 - bx1)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return inter_w / min(width_a, width_b)



# =============================================================================
# vertical_gap_px
# İki kutu üst üste gelmiyorsa aralarındaki dikey boşluk piksel cinsinden hesaplanır.
# Dikey olarak örtüşen kutularda boşluk sıfır kabul edilir.
# Bu değer detection birleştirme kararının geometrik koşullarından biridir.
# =============================================================================
def vertical_gap_px(box_a: Box, box_b: Box) -> float:
    """İki kutu arasındaki dikey boşluğu piksel cinsinden hesaplar."""
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if ay2 < by1:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return by1 - ay2

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if by2 < ay1:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return ay1 - by2

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return 0.0



# =============================================================================
# center_x_distance_ratio
# Yatay merkez uzaklığı kutu genişliğine göre normalize edilir.
# Böylece çözünürlükten ve mutlak piksel boyutundan daha az etkilenen bir yakınlık ölçüsü elde edilir.
# Aynı gemiye ait kutuların merkezlerinin aşırı uzaklaşması bu metrikle engellenir.
# =============================================================================
def center_x_distance_ratio(box_a: Box, box_b: Box) -> float:
    """İki kutu merkezinin yatay uzaklığını normalize eder."""
    ax1, _, ax2, _ = box_a
    bx1, _, bx2, _ = box_b

    center_a = (ax1 + ax2) / 2.0
    center_b = (bx1 + bx2) / 2.0

    width_a = max(1.0, ax2 - ax1)
    width_b = max(1.0, bx2 - bx1)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return abs(center_a - center_b) / max(width_a, width_b)



# =============================================================================
# box_to_int
# OpenCV çizim ve görünür alan işlemleri integer koordinatlarla daha doğal çalışır.
# Float bbox koordinatları burada yalnızca yuvarlanarak integer biçime çevrilir.
# Koordinatların görüntü sınırlarına kırpılması ayrı visible_box fonksiyonunda yapılır.
# =============================================================================
def box_to_int(box: Box) -> IntBox:
    """Float koordinatlı kutuyu integer koordinatlı kutuya çevirir."""
    x1, y1, x2, y2 = box

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))



# =============================================================================
# visible_box
# Track kutuları geçici olarak frame dışına taşabildiği için görünür bölüm burada sınırlandırılır.
# Bu işlem orijinal track kutusunu değiştirmez; yalnızca kullanılabilir görünür koordinat üretir.
# Sonuç çizim, alan hesabı ve optical-flow maskesi gibi işlemlerde güvenle kullanılabilir.
# =============================================================================
def visible_box(box: Box) -> IntBox:
    """Bounding box koordinatlarını görüntü sınırlarına kırpar."""
    x1, y1, x2, y2 = box_to_int(box)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return (
        max(0, x1),
        max(0, y1),
        min(PROCESS_WIDTH - 1, x2),
        min(PROCESS_HEIGHT - 1, y2),
    )



# =============================================================================
# clamp_track_box
# Tracking sırasında prediction veya optical flow kutuyu frame dışına taşıyabilir.
# Tamamen sert biçimde frame içine kırpmak yerine sınırlı taşmaya izin verilir.
# Bu yaklaşım hedef görüntüden geçici olarak çıktığında track geometrisinin hemen bozulmasını azaltır.
# =============================================================================
def clamp_track_box(box: Box) -> Box:
    """Track kutusunu aşırı taşmalara karşı güvenli aralıkta tutar."""
    x1, y1, x2, y2 = box

    x1 = max(-2.0 * PROCESS_WIDTH, min(3.0 * PROCESS_WIDTH, x1))
    y1 = max(-2.0 * PROCESS_HEIGHT, min(3.0 * PROCESS_HEIGHT, y1))
    x2 = max(x1 + 2.0, min(3.0 * PROCESS_WIDTH, x2))
    y2 = max(y1 + 2.0, min(3.0 * PROCESS_HEIGHT, y2))

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return x1, y1, x2, y2



# =============================================================================
# get_waterline_ratio
# BBox içindeki su hattı tahmini normal ve zoomlu görüntülerde farklı oranda seçilir.
# Dar FOV'da kutunun alt kısmının kompozisyonu değişebileceği için ayrı oran kullanılır.
# Bu fonksiyon yalnızca oran seçer; gerçek piksel koordinatı sonraki yardımcıda hesaplanır.
# =============================================================================
def get_waterline_ratio(sensor_info: SensorRow) -> float:
    """Bounding box içindeki su hattı oranını FOV değerine göre seçer."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if get_sensor_fov_h(sensor_info) < ZOOM_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return WATERLINE_RATIO_ZOOM

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return WATERLINE_RATIO_NORMAL



# =============================================================================
# get_water_point_from_box
# Mesafe geometrisinde geminin suyla temas ettiği yaklaşık nokta kullanılmak istenir.
# X koordinatı bbox merkezinden, Y koordinatı ise seçilen waterline oranından türetilir.
# Detection ve tracker aynı yardımcıyı kullandığı için su noktası tanımı tutarlı kalır.
# =============================================================================
def get_water_point_from_box(
    box: Box, sensor_info: SensorRow
) -> tuple[float, float]:
    """Bounding box içinden mesafe hesabında kullanılacak su hattı noktasını alır."""
    x1, y1, x2, y2 = box
    height = max(1.0, y2 - y1)

    water_x = (x1 + x2) / 2.0
    water_y = y1 + get_waterline_ratio(sensor_info) * height

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return water_x, water_y



# =============================================================================
# is_own_ship_box
# Kamerayı taşıyan platformun kendi gövdesi görüntünün alt kısmında yanlış detection üretebilir.
# Boyut, alt kenara yakınlık ve toplam alan gibi kaba geometrik sinyaller birlikte değerlendirilir.
# Amaç uzak hedefleri değil, kameraya çok yakın büyük yapı parçalarını erken elemekdir.
# =============================================================================
def is_own_ship_box(box: Box) -> bool:
    """Detection kutusunun kameraya ait gemi parçası olup olmadığını kontrol eder."""
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height
    frame_area = PROCESS_WIDTH * PROCESS_HEIGHT

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        y2 >= PROCESS_HEIGHT * OWN_SHIP_BOTTOM_RATIO
        and height >= PROCESS_HEIGHT * OWN_SHIP_MIN_HEIGHT_RATIO
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if area >= frame_area * OWN_SHIP_MAX_AREA_RATIO:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        y2 >= PROCESS_HEIGHT * 0.97
        and height >= PROCESS_HEIGHT * 0.18
        and width >= PROCESS_WIDTH * 0.35
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return False



# =============================================================================
# max_search_depth_below_horizon
# YOLO deep-search bölgesinin ufkun ne kadar altına ineceği FOV'a göre sınırlandırılır.
# Dar FOV'da çok aşağı bölgeleri taramak kıyı ve bina false-positive riskini artırabilir.
# bottom_deep modu bilinçli olarak tüm yüksekliğe erişebilen daha geniş arama davranışıdır.
# =============================================================================
def max_search_depth_below_horizon(sensor_info: SensorRow, mode: str) -> int:
    """YOLO aramasının ufuk altında ne kadar derine ineceğini belirler."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if mode == "bottom_deep":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return PROCESS_HEIGHT

    fov_h = get_sensor_fov_h(sensor_info)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return STRONG_ZOOM_SEARCH_DEPTH_PX

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < ZOOM_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return ZOOM_SEARCH_DEPTH_PX

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < MID_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return MID_FOV_SEARCH_DEPTH_PX

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return PROCESS_HEIGHT



# =============================================================================
# max_valid_depth_below_horizon
# Arama yapılabilen alan ile geçerli detection kabul edilen alan aynı olmak zorunda değildir.
# Bu fonksiyon detection filtresinde kullanılacak daha toleranslı geçerlilik derinliğini belirler.
# FOV daraldıkça hedefin beklenen deniz bandı daha kontrollü tutulur.
# =============================================================================
def max_valid_depth_below_horizon(sensor_info: SensorRow) -> int:
    """Detection su hattı için izin verilen maksimum ufuk altı derinliği."""
    fov_h = get_sensor_fov_h(sensor_info)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return STRONG_ZOOM_VALID_DEPTH_PX

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < ZOOM_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return ZOOM_VALID_DEPTH_PX

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < MID_FOV_H_DEG:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return MID_FOV_VALID_DEPTH_PX

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return PROCESS_HEIGHT



# =============================================================================
# is_probable_bottom_structure
# RGB görüntülerde alt bölgede bulunan bina, çatı ve kıyı yapıları boat sınıfına benzeyebilir.
# Bu fonksiyon bbox geometrisi ile ufka göre konumu birlikte kullanarak bu yapıları ayıklar.
# Termal görüntü için ayrı filtreler bulunduğundan termal detection burada doğrudan reddedilmez.
# =============================================================================
def is_probable_bottom_structure(
    det: Detection, sensor_info: SensorRow, horizon_state: HorizonState
) -> bool:
    """RGB görüntüde bina/çatı/sahil parçası olabilecek kutuları ayıklar."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if det.get("channel") == "thermal":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    aspect = width / max(height, 1.0)

    water_y = float(det["water_y"])
    y_horizon = horizon_y_at(horizon_state, float(det["water_x"]))
    depth_below_horizon = water_y - y_horizon

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        get_sensor_fov_h(sensor_info) < ZOOM_FOV_H_DEG
        and y1 > PROCESS_HEIGHT * BOTTOM_STRUCTURE_Y1_RATIO
        and water_y > PROCESS_HEIGHT * BOTTOM_STRUCTURE_WATER_RATIO
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        y1 > PROCESS_HEIGHT * BOTTOM_STRUCTURE_Y1_RATIO
        and area_ratio > BOTTOM_STRUCTURE_AREA_RATIO
        and (aspect < 0.55 or aspect > 9.5)
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if depth_below_horizon > max_valid_depth_below_horizon(sensor_info):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if y2 > PROCESS_HEIGHT * 0.92 and area_ratio > 0.018:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return False



# =============================================================================
# is_thermal_edge_fragment
# Dar FOV termalde geminin veya kıyı yapısının küçük bir parçası ekran kenarında boat olarak algılanabilir.
# Kenar teması, kutu boyutu, aspect ratio ve confidence birlikte değerlendirilir.
# Büyük ve muhtemelen tam gemi kutularını korumak için filtre yalnızca belirli fragment geometrilerine uygulanır.
# =============================================================================
def is_thermal_edge_fragment(det: Detection, sensor_info: SensorRow) -> bool:
    """Dar FOV termalde ekran kenarındaki parça kutuları ayıklar.

    Args:
        det: Kontrol edilecek detection.
        sensor_info: Mevcut frame'e ait sensör bilgisi.

    Returns:
        Detection yakın gemi/yapı parçası gibi görünüyorsa True.
    """
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if det.get("channel") != "thermal":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if not is_narrow_fov(sensor_info):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / max(height, 1.0)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    width_ratio = width / PROCESS_WIDTH
    conf = float(det.get("conf", 0.0))
    source = str(det.get("source", ""))

    left_edge = x1 <= PROCESS_WIDTH * THERMAL_EDGE_MARGIN_RATIO
    right_edge = x2 >= PROCESS_WIDTH * (1.0 - THERMAL_EDGE_MARGIN_RATIO)
    touches_side = left_edge or right_edge

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if not touches_side:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Çok büyük ana gemi kutularını değil, kenarda görünen küçük/orta parçaları
    # hedefliyoruz.
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        width_ratio <= THERMAL_EDGE_FRAGMENT_MAX_WIDTH_RATIO
        and area_ratio <= THERMAL_EDGE_FRAGMENT_MAX_AREA_RATIO
        and aspect <= THERMAL_EDGE_FRAGMENT_MAX_ASPECT
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Full-frame kaynaklı ve yüksek güvenli olmayan kenar kutuları da risklidir.
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if source == "full" and conf < THERMAL_FRAGMENT_LOW_CONF:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Kenara yapışık, alt/orta bölgede ve dikey-kompakt görünen kutular termal
    # yapısal parçalar olabilir.
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        y1 > PROCESS_HEIGHT * 0.25
        and width_ratio <= 0.36
        and area_ratio <= 0.090
        and aspect <= 3.2
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return False



# =============================================================================
# is_thermal_low_quality_fragment
# Termal dar FOV koşullarında düşük confidence'lı küçük detection'lar daha sık false-positive üretir.
# Full-frame ve deep-region kaynakları için farklı minimum confidence mantığı korunur.
# Ek küçük-kutu kontrolü, düşük kaliteli fragment sonuçlarını daha erken eler.
# =============================================================================
def is_thermal_low_quality_fragment(
    det: Detection, sensor_info: SensorRow
) -> bool:
    """Dar FOV termalde düşük güvenli küçük parça detectionlarını ayıklar."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if det.get("channel") != "thermal":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if not is_narrow_fov(sensor_info):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_ratio = (width * height) / float(PROCESS_WIDTH * PROCESS_HEIGHT)
    width_ratio = width / PROCESS_WIDTH
    conf = float(det.get("conf", 0.0))
    source = str(det.get("source", ""))

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if source == "full" and conf < THERMAL_NARROW_FULL_MIN_CONF:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if source != "full" and conf < THERMAL_NARROW_DEEP_MIN_CONF:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        area_ratio < 0.012
        and width_ratio < 0.22
        and conf < THERMAL_FRAGMENT_LOW_CONF
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return False



# =============================================================================
# filter_detection
# Bu fonksiyon YOLO'dan gelen ham bir kutunun projede kullanılabilir olup olmadığına karar veren ana kapıdır.
# Boyut, aspect ratio, kanal tipi, ufuk konumu ve yakın platform geometrisi sırasıyla kontrol edilir.
# Erken return yaklaşımı sayesinde başarısız adaylar pahalı veya gereksiz sonraki kontrollere taşınmaz.
# =============================================================================
def filter_detection(
    det: Detection, sensor_info: SensorRow, horizon_state: HorizonState
) -> bool:
    """Ham detection sonucunu geometri ve boyut kurallarına göre filtreler."""
    box = cast(Box, det["box"])
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1
    area = width * height
    frame_area = PROCESS_WIDTH * PROCESS_HEIGHT
    is_thermal = det.get("channel") == "thermal"
    fov_h = get_sensor_fov_h(sensor_info)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_thermal:
        min_width = 14
        min_height = 8
    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        min_width = 6
        min_height = 5

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if width < min_width or height < min_height:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_thermal:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if fov_h < ZOOM_FOV_H_DEG:
            min_area = 95
        # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
        elif fov_h < MID_FOV_H_DEG:
            min_area = 130
        # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
        else:
            min_area = 180
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif fov_h < STRONG_ZOOM_FOV_H_DEG:
        min_area = 26
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif fov_h < ZOOM_FOV_H_DEG:
        min_area = 40
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif fov_h < MID_FOV_H_DEG:
        min_area = 90
    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        min_area = 220

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if area < min_area:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if area > frame_area * 0.45:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    aspect = width / max(height, 1.0)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_thermal:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if aspect < 0.45 or aspect > 16.0:
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if y1 > PROCESS_HEIGHT * 0.82:
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if float(det["water_y"]) > PROCESS_HEIGHT * 0.94:
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if is_thermal_edge_fragment(det, sensor_info):
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if is_thermal_low_quality_fragment(det, sensor_info):
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if aspect < 0.35 or aspect > 16.0:
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if fov_h < ZOOM_FOV_H_DEG and aspect < 0.55:
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_own_ship_box(box):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    y_horizon = horizon_y_at(horizon_state, float(det["water_x"]))

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if float(det["water_y"]) <= y_horizon + 0.5:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_probable_bottom_structure(det, sensor_info, horizon_state):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    result = sea_distance_from_image_point(
        float(det["water_x"]),
        float(det["water_y"]),
        sensor_info,
        horizon_state,
    )

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        result["valid"]
        and float(result["distance"]) < OWN_SHIP_NEAR_DISTANCE_M
        and y2 > PROCESS_HEIGHT * OWN_SHIP_NEAR_BOTTOM_RATIO
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return False

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return True



# =============================================================================
# build_search_regions
# Tek bir full-frame inference yerine gerektiğinde ufuk, deniz bandı ve tile bölgeleri de üretilir.
# Bölge boyutları FOV'a göre değişerek uzak/küçük gemilerin daha büyük giriş boyutuyla görülmesini sağlar.
# full_only modunda ek region üretmeden doğrudan tam görüntü taramasıyla yetinilir.
# =============================================================================
def build_search_regions(
    sensor_info: SensorRow, horizon_state: HorizonState, mode: str
) -> list[Region]:
    """YOLO'nun çalışacağı görüntü bölgelerini üretir."""
    y_h = int(max(10, min(PROCESS_HEIGHT - 40, float(horizon_state["y"]))))
    fov_h = get_sensor_fov_h(sensor_info)

    regions: list[Region] = [("full", 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT)]

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if mode == "full_only":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return regions

    search_depth = max_search_depth_below_horizon(sensor_info, mode)
    y_bottom = min(PROCESS_HEIGHT, y_h + search_depth)

    regions.append(
        (
            "horizon_strip",
            0,
            max(0, y_h - 70),
            PROCESS_WIDTH,
            min(PROCESS_HEIGHT, y_h + 170),
        )
    )

    regions.append(("sea_band", 0, max(0, y_h - 30), PROCESS_WIDTH, y_bottom))

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if mode == "bottom_deep":
        regions.extend(
            [
                (
                    "bottom_75",
                    0,
                    int(PROCESS_HEIGHT * 0.25),
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT,
                ),
                (
                    "bottom_55",
                    0,
                    int(PROCESS_HEIGHT * 0.45),
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT,
                ),
            ]
        )

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if fov_h < STRONG_ZOOM_FOV_H_DEG:
        tile_w = 420
        step = 260
        tile_y0 = max(0, y_h - 55)
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif fov_h < ZOOM_FOV_H_DEG:
        tile_w = 520
        step = 340
        tile_y0 = max(0, y_h - 45)
    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        tile_w = 640
        step = 460
        tile_y0 = max(0, y_h - 30)

    x_value = 0

    # Gerekli search region'lar görüntü boyunca üretilene kadar döngü devam eder.
    while x_value < PROCESS_WIDTH:
        x2 = min(PROCESS_WIDTH, x_value + tile_w)
        x1 = max(0, x2 - tile_w)

        regions.append((f"tile_{x1}", x1, tile_y0, x2, y_bottom))

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if x2 >= PROCESS_WIDTH:
            break

        x_value += step

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return regions



# =============================================================================
# prepare_frame_for_detection
# RGB görüntü doğrudan kullanılabilirken termal görüntü için kontrast iyileştirme uygulanır.
# Normalize + CLAHE + hafif blur sırası termal yapıları YOLO için daha belirgin hale getirir.
# Çıktı tekrar BGR'ye çevrilerek model giriş biçimi iki kanal türünde aynı tutulur.
# =============================================================================
def prepare_frame_for_detection(frame: np.ndarray, channel: str) -> np.ndarray:
    """Detection öncesi frame'i kanal tipine göre hazırlar."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if channel != "thermal":
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray.astype(np.uint8))
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)



# =============================================================================
# region_confidence
# Farklı search region türleri aynı minimum confidence eşiğini kullanmak zorunda değildir.
# Full frame daha sıkı, küçük/uzak hedef arayan region'lar daha toleranslı çalışabilir.
# Dar FOV termalde alt confidence sınırları ayrıca yükseltilerek fragment false-positive'leri azaltılır.
# =============================================================================
def region_confidence(
    region_name: str,
    base_full_conf: float,
    base_deep_conf: float,
    channel: str,
    sensor_info: SensorRow,
) -> float:
    """Region adına göre YOLO confidence eşiğini seçer."""
    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if region_name == "full":
        conf = base_full_conf
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif region_name in {"horizon_strip", "sea_band"}:
        conf = base_deep_conf
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif region_name.startswith("tile_"):
        conf = base_deep_conf
    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        conf = base_full_conf

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if channel == "thermal" and is_narrow_fov(sensor_info):
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if region_name == "full":
            # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
            return max(conf, THERMAL_NARROW_FULL_MIN_CONF)

        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return max(conf, THERMAL_NARROW_DEEP_MIN_CONF)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return conf



# =============================================================================
# run_yolo_region
# YOLO inference işleminin tek bir crop/region üzerinde çalıştığı temel fonksiyon budur.
# Region içindeki bbox koordinatları daha sonra full-frame koordinat sistemine geri taşınır.
# Ham model çıktıları kullanılmadan önce ortak filter_detection süzgecinden geçirilir.
# =============================================================================
def run_yolo_region(
    frame: np.ndarray,
    model: object,
    region: Region,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    conf_thres: float,
    imgsz: int,
    channel: str = "rgb",
) -> list[Detection]:
    """Tek bir region üzerinde YOLO inference çalıştırır."""
    region_name, x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if crop.size == 0 or (y2 - y1) < 24:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return []

    detections: list[Detection] = []

    # Asıl YOLO inference çağrısı burada yapılır.
    # Crop doğrudan modele verilir; confidence, imgsz ve IoU değerleri çağrı bazında belirlenir.
    # classes=[8] COCO boat sınıfına odaklanarak diğer sınıfların gereksiz sonuç üretmesini engeller.
    # max_det termal görüntüde daha düşük tutularak gürültülü aday sayısı sınırlandırılır.
    results = model.predict(
        crop,
        conf=conf_thres,
        imgsz=imgsz,
        iou=YOLO_IOU_THRES,
        verbose=False,
        classes=[8],
        max_det=18 if channel == "thermal" else 60,
        device=YOLO_DEVICE,
    )

    # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
    for result in results:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if result.boxes is None:
            continue

        # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            name = get_class_name(model, cls_id)

            # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
            if name != "boat":
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()

            abs_box = (
                float(max(0, bx1 + x1)),
                float(max(0, by1 + y1)),
                float(min(PROCESS_WIDTH - 1, bx2 + x1)),
                float(min(PROCESS_HEIGHT - 1, by2 + y1)),
            )

            water_x, water_y = get_water_point_from_box(abs_box, sensor_info)

            det: Detection = {
                "box": abs_box,
                "conf": conf,
                "water_x": water_x,
                "water_y": water_y,
                "source": region_name,
                "channel": channel,
            }

            # Model confidence kontrolünü geçmiş olsa bile detection doğrudan kabul edilmez.
            # Projeye özel geometrik ve ufuk tabanlı filtrelerin tamamı bu noktada son kez uygulanır.
            # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
            if filter_detection(det, sensor_info, horizon_state):
                detections.append(det)

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return detections



# =============================================================================
# detection_quality_score
# Aynı hedef için birden fazla region farklı bbox üretebildiğinden bir kalite skoru gerekir.
# Confidence ana bileşendir; alan, waterline konumu ve region kaynağı ek sinyal olarak kullanılır.
# Bu skor gerçek sınıflandırma confidence'ı değildir; yalnızca birleştirme önceliği için kullanılır.
# =============================================================================
def detection_quality_score(det: Detection) -> float:
    """Detection kutusunun birleştirme sırasındaki kalite skorunu hesaplar."""
    x1, y1, x2, y2 = cast(Box, det["box"])

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height

    area_score = min(area / (PROCESS_WIDTH * PROCESS_HEIGHT), 0.35)
    water_score = float(det["water_y"]) / PROCESS_HEIGHT

    source = str(det.get("source", ""))
    region_bonus = 0.0

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if source == "horizon_strip":
        region_bonus = 0.25
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif source == "sea_band":
        region_bonus = 0.18
    # Önceki koşul sağlanmadığında alternatif eşik veya görünüm durumu burada değerlendirilir.
    elif source.startswith("tile_"):
        region_bonus = 0.12

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return (
        2.2 * float(det["conf"])
        + 1.2 * area_score
        + 0.8 * water_score
        + region_bonus
    )



# =============================================================================
# same_vessel
# İki detection'ın aynı fiziksel gemiye ait olup olmadığı birden fazla geometrik ölçütle değerlendirilir.
# Önce güçlü örtüşme kontrolleri yapılır, ardından yatay örtüşme + dikey boşluk + merkez uzaklığı kombinasyonu denenir.
# Bir koşul yeterince güçlü ise iki kutu aynı hedef kabul edilir.
# =============================================================================
def same_vessel(det_a: Detection, det_b: Detection) -> bool:
    """İki detection'ın aynı gemiye ait olup olmadığını kontrol eder."""
    box_a = cast(Box, det_a["box"])
    box_b = cast(Box, det_b["box"])

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if calculate_iou(box_a, box_b) > MERGE_IOU_THRES:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if overlap_ratio_small_inside_large(box_a, box_b) > MERGE_INSIDE_THRES:
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if (
        horizontal_overlap_ratio(box_a, box_b)
        >= MERGE_HORIZONTAL_OVERLAP_THRES
        and vertical_gap_px(box_a, box_b) <= MERGE_VERTICAL_GAP_PX
        and center_x_distance_ratio(box_a, box_b)
        <= MERGE_CENTER_DISTANCE_RATIO
    ):
        # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
        return True

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return False



# =============================================================================
# merge_detection_group
# Aynı hedefe ait olduğu belirlenen detection grubundan tek temsilci kutu seçilir.
# Kutu en yüksek kalite skoruna sahip detection'dan alınırken confidence grup içindeki maksimum değerden korunur.
# Water-point seçilen ana kutu üzerinden yeniden hesaplanarak alanların birbiriyle tutarlı kalması sağlanır.
# =============================================================================
def merge_detection_group(
    group: list[Detection], sensor_info: SensorRow
) -> Detection:
    """Aynı hedefe ait detection grubunu tek detection'a indirger."""
    main_det = max(group, key=detection_quality_score)

    water_x, water_y = get_water_point_from_box(
        cast(Box, main_det["box"]), sensor_info
    )

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return {
        "box": main_det["box"],
        "conf": max(float(det["conf"]) for det in group),
        "water_x": water_x,
        "water_y": water_y,
        "source": main_det["source"],
        "channel": main_det.get("channel", "rgb"),
    }



# =============================================================================
# merge_same_vessel_detections
# Farklı crop ve tile inference sonuçları aynı gemiyi birden fazla kez döndürebilir.
# Önce benzer detection'lar gruplandırılır, sonra her grup tek detection'a indirgenir.
# Son bir tekrar kontrolüyle gruplar arası kalan olası duplicate kutular da elenir.
# =============================================================================
def merge_same_vessel_detections(
    detections: list[Detection], sensor_info: SensorRow
) -> list[Detection]:
    """Aynı gemiye ait tekrar detection kutularını birleştirir."""
    detections = sorted(detections, key=detection_quality_score, reverse=True)

    groups: list[list[Detection]] = []

    # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
    for det in detections:
        placed = False

        # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
        for group in groups:
            # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
            if any(same_vessel(det, other) for other in group):
                group.append(det)
                placed = True
                break

        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if not placed:
            groups.append([det])

    merged = [merge_detection_group(group, sensor_info) for group in groups]

    merged = sorted(merged, key=detection_quality_score, reverse=True)

    kept: list[Detection] = []

    # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
    for det in merged:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if not any(same_vessel(det, kept_det) for kept_det in kept):
            kept.append(det)

    # İkinci duplicate kontrolünden sonra yalnızca benzersiz kabul edilen detection'lar dışarı verilir.
    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return kept



# =============================================================================
# detect_boats
# Bu fonksiyon detector modülünün dışarıya açılan ana detection akışıdır.
# Kanal ve FOV'a göre imgsz/confidence seçilir, search region'lar taranır ve sonuçlar birleştirilir.
# Termal görüntüde ek ön işleme ve sonuç sayısı sınırlaması uygulanırken RGB akışı kendi eşiklerini kullanır.
# =============================================================================
def detect_boats(
    frame: np.ndarray,
    model: object,
    sensor_info: SensorRow,
    horizon_state: HorizonState,
    mode: str,
    channel: str = "rgb",
) -> list[Detection]:
    """RGB veya termal frame üzerinde tekne detection akışını çalıştırır."""
    is_thermal = channel == "thermal"
    fov_h = get_sensor_fov_h(sensor_info)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_thermal:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if fov_h < MID_FOV_H_DEG:
            imgsz = THERMAL_YOLO_IMGSZ_DEEP
        # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
        else:
            imgsz = THERMAL_YOLO_IMGSZ_FULL

        conf_full = THERMAL_YOLO_CONF_FULL
        conf_deep = THERMAL_YOLO_CONF_DEEP
        # Termal kanal YOLO'ya verilmeden önce kontrast iyileştirme hattından geçirilir.
        # Orijinal frame değiştirilmez; yalnızca inference için ayrı hazırlanmış görüntü kullanılır.
        detection_frame = prepare_frame_for_detection(frame, "thermal")
    # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
    else:
        # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
        if fov_h < MID_FOV_H_DEG:
            imgsz = YOLO_IMGSZ_DEEP
        # Yukarıdaki özel durumların dışında kalan genel akış bu dalda devam eder.
        else:
            imgsz = YOLO_IMGSZ_FULL

        conf_full = YOLO_CONF_FULL
        conf_deep = YOLO_CONF_DEEP
        detection_frame = frame

    # Mevcut FOV, ufuk konumu ve detection modu için taranacak region listesi bir kez oluşturulur.
    # Aynı liste bu çağrı boyunca kullanıldığı için region geometrisi tutarlı kalır.
    regions = build_search_regions(sensor_info, horizon_state, mode)
    detections: list[Detection] = []

    # Koleksiyondaki öğeler tek tek değerlendirilerek aynı detection kuralları uygulanır.
    for region in regions:
        region_name = region[0]
        conf = region_confidence(
            region_name, conf_full, conf_deep, channel, sensor_info
        )

        detections.extend(
            run_yolo_region(
                detection_frame,
                model,
                region,
                sensor_info,
                horizon_state,
                conf,
                imgsz,
                channel=channel,
            )
        )


    # Tüm region'lardan gelen ham detection'lar tek listede toplandıktan sonra duplicate hedefler birleştirilir.
    # Böylece tracker aynı fiziksel gemi için birden fazla yeni track açmak zorunda kalmaz.
    merged = merge_same_vessel_detections(detections, sensor_info)

    # Bu koşul, ilgili geometrik/sensör durumu sağlandığında özel kontrol dalını devreye alır.
    if is_thermal:
        merged = sorted(merged, key=detection_quality_score, reverse=True)[:6]

    # Hesaplanan/kararlaştırılan değer bu noktada çağıran katmana sonuç olarak döndürülür.
    return merged
