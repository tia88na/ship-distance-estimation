"""Sensör CSV dosyasını okuma ve zamana göre ara değer üretme yardımcıları.

Bu dosya RGB ve termal video akışlarıyla eşleşen sensör verilerini okur.
CSV içinden zaman, FOV, zoom, tilt, roll ve pan kolonları bulunur. Daha sonra
video zamanına göre ilgili sensör değeri seçilir veya iki satır arasında
interpolation yapılır.
"""

import bisect
import csv
import math
from pathlib import Path
from typing import TypeAlias


SensorValue: TypeAlias = float | str | None
SensorRow: TypeAlias = dict[str, SensorValue]

DEFAULT_FOV_H_DEG = 65.7
DEFAULT_FOV_V_DEG = 39.9
DEFAULT_THERMAL_FOV_H_DEG = 32.4
DEFAULT_THERMAL_FOV_V_DEG = 24.6


def parse_float(value: object) -> float | None:
    """Ham CSV değerini mümkünse float değere çevirir.

    CSV dosyasında sayılar bazen string olarak, bazen boş değer olarak, bazen
    de virgüllü ondalık formatıyla gelebilir. Bu fonksiyon bu değerleri güvenli
    şekilde sayıya çevirmeye çalışır.

    Args:
        value: CSV dosyasından okunan ham değer.

    Returns:
        Float değer veya geçersiz/boş veri durumunda None.
    """
    # CSV içinde eksik alan varsa doğrudan None döndürülür.
    if value is None:
        return None

    text = str(value).strip()

    # Boş veya metinsel null değerleri sayıya çevrilmez.
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None

    # Bazı CSV kaynaklarında ondalık ayırıcı virgül olabilir.
    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def find_column(
    fieldnames: list[str] | None, possible_names: list[str]
) -> str | None:
    """CSV kolonları içinde olası kolon adlarından birini bulur.

    Farklı kayıt dosyalarında aynı bilgi farklı kolon adlarıyla tutulabilir.
    Bu nedenle önce birebir eşleşme, sonra kısmi eşleşme denenir.

    Args:
        fieldnames: CSV dosyasındaki mevcut kolon adları.
        possible_names: Aranacak olası kolon adları.

    Returns:
        Bulunan gerçek kolon adı veya eşleşme yoksa None.
    """
    if not fieldnames:
        return None

    # Kolon isimleri küçük harfe çevrilerek karşılaştırma daha toleranslı
    # hale getirilir.
    lowered = {name.lower().strip(): name for name in fieldnames}

    # Önce birebir kolon adı eşleşmesi aranır.
    for possible in possible_names:
        key = possible.lower().strip()

        if key in lowered:
            return lowered[key]

    # Birebir eşleşme yoksa kolon adının içinde geçen ifadeler kontrol edilir.
    for name in fieldnames:
        low = name.lower().strip()

        for possible in possible_names:
            if possible.lower().strip() in low:
                return name

    return None


def parse_time_to_seconds(
    value: object, first_absolute_time: float | None = None
) -> tuple[float | None, float | None]:
    """CSV zaman değerini video başlangıcına göre saniyeye çevirir.

    Zaman değeri doğrudan saniye olarak gelebilir veya saat:dakika:saniye
    formatında yazılmış olabilir. Mutlak saat formatı kullanılıyorsa ilk okunan
    zaman referans alınır ve sonraki değerler göreli saniyeye çevrilir.

    Args:
        value: CSV dosyasından okunan zaman değeri.
        first_absolute_time: İlk mutlak zamanın saniye karşılığı.

    Returns:
        Göreli saniye değeri ve güncellenmiş ilk mutlak zaman.
    """
    if value is None:
        return None, first_absolute_time

    text = str(value).strip()

    if text == "":
        return None, first_absolute_time

    # Zaman zaten sayısal saniye olarak verilmişse doğrudan kullanılır.
    numeric = parse_float(text)

    if numeric is not None:
        return numeric, first_absolute_time

    # Tarih + saat formatı varsa son parça saat bilgisi olarak alınır.
    parts = text.split()
    time_part = parts[-1] if len(parts) >= 2 else text
    chunks = time_part.split(":")

    if len(chunks) < 2:
        return None, first_absolute_time

    try:
        hour = float(chunks[0])
        minute = float(chunks[1])
        second = float(chunks[2]) if len(chunks) >= 3 else 0.0

        absolute_seconds = hour * 3600.0 + minute * 60.0 + second

        # İlk mutlak zaman video başlangıcı gibi kabul edilir.
        if first_absolute_time is None:
            first_absolute_time = absolute_seconds

        relative_seconds = absolute_seconds - first_absolute_time

        # Gün değişimi veya hatalı zaman sırası varsa negatif değer sıfırlanır.
        if relative_seconds < 0:
            relative_seconds = 0.0

        return relative_seconds, first_absolute_time

    except ValueError:
        return None, first_absolute_time


def normalize_fov(value: float | None, default_value: float) -> float:
    """FOV değerini derece cinsinden güvenli aralığa normalize eder.

    Bazı kaynaklarda FOV radyan olarak gelebilir. Değer küçük bir aralıkta ise
    radyan kabul edilip dereceye çevrilir. Geçersiz veya aşırı büyük değerlerde
    varsayılan FOV kullanılır.

    Args:
        value: CSV'den okunan FOV değeri.
        default_value: Geçersiz veri durumunda kullanılacak varsayılan FOV.

    Returns:
        Derece cinsinden normalize edilmiş FOV değeri.
    """
    if value is None:
        return default_value

    # 0.01 ile 3.2 arası değerler büyük ihtimalle radyan formatındadır.
    if 0.01 < value < 3.2:
        value = math.degrees(value)

    if value <= 0.01:
        return default_value

    if value > 120.0:
        return default_value

    return value


def channel_column_names(base_names: list[str], channel: str) -> list[str]:
    """Kanal tipine göre olası CSV kolon adlarını üretir.

    RGB ve termal kanal için aynı temel bilgi farklı eklerle yazılmış olabilir.
    Örneğin fov_h_rgb, rgb_fov_h, fov_hthermal gibi varyasyonlar bu fonksiyonla
    üretilir.

    Args:
        base_names: Temel kolon adı adayları.
        channel: İşlenen kanal tipi. "rgb" veya "thermal" olabilir.

    Returns:
        Kanal alias'larıyla genişletilmiş benzersiz kolon adı listesi.
    """
    names: list[str] = []

    channel_aliases = {
        "rgb": ["rgb", "visible", "vis", "color"],
        "thermal": ["thermal", "therm", "ir", "tir", "th"],
    }

    # Her temel kolon adı, ilgili kanal alias'larıyla farklı biçimlerde denenir.
    for base_name in base_names:
        for alias in channel_aliases.get(channel, []):
            names.extend(
                [
                    f"{base_name}_{alias}",
                    f"{alias}_{base_name}",
                    f"{base_name}{alias}",
                    f"{alias}{base_name}",
                ]
            )

    # Kanal eki olmayan temel adlar da arama listesine eklenir.
    names.extend(base_names)

    unique_names: list[str] = []
    seen: set[str] = set()

    # Aynı kolon adı farklı yollardan üretildiyse tekrarları temizlenir.
    for name in names:
        key = name.lower().strip()

        if key not in seen:
            unique_names.append(name)
            seen.add(key)

    return unique_names


def channel_default_fov(channel: str) -> tuple[float, float]:
    """Kanal tipine göre varsayılan yatay ve dikey FOV değerini döndürür.

    Args:
        channel: İşlenen kanal tipi.

    Returns:
        Yatay ve dikey FOV değerleri.
    """
    if channel == "thermal":
        return DEFAULT_THERMAL_FOV_H_DEG, DEFAULT_THERMAL_FOV_V_DEG

    return DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG


def load_sensor_csv(
    csv_path: str | Path, channel: str = "rgb"
) -> list[SensorRow]:
    """Sensör CSV dosyasını okur ve normalize edilmiş satır listesi üretir.

    Fonksiyon CSV delimiter tipini otomatik algılamaya çalışır. Ardından zaman,
    FOV, zoom, tilt, roll ve pan kolonlarını bulur. Eksik FOV değerleri kanalın
    varsayılan değerleriyle tamamlanır.

    Args:
        csv_path: Okunacak sensör CSV dosyasının yolu.
        channel: Sensör verisinin kullanılacağı kanal. "rgb" veya "thermal".

    Returns:
        Saniyeye göre sıralanmış sensör satırları listesi.
    """
    csv_file = Path(csv_path)
    default_fov_h, default_fov_v = channel_default_fov(channel)

    if not csv_file.exists():
        print(f"CSV bulunamadi: {csv_path}")
        return []

    with csv_file.open("r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            # CSV dosyası virgül, noktalı virgül veya tab ile ayrılmış olabilir.
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)
        fieldnames = reader.fieldnames

        print(f"CSV kolonlari ({channel}):", fieldnames)

        time_col = find_column(
            fieldnames,
            [
                "video_time",
                "time_sec",
                "seconds",
                "second",
                "timestamp",
                "time",
                "datetime",
            ],
        )

        fov_h_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "fov_h",
                    "hfov",
                    "horizontal_fov",
                    "fov_horizontal",
                    "camera_fov_h",
                    "cam_fov_h",
                ],
                channel,
            ),
        )

        fov_v_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "fov_v",
                    "vfov",
                    "vertical_fov",
                    "fov_vertical",
                    "camera_fov_v",
                    "cam_fov_v",
                ],
                channel,
            ),
        )

        zoom_col = find_column(
            fieldnames,
            channel_column_names(
                ["zoom", "zoom_value", "camera_zoom", "cam_zoom"], channel
            ),
        )

        tilt_col = find_column(
            fieldnames,
            channel_column_names(
                [
                    "tilt",
                    "camera_tilt",
                    "ptz_tilt",
                    "tilt_angle",
                    "pitch",
                    "camera_pitch",
                ],
                channel,
            ),
        )

        roll_col = find_column(
            fieldnames,
            channel_column_names(
                ["roll", "camera_roll", "roll_angle"], channel
            ),
        )

        pan_col = find_column(
            fieldnames,
            channel_column_names(
                ["pan", "camera_pan", "ptz_pan", "pan_angle", "yaw"], channel
            ),
        )

        print(f"time kolonu ({channel}):", time_col)
        print(f"fov_h kolonu ({channel}):", fov_h_col)
        print(f"fov_v kolonu ({channel}):", fov_v_col)
        print(f"zoom kolonu ({channel}):", zoom_col)
        print(f"tilt kolonu ({channel}):", tilt_col)
        print(f"roll kolonu ({channel}):", roll_col)
        print(f"pan kolonu ({channel}):", pan_col)

        rows: list[SensorRow] = []
        first_absolute_time = None

        for index, row in enumerate(reader):
            # Zaman kolonu varsa gerçek zaman kullanılır. Yoksa satır index'i
            # yaklaşık saniye değeri gibi kullanılarak iş akışı korunur.
            if time_col:
                second, first_absolute_time = parse_time_to_seconds(
                    row.get(time_col), first_absolute_time
                )
            else:
                second = float(index)

            if second is None:
                second = float(index)

            # Her sensör alanı güvenli float dönüşümünden geçirilir.
            fov_h = parse_float(row.get(fov_h_col)) if fov_h_col else None
            fov_v = parse_float(row.get(fov_v_col)) if fov_v_col else None
            zoom = parse_float(row.get(zoom_col)) if zoom_col else None
            tilt = parse_float(row.get(tilt_col)) if tilt_col else None
            roll = parse_float(row.get(roll_col)) if roll_col else None
            pan = parse_float(row.get(pan_col)) if pan_col else None

            rows.append(
                {
                    "second": float(second),
                    "fov_h": normalize_fov(fov_h, default_fov_h),
                    "fov_v": normalize_fov(fov_v, default_fov_v),
                    "zoom": zoom,
                    "tilt": tilt,
                    "roll": roll,
                    "pan": pan,
                }
            )

    # Interpolation sırasında bisect kullanılacağı için satırlar zamana göre
    # sıralı tutulmalıdır.
    rows.sort(key=lambda item: item["second"])

    print(f"Okunan sensor satiri ({channel}): {len(rows)}")

    return rows


def interpolate_value(
    a_value: float | None, b_value: float | None, ratio: float
) -> float | None:
    """İki sayısal sensör değeri arasında lineer interpolation yapar.

    Args:
        a_value: Başlangıç satırındaki değer.
        b_value: Bitiş satırındaki değer.
        ratio: İki zaman arasındaki konum oranı.

    Returns:
        Interpolation sonucu veya iki değer de yoksa None.
    """
    if a_value is None and b_value is None:
        return None

    if a_value is None:
        return b_value

    if b_value is None:
        return a_value

    return a_value + (b_value - a_value) * ratio


def get_sensor_for_time(
    sensor_rows: list[SensorRow], video_second: float
) -> SensorRow:
    """Video zamanına karşılık gelen sensör bilgisini döndürür.

    Eğer iki sensör satırı arasında kalınıyorsa FOV, zoom, tilt, roll ve pan
    değerleri lineer interpolation ile hesaplanır.

    Args:
        sensor_rows: Zamana göre sıralanmış sensör satırları.
        video_second: Video içinde sorgulanan saniye değeri.

    Returns:
        İlgili zamana ait sensör bilgisi.
    """
    if not sensor_rows:
        return {
            "fov_h": DEFAULT_FOV_H_DEG,
            "fov_v": DEFAULT_FOV_V_DEG,
            "zoom": None,
            "tilt": None,
            "roll": None,
            "pan": None,
            "source": "DEFAULT",
        }

    if len(sensor_rows) == 1:
        row = sensor_rows[0]

        return {**row, "source": "CSV"}

    seconds = [float(row["second"]) for row in sensor_rows]
    idx = bisect.bisect_right(seconds, video_second) - 1

    if idx < 0:
        idx = 0

    if idx >= len(sensor_rows) - 1:
        row = sensor_rows[-1]

        return {**row, "source": "CSV"}

    row_a = sensor_rows[idx]
    row_b = sensor_rows[idx + 1]

    time_a = float(row_a["second"])
    time_b = float(row_b["second"])

    if time_b <= time_a:
        ratio = 0.0
    else:
        ratio = (video_second - time_a) / (time_b - time_a)
        ratio = max(0.0, min(1.0, ratio))

    return {
        "fov_h": interpolate_value(
            row_a["fov_h"] if isinstance(row_a["fov_h"], float) else None,
            row_b["fov_h"] if isinstance(row_b["fov_h"], float) else None,
            ratio,
        ),
        "fov_v": interpolate_value(
            row_a["fov_v"] if isinstance(row_a["fov_v"], float) else None,
            row_b["fov_v"] if isinstance(row_b["fov_v"], float) else None,
            ratio,
        ),
        "zoom": interpolate_value(
            row_a["zoom"] if isinstance(row_a["zoom"], float) else None,
            row_b["zoom"] if isinstance(row_b["zoom"], float) else None,
            ratio,
        ),
        "tilt": interpolate_value(
            row_a["tilt"] if isinstance(row_a["tilt"], float) else None,
            row_b["tilt"] if isinstance(row_b["tilt"], float) else None,
            ratio,
        ),
        "roll": interpolate_value(
            row_a["roll"] if isinstance(row_a["roll"], float) else None,
            row_b["roll"] if isinstance(row_b["roll"], float) else None,
            ratio,
        ),
        "pan": interpolate_value(
            row_a["pan"] if isinstance(row_a["pan"], float) else None,
            row_b["pan"] if isinstance(row_b["pan"], float) else None,
            ratio,
        ),
        "source": "CSV_INTERP",
    }


def smooth_sensor(
    previous_sensor: SensorRow | None, new_sensor: SensorRow
) -> SensorRow:
    """Sensör değerlerini frame'ler arasında yumuşatır.

    Kamera FOV, zoom, tilt, roll ve pan değerleri CSV içinde anlık sıçramalar
    gösterebilir. Bu fonksiyon önceki sensör bilgisiyle yeni sensör bilgisini
    ağırlıklı ortalama kullanarak birleştirir.

    Args:
        previous_sensor: Bir önceki frame'den kalan yumuşatılmış sensör değeri.
        new_sensor: Mevcut frame için okunan ham veya interpolated sensör değeri.

    Returns:
        Yumuşatılmış sensör bilgisi.
    """
    if previous_sensor is None:
        return new_sensor.copy()

    alphas = {
        "fov_h": 0.45,
        "fov_v": 0.45,
        "zoom": 0.45,
        "tilt": 0.20,
        "roll": 0.20,
        "pan": 0.35,
    }

    smoothed = new_sensor.copy()

    for key, alpha in alphas.items():
        old = previous_sensor.get(key)
        new = new_sensor.get(key)

        # Eski değer yoksa yeni değer doğrudan kullanılır.
        if old is None:
            smoothed[key] = new

        # Yeni değer yoksa eski değer korunur.
        elif new is None:
            smoothed[key] = old

        # İki değer de sayısalsa EMA benzeri ağırlıklı ortalama uygulanır.
        elif isinstance(old, float) and isinstance(new, float):
            smoothed[key] = (1.0 - alpha) * old + alpha * new

    smoothed["source"] = new_sensor.get("source", "CSV")

    return smoothed
