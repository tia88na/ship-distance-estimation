"""Sensor CSV reading and interpolation helpers."""

import bisect
import csv
import math
from pathlib import Path


DEFAULT_FOV_H_DEG = 65.7
DEFAULT_FOV_V_DEG = 39.9
DEFAULT_THERMAL_FOV_H_DEG = 32.4
DEFAULT_THERMAL_FOV_V_DEG = 24.6


def parse_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def find_column(fieldnames, possible_names):
    if not fieldnames:
        return None

    lowered = {name.lower().strip(): name for name in fieldnames}

    for possible in possible_names:
        key = possible.lower().strip()

        if key in lowered:
            return lowered[key]

    for name in fieldnames:
        low = name.lower().strip()

        for possible in possible_names:
            if possible.lower().strip() in low:
                return name

    return None


def parse_time_to_seconds(value, first_absolute_time=None):
    if value is None:
        return None, first_absolute_time

    text = str(value).strip()

    if text == "":
        return None, first_absolute_time

    numeric = parse_float(text)

    if numeric is not None:
        return numeric, first_absolute_time

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

        if first_absolute_time is None:
            first_absolute_time = absolute_seconds

        relative_seconds = absolute_seconds - first_absolute_time

        if relative_seconds < 0:
            relative_seconds = 0.0

        return relative_seconds, first_absolute_time

    except ValueError:
        return None, first_absolute_time


def normalize_fov(value, default_value):
    if value is None:
        return default_value

    if 0.01 < value < 3.2:
        value = math.degrees(value)

    if value <= 0.01:
        return default_value

    if value > 120.0:
        return default_value

    return value


def channel_column_names(base_names, channel):
    names = []

    channel_aliases = {
        "rgb": ["rgb", "visible", "vis", "color"],
        "thermal": ["thermal", "therm", "ir", "tir", "th"],
    }

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

    names.extend(base_names)

    unique_names = []
    seen = set()

    for name in names:
        key = name.lower().strip()

        if key not in seen:
            unique_names.append(name)
            seen.add(key)

    return unique_names


def channel_default_fov(channel):
    if channel == "thermal":
        return DEFAULT_THERMAL_FOV_H_DEG, DEFAULT_THERMAL_FOV_V_DEG

    return DEFAULT_FOV_H_DEG, DEFAULT_FOV_V_DEG


def load_sensor_csv(csv_path, channel="rgb"):
    csv_file = Path(csv_path)
    default_fov_h, default_fov_v = channel_default_fov(channel)

    if not csv_file.exists():
        print(f"CSV bulunamadi: {csv_path}")
        return []

    with csv_file.open("r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(4096)
        file.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;	")
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

        rows = []
        first_absolute_time = None

        for index, row in enumerate(reader):
            if time_col:
                second, first_absolute_time = parse_time_to_seconds(
                    row.get(time_col), first_absolute_time
                )
            else:
                second = float(index)

            if second is None:
                second = float(index)

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

    rows.sort(key=lambda item: item["second"])

    print(f"Okunan sensor satiri ({channel}): {len(rows)}")

    return rows


def interpolate_value(a_value, b_value, ratio):
    if a_value is None and b_value is None:
        return None

    if a_value is None:
        return b_value

    if b_value is None:
        return a_value

    return a_value + (b_value - a_value) * ratio


def get_sensor_for_time(sensor_rows, video_second):
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

    seconds = [row["second"] for row in sensor_rows]
    idx = bisect.bisect_right(seconds, video_second) - 1

    if idx < 0:
        idx = 0

    if idx >= len(sensor_rows) - 1:
        row = sensor_rows[-1]

        return {**row, "source": "CSV"}

    row_a = sensor_rows[idx]
    row_b = sensor_rows[idx + 1]

    time_a = row_a["second"]
    time_b = row_b["second"]

    if time_b <= time_a:
        ratio = 0.0
    else:
        ratio = (video_second - time_a) / (time_b - time_a)
        ratio = max(0.0, min(1.0, ratio))

    return {
        "fov_h": interpolate_value(row_a["fov_h"], row_b["fov_h"], ratio),
        "fov_v": interpolate_value(row_a["fov_v"], row_b["fov_v"], ratio),
        "zoom": interpolate_value(row_a["zoom"], row_b["zoom"], ratio),
        "tilt": interpolate_value(row_a["tilt"], row_b["tilt"], ratio),
        "roll": interpolate_value(row_a["roll"], row_b["roll"], ratio),
        "pan": interpolate_value(row_a["pan"], row_b["pan"], ratio),
        "source": "CSV_INTERP",
    }


def smooth_sensor(previous_sensor, new_sensor):
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

        if old is None:
            smoothed[key] = new
        elif new is None:
            smoothed[key] = old
        else:
            smoothed[key] = (1.0 - alpha) * old + alpha * new

    smoothed["source"] = new_sensor.get("source", "CSV")

    return smoothed
