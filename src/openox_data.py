from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


FEATURE_NAMES = ["red_signal", "ir_signal"]
PPG_CHANNEL_NAMES = ["Red Signal", "IR Signal"]
PPG_CHANNEL_ALIASES = {
    "red_signal": ["red signal", "red", "red ppg", "ppg red", "red led", "red_signal"],
    "ir_signal": ["ir signal", "ir", "infrared", "infrared signal", "ir ppg", "ppg ir", "ir led", "ir_signal"],
}
DEMO_FEATURE_NAMES = [
    "age_norm",
    "sex_female",
    "sex_male",
    "race_black",
    "race_asian",
    "race_white",
    "race_other",
    "fitzpatrick_norm",
    "warming",
]

DEFAULT_OPENOX_WINDOW_SEC = 6.0
DEFAULT_OPENOX_SAMPLING_RATE = 50.0
OPENOX_BLOODGAS_TASKS = {
    "so2": {
        "column": "so2",
        "target_name": "SO2",
        "unit": "%",
        "target_min": 60.0,
        "target_max": 100.0,
    },
    "sao2_spo2": {
        "column": "so2",
        "target_name": "SaO2_SpO2",
        "target_names": ["SaO2", "SpO2"],
        "unit": "%",
        "target_units": ["%", "%"],
        "target_min": 60.0,
        "target_max": 100.0,
    },
}
OPENOX_TARGET_ALIASES = {
    "sao2": "so2",
    "so2": "so2",
    "spo2": "spo2",
    "sao2_spo2": "sao2_spo2",
    "so2_spo2": "sao2_spo2",
    "dual": "sao2_spo2",
}


def normalize_openox_target(target):
    key = str(target).lower()
    if key.startswith("openox_"):
        key = key[len("openox_"):]
    if key not in OPENOX_TARGET_ALIASES:
        valid = sorted(set(OPENOX_TARGET_ALIASES))
        raise ValueError(f"OpenOx target must be one of: {', '.join(valid)}")
    return OPENOX_TARGET_ALIASES[key]


def get_openox_target_config(target):
    target = normalize_openox_target(target)
    if target == "spo2":
        return {
            "column": "spo2",
            "target_name": "SpO2",
            "unit": "%",
            "target_min": 60.0,
            "target_max": 100.0,
        }
    return OPENOX_BLOODGAS_TASKS[target].copy()


def _resolve_output_len(window_sec, output_len=0, sampling_rate=DEFAULT_OPENOX_SAMPLING_RATE):
    if output_len is not None and int(output_len) > 0:
        return int(output_len)
    if sampling_rate is None or float(sampling_rate) <= 0:
        raise ValueError("sampling_rate must be positive when output_len is not set")
    resolved = int(round(float(window_sec) * float(sampling_rate)))
    if resolved <= 1:
        raise ValueError("Resolved OpenOx output length must be greater than 1")
    return resolved


def build_red_ir_observation_matrix(
        spectral_min=600.0,
        spectral_max=1000.0,
        spectral_step=5.0,
        sigma=25.0):
    """Approximate clinical pulse oximeter red/IR responses."""
    if spectral_step <= 0:
        raise ValueError("spectral_step must be positive")
    wavelengths = np.arange(
        float(spectral_min),
        float(spectral_max) + float(spectral_step) * 0.5,
        float(spectral_step),
        dtype=np.float32)
    rows = []
    for center in [660.0, 940.0]:
        row = np.exp(-0.5 * ((wavelengths - center) / float(sigma)) ** 2)
        row = np.clip(row, 0.0, None)
        denom = row.sum()
        if denom > 0:
            row = row / denom
        rows.append(row.astype(np.float32))
    return np.stack(rows, axis=0), wavelengths


def _resolve_root(root):
    root = Path(root)
    if (root / "bloodgas.csv").exists():
        return root
    version_root = root / "1.1.1"
    if (version_root / "bloodgas.csv").exists():
        return version_root
    raise FileNotFoundError(
        f"OpenOx files not found under {root}. Expected bloodgas.csv or 1.1.1/bloodgas.csv.")


def _parse_sample(value):
    if pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _parse_header_signal(line):
    parts = line.strip().split(maxsplit=8)
    if len(parts) < 9:
        raise ValueError(f"Cannot parse WFDB signal header line: {line}")
    file_name = parts[0]
    fmt = parts[1]
    gain_text = parts[2]
    baseline_text = parts[4]
    signal_name = parts[8]

    try:
        gain = float(gain_text.split("(")[0])
    except ValueError:
        gain = 1.0
    if not np.isfinite(gain) or gain == 0:
        gain = 1.0
    try:
        baseline = float(baseline_text)
    except ValueError:
        baseline = 0.0
    return file_name, fmt, gain, baseline, signal_name


def _read_ppg_header(hea_path):
    lines = Path(hea_path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty WFDB header: {hea_path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Cannot parse WFDB header first line: {lines[0]}")
    n_signals = int(first[1])
    fs = float(first[2])
    n_samples = int(first[3])
    start_time = first[4] if len(first) >= 5 else None

    signals = []
    for line in lines[1:1 + n_signals]:
        signals.append(_parse_header_signal(line))
    return {
        "n_signals": n_signals,
        "fs": fs,
        "n_samples": n_samples,
        "start_time": start_time,
        "signals": signals,
    }


def _read_red_ir_ppg(hea_path):
    header = _read_ppg_header(hea_path)
    formats = {signal[1] for signal in header["signals"]}
    files = {signal[0] for signal in header["signals"]}
    if formats != {"32"} or len(files) != 1:
        raise ValueError(
            f"{hea_path} uses WFDB formats/files {formats}/{files}. "
            "This lightweight reader supports the OpenOx 32-bit interleaved PPG files only.")

    dat_path = Path(hea_path).with_suffix(".dat")
    raw = np.fromfile(dat_path, dtype="<i4")
    expected = header["n_samples"] * header["n_signals"]
    if raw.size < expected:
        raise ValueError(f"PPG file is shorter than expected: {dat_path}")
    raw = raw[:expected].reshape(header["n_samples"], header["n_signals"]).T.astype(np.float32)

    signal_names = [signal[4] for signal in header["signals"]]
    normalized_names = [name.strip().lower() for name in signal_names]
    channel_indices = []
    for key, display_name in zip(["red_signal", "ir_signal"], PPG_CHANNEL_NAMES):
        aliases = PPG_CHANNEL_ALIASES[key]
        matched = None
        for alias in aliases:
            if alias in normalized_names:
                matched = normalized_names.index(alias)
                break
        if matched is None:
            raise ValueError(f"{hea_path} does not contain channel '{display_name}'. Found: {signal_names}")
        channel_indices.append(matched)

    selected = []
    for idx in channel_indices:
        _, _, gain, baseline, _ = header["signals"][idx]
        selected.append((raw[idx] - baseline) / gain)
    return np.stack(selected, axis=0).astype(np.float32), header


def _record_start_datetime(encounter_date, start_time):
    if not start_time or pd.isna(encounter_date):
        return pd.NaT
    return pd.to_datetime(f"{encounter_date} {start_time}", errors="coerce")


def _acdc_window(window, fs, eps=1e-6):
    fs = float(fs)
    nyquist = fs * 0.5
    if nyquist <= 0.6:
        raise ValueError("fs is too low for OpenOx AC/DC preprocessing")
    high_cut = min(12.0, nyquist * 0.95)
    low_cut = 0.5
    if high_cut <= low_cut:
        raise ValueError("fs is too low for 0.5-12 Hz AC band extraction")

    dc_sos = butter(2, low_cut, btype="lowpass", fs=fs, output="sos")
    ac_sos = butter(2, [low_cut, high_cut], btype="bandpass", fs=fs, output="sos")
    dc = sosfiltfilt(dc_sos, window, axis=1).astype(np.float32)
    ac = sosfiltfilt(ac_sos, window, axis=1).astype(np.float32)

    denom = np.maximum(np.abs(dc), eps)
    return (ac / denom).astype(np.float32)


def _remove_baseline_drift(ppg, fs, cutoff_hz=0.3, order=2):
    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0:
        return ppg.astype(np.float32)
    fs = float(fs)
    nyquist = fs * 0.5
    if nyquist <= cutoff_hz:
        raise ValueError("fs is too low for OpenOx baseline drift removal")
    sos = butter(
        int(order),
        cutoff_hz / nyquist,
        btype="highpass",
        output="sos")
    try:
        return sosfiltfilt(sos, ppg, axis=1).astype(np.float32)
    except ValueError:
        return (ppg - ppg.mean(axis=1, keepdims=True)).astype(np.float32)


def _is_acdc_mode(mode):
    return str(mode).lower() in ["acdc", "ac_dc", "acdc_ratio", "ac_dc_ratio"]


def _normalize_window(window, mode, fs=None, eps=1e-6):
    mode = str(mode).lower()
    if mode in ["none", "raw"]:
        return window
    if _is_acdc_mode(mode):
        if fs is None:
            raise ValueError("fs is required for OpenOx AC/DC preprocessing")
        return _acdc_window(window, fs, eps=eps)
    if mode in ["window_minmax", "window", "minmax"]:
        mins = window.min(axis=1, keepdims=True)
        maxs = window.max(axis=1, keepdims=True)
        return (window - mins) / np.maximum(maxs - mins, eps)
    if mode in ["zscore", "window_zscore"]:
        means = window.mean(axis=1, keepdims=True)
        stds = window.std(axis=1, keepdims=True)
        return (window - means) / np.maximum(stds, eps)
    raise ValueError("normalize must be one of: acdc, window_minmax, zscore, none")


def _resample_window(window, output_len):
    output_len = int(output_len)
    if window.shape[1] == output_len:
        return window.astype(np.float32)
    old_grid = np.linspace(0.0, 1.0, num=window.shape[1], dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, num=output_len, dtype=np.float32)
    resampled = [np.interp(new_grid, old_grid, channel).astype(np.float32) for channel in window]
    return np.stack(resampled, axis=0)


def _build_sample_time_map(root):
    sample_time_by_encounter = {}
    for csv_path in (root / "waveforms").rglob("*_2hz.csv"):
        encounter_id = csv_path.stem[:-4]
        try:
            df = pd.read_csv(csv_path, usecols=["Sample", "Timestamp"])
        except ValueError:
            continue
        df = df.dropna(subset=["Sample", "Timestamp"]).copy()
        if df.empty:
            continue
        df["sample_number"] = df["Sample"].map(_parse_sample)
        df["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["sample_number", "timestamp"])
        if df.empty:
            continue
        grouped = df.groupby("sample_number")["timestamp"].min()
        sample_time_by_encounter[encounter_id] = grouped.to_dict()
    return sample_time_by_encounter


def _encode_demo_rows(root):
    encounters = pd.read_csv(root / "encounter.csv", low_memory=False)
    patients = pd.read_csv(root / "patient.csv", low_memory=False)
    df = encounters.merge(patients, on="patient_id", how="left", suffixes=("", "_patient"))
    demo_by_encounter = {}
    patient_by_encounter = {}
    date_by_encounter = {}

    for row in df.itertuples(index=False):
        encounter_id = str(getattr(row, "encounter_id"))
        patient_id = str(getattr(row, "patient_id"))
        patient_by_encounter[encounter_id] = patient_id
        date_by_encounter[encounter_id] = getattr(row, "encounter_date", None)

        age = getattr(row, "age_at_encounter", np.nan)
        fitz = getattr(row, "fitzpatrick", np.nan)
        warming = getattr(row, "warming", np.nan)
        sex = str(getattr(row, "assigned_sex", "")).lower()
        race = str(getattr(row, "race", "")).lower()

        age_norm = 0.0 if pd.isna(age) else float(np.clip(float(age) / 100.0, 0.0, 1.0))
        fitz_norm = 0.0 if pd.isna(fitz) else float(np.clip((float(fitz) - 1.0) / 5.0, 0.0, 1.0))
        warming_value = 0.0 if pd.isna(warming) else float(np.clip(float(warming), 0.0, 1.0))
        race_black = float("black" in race or "african" in race)
        race_asian = float("asian" in race)
        race_white = float("white" in race or "caucasian" in race)
        race_known = race not in ["", "nan", "none"]
        race_other = float(race_known and not (race_black or race_asian or race_white))

        demo_by_encounter[encounter_id] = np.asarray([
            age_norm,
            float("female" in sex),
            float("male" in sex),
            race_black,
            race_asian,
            race_white,
            race_other,
            fitz_norm,
            warming_value,
        ], dtype=np.float32)

    return demo_by_encounter, patient_by_encounter, date_by_encounter


def _label_table(
        root,
        target,
        target_min=None,
        target_max=None,
        spo2_source="continuous_2hz",
        spo2_aggregation="median",
        spo2_max_device_range=10.0,
        spo2_stride_sec=30.0):
    target = normalize_openox_target(target)
    config = get_openox_target_config(target)
    if target_min is None:
        target_min = config["target_min"]
    if target_max is None:
        target_max = config["target_max"]

    if target in OPENOX_BLOODGAS_TASKS:
        column = config["column"]
        bloodgas = pd.read_csv(root / "bloodgas.csv", low_memory=False)
        labels = bloodgas[["patient_id", "encounter_id", "sample", "date", "time", column]].copy()
        labels = labels.rename(columns={"sample": "sample_number", column: "target"})
        labels["sample_number"] = labels["sample_number"].map(_parse_sample)
        labels["fallback_time"] = pd.to_datetime(
            labels["date"].astype(str) + " " + labels["time"].astype(str),
            errors="coerce")
        labels["label_source"] = target
        return labels

    if target == "spo2":
        source = str(spo2_source).lower()
        if source in ["continuous_2hz", "2hz", "waveform_2hz"]:
            return _continuous_spo2_label_table(
                root,
                target_min=target_min,
                target_max=target_max,
                spo2_aggregation=spo2_aggregation,
                spo2_max_device_range=spo2_max_device_range,
                spo2_stride_sec=spo2_stride_sec)
        if source not in ["pulseoximeter", "pulse", "sample"]:
            raise ValueError("spo2_source must be 'continuous_2hz' or 'pulseoximeter'")
        pulse = pd.read_csv(root / "pulseoximeter.csv", low_memory=False)
        pulse = pulse[["encounter_id", "sample_number", "saturation"]].copy()
        pulse["sample_number"] = pulse["sample_number"].map(_parse_sample)
        pulse["target"] = pd.to_numeric(pulse["saturation"], errors="coerce")
        pulse = pulse.dropna(subset=["encounter_id", "sample_number", "target"])
        pulse = pulse[(pulse["target"] >= float(target_min)) & (pulse["target"] <= float(target_max))].copy()
        aggregation = str(spo2_aggregation).lower()
        if aggregation not in ["median", "mean"]:
            raise ValueError("spo2_aggregation must be 'median' or 'mean'")
        grouped = pulse.groupby(["encounter_id", "sample_number"], as_index=False).agg(
            target=("target", aggregation),
            spo2_device_count=("target", "count"),
            spo2_device_min=("target", "min"),
            spo2_device_max=("target", "max"),
            spo2_device_std=("target", "std"))
        grouped["spo2_device_std"] = grouped["spo2_device_std"].fillna(0.0)
        grouped["spo2_device_range"] = grouped["spo2_device_max"] - grouped["spo2_device_min"]
        max_range = float(spo2_max_device_range)
        if max_range > 0:
            grouped = grouped[grouped["spo2_device_range"] <= max_range].copy()
        grouped["fallback_time"] = pd.NaT
        range_text = f"_r{max_range:g}" if max_range > 0 else "_rnone"
        grouped["label_source"] = f"spo2_{aggregation}_clean{range_text}"
        return grouped

    raise ValueError("OpenOx target must be 'so2' or 'spo2'")


def _continuous_spo2_label_table(
        root,
        target_min=60.0,
        target_max=100.0,
        spo2_aggregation="median",
        spo2_max_device_range=10.0,
        spo2_stride_sec=30.0):
    aggregation = str(spo2_aggregation).lower()
    if aggregation not in ["median", "mean"]:
        raise ValueError("spo2_aggregation must be 'median' or 'mean'")

    rows = []
    max_range = float(spo2_max_device_range)
    stride_sec = float(spo2_stride_sec)

    for csv_path in (root / "waveforms").rglob("*_2hz.csv"):
        encounter_id = csv_path.stem[:-4]
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        spo2_cols = [col for col in df.columns if str(col).endswith("_SpO2")]
        if "Timestamp" not in df.columns or not spo2_cols:
            continue

        timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
        values = df[spo2_cols].apply(pd.to_numeric, errors="coerce")
        values = values.where((values >= float(target_min)) & (values <= float(target_max)))
        device_count = values.notna().sum(axis=1)
        if aggregation == "median":
            target = values.median(axis=1, skipna=True)
        else:
            target = values.mean(axis=1, skipna=True)
        device_min = values.min(axis=1, skipna=True)
        device_max = values.max(axis=1, skipna=True)
        device_std = values.std(axis=1, skipna=True).fillna(0.0)
        device_range = device_max - device_min

        valid = timestamps.notna() & target.notna() & (device_count > 0)
        if max_range > 0:
            valid = valid & (device_range <= max_range)
        if not valid.any():
            continue

        temp = pd.DataFrame({
            "encounter_id": encounter_id,
            "label_time": timestamps[valid].to_numpy(),
            "target": target[valid].astype(float).to_numpy(),
            "spo2_device_count": device_count[valid].astype(float).to_numpy(),
            "spo2_device_min": device_min[valid].astype(float).to_numpy(),
            "spo2_device_max": device_max[valid].astype(float).to_numpy(),
            "spo2_device_std": device_std[valid].astype(float).to_numpy(),
            "spo2_device_range": device_range[valid].astype(float).to_numpy(),
        }).sort_values("label_time").reset_index(drop=True)

        if stride_sec > 0 and not temp.empty:
            elapsed = (temp["label_time"] - temp["label_time"].iloc[0]).dt.total_seconds()
            temp["stride_bin"] = np.floor(elapsed / stride_sec).astype(int)
            temp = temp.groupby("stride_bin", as_index=False).first()
            temp = temp.drop(columns=["stride_bin"])

        temp["sample_number"] = np.arange(1, len(temp) + 1, dtype=np.int64)
        temp["fallback_time"] = temp["label_time"]
        stride_text = f"_s{stride_sec:g}" if stride_sec > 0 else "_sall"
        range_text = f"_r{max_range:g}" if max_range > 0 else "_rnone"
        temp["label_source"] = f"spo2_2hz_{aggregation}{range_text}{stride_text}"
        rows.append(temp)

    if not rows:
        return pd.DataFrame(columns=[
            "encounter_id",
            "sample_number",
            "target",
            "fallback_time",
            "label_time",
            "label_source",
            "spo2_device_count",
            "spo2_device_min",
            "spo2_device_max",
            "spo2_device_std",
            "spo2_device_range",
        ])
    return pd.concat(rows, ignore_index=True)


def _cache_path(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec):
    target = normalize_openox_target(target)
    config = get_openox_target_config(target)
    if target_min is None:
        target_min = config["target_min"]
    if target_max is None:
        target_max = config["target_max"]
    cache_dir = root / "processed_codex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    detrend_suffix = f"_detrend{float(detrend_cutoff):g}" if bool(detrend) else ""
    spo2_suffix = ""
    if str(target).lower() == "spo2":
        range_text = f"r{float(spo2_max_device_range):g}" if float(spo2_max_device_range) > 0 else "rnone"
        source_text = str(spo2_source).lower().replace("_", "")
        stride_text = f"_s{float(spo2_stride_sec):g}" if float(spo2_stride_sec) > 0 else "_sall"
        spo2_suffix = f"_spo2{source_text}_{str(spo2_aggregation).lower()}_{range_text}{stride_text}"
    elif str(target).lower() == "sao2_spo2":
        range_text = f"r{float(spo2_max_device_range):g}" if float(spo2_max_device_range) > 0 else "rnone"
        source_text = str(spo2_source).lower().replace("_", "")
        spo2_suffix = (
            f"_spo2{source_text}_{str(spo2_aggregation).lower()}_{range_text}"
            f"_sync{float(window_sec):g}s")
    name = (
        f"openox_{target}_red_ir_v2"
        f"_win{float(window_sec):g}"
        f"_len{int(output_len)}"
        f"_{normalize}"
        f"{detrend_suffix}"
        f"{spo2_suffix}"
        f"_y{float(target_min):g}-{float(target_max):g}.npz"
    )
    return cache_dir / name


def _build_samples(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec):
    target = normalize_openox_target(target)
    config = get_openox_target_config(target)
    is_dual_target = target == "sao2_spo2"
    if is_dual_target and str(spo2_source).lower() not in ["continuous_2hz", "2hz", "waveform_2hz"]:
        raise ValueError("The synchronized SaO2+SpO2 task requires spo2_source=continuous_2hz")
    if target_min is None:
        target_min = config["target_min"]
    if target_max is None:
        target_max = config["target_max"]
    labels = _label_table(
        root,
        target,
        target_min=target_min,
        target_max=target_max,
        spo2_source=spo2_source,
        spo2_aggregation=spo2_aggregation,
        spo2_max_device_range=spo2_max_device_range,
        spo2_stride_sec=spo2_stride_sec)
    labels["encounter_id"] = labels["encounter_id"].astype(str)
    labels = labels.dropna(subset=["encounter_id", "sample_number", "target"])
    labels["target"] = pd.to_numeric(labels["target"], errors="coerce")
    labels = labels.dropna(subset=["target"])
    labels = labels[(labels["target"] >= target_min) & (labels["target"] <= target_max)].copy()

    spo2_by_encounter = {}
    if is_dual_target:
        synchronized_spo2 = _continuous_spo2_label_table(
            root,
            target_min=target_min,
            target_max=target_max,
            spo2_aggregation=spo2_aggregation,
            spo2_max_device_range=spo2_max_device_range,
            spo2_stride_sec=0.0)
        synchronized_spo2["encounter_id"] = synchronized_spo2["encounter_id"].astype(str)
        synchronized_spo2["label_time"] = pd.to_datetime(
            synchronized_spo2["label_time"], errors="coerce")
        synchronized_spo2 = synchronized_spo2.dropna(subset=["label_time", "target"])
        spo2_by_encounter = {
            encounter_id: group.sort_values("label_time").reset_index(drop=True)
            for encounter_id, group in synchronized_spo2.groupby("encounter_id")
        }

    sample_time_by_encounter = _build_sample_time_map(root)
    demo_by_encounter, patient_by_encounter, date_by_encounter = _encode_demo_rows(root)
    ppg_headers = {
        path.stem[:-4]: path
        for path in (root / "waveforms").rglob("*_ppg.hea")
    }
    labels = labels[labels["encounter_id"].isin(ppg_headers)].copy()

    windows = []
    targets = []
    demos = []
    metadata_rows = []
    skipped = {
        "missing_patient": 0,
        "missing_time": 0,
        "missing_spo2": 0,
        "outside_ppg": 0,
        "read_error": 0,
    }

    for encounter_id, group in labels.groupby("encounter_id"):
        patient_id = patient_by_encounter.get(encounter_id)
        encounter_date = date_by_encounter.get(encounter_id)
        if not patient_id or encounter_id not in demo_by_encounter:
            skipped["missing_patient"] += len(group)
            continue

        try:
            ppg, header = _read_red_ir_ppg(ppg_headers[encounter_id])
        except Exception:
            skipped["read_error"] += len(group)
            continue

        start_dt = _record_start_datetime(encounter_date, header["start_time"])
        fs = float(header["fs"])
        window_size = int(round(float(window_sec) * fs))
        if window_size <= 1 or pd.isna(start_dt):
            skipped["missing_time"] += len(group)
            continue
        try:
            if _is_acdc_mode(normalize):
                ppg_for_windows = _normalize_window(ppg, normalize, fs=fs)
            elif bool(detrend):
                ppg_for_windows = _remove_baseline_drift(ppg, fs, cutoff_hz=detrend_cutoff)
            else:
                ppg_for_windows = ppg
        except Exception:
            skipped["read_error"] += len(group)
            continue

        sample_times = sample_time_by_encounter.get(encounter_id, {})
        for row in group.itertuples(index=False):
            sample_number = int(getattr(row, "sample_number"))
            label_time = getattr(row, "label_time", pd.NaT)
            if pd.isna(label_time):
                label_time = sample_times.get(sample_number, pd.NaT)
            if pd.isna(label_time):
                label_time = getattr(row, "fallback_time", pd.NaT)
            if pd.isna(label_time):
                skipped["missing_time"] += 1
                continue

            label_time = pd.Timestamp(label_time)
            if label_time < start_dt - pd.Timedelta(hours=12):
                label_time = label_time + pd.Timedelta(days=1)
            sao2_value = float(getattr(row, "target"))
            target_value = sao2_value
            spo2_device_count = getattr(row, "spo2_device_count", np.nan)
            spo2_device_min = getattr(row, "spo2_device_min", np.nan)
            spo2_device_max = getattr(row, "spo2_device_max", np.nan)
            spo2_device_std = getattr(row, "spo2_device_std", np.nan)
            spo2_device_range = getattr(row, "spo2_device_range", np.nan)
            label_source = getattr(row, "label_source", target)
            if is_dual_target:
                spo2_series = spo2_by_encounter.get(encounter_id)
                if spo2_series is None or spo2_series.empty:
                    skipped["missing_spo2"] += 1
                    continue
                half_window = pd.Timedelta(seconds=float(window_sec) / 2.0)
                in_window = spo2_series[
                    (spo2_series["label_time"] >= label_time - half_window) &
                    (spo2_series["label_time"] <= label_time + half_window)]
                if in_window.empty:
                    skipped["missing_spo2"] += 1
                    continue
                spo2_values = pd.to_numeric(in_window["target"], errors="coerce").dropna()
                if spo2_values.empty:
                    skipped["missing_spo2"] += 1
                    continue
                if str(spo2_aggregation).lower() == "mean":
                    spo2_value = float(spo2_values.mean())
                else:
                    spo2_value = float(spo2_values.median())
                if not (float(target_min) <= spo2_value <= float(target_max)):
                    skipped["missing_spo2"] += 1
                    continue
                target_value = [sao2_value, spo2_value]
                spo2_device_count = float(in_window["spo2_device_count"].median())
                spo2_device_min = float(in_window["spo2_device_min"].min())
                spo2_device_max = float(in_window["spo2_device_max"].max())
                spo2_device_std = float(spo2_values.std(ddof=0))
                spo2_device_range = spo2_device_max - spo2_device_min
                label_source = (
                    f"bloodgas_so2+spo2_2hz_{str(spo2_aggregation).lower()}"
                    f"_sync{float(window_sec):g}s")

            center = (label_time - start_dt).total_seconds()
            start = int(round(center * fs - window_size / 2.0))
            end = start + window_size
            if start < 0 or end > ppg.shape[1]:
                skipped["outside_ppg"] += 1
                continue

            window = ppg_for_windows[:, start:end]
            if not _is_acdc_mode(normalize):
                try:
                    window = _normalize_window(window, normalize, fs=fs)
                except Exception:
                    skipped["read_error"] += 1
                    continue
            window = _resample_window(window, output_len)
            if not np.all(np.isfinite(window)):
                skipped["read_error"] += 1
                continue

            windows.append(window.astype(np.float32))
            targets.append(target_value)
            demos.append(demo_by_encounter[encounter_id])
            metadata_rows.append({
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "sample_number": sample_number,
                "label_time": str(label_time),
                "ppg_start_time": str(start_dt),
                "ppg_start_index": start,
                "ppg_end_index": end,
                "target": sao2_value,
                "target_sao2": sao2_value,
                "target_spo2": target_value[1] if is_dual_target else np.nan,
                "label_source": label_source,
                "spo2_device_count": spo2_device_count,
                "spo2_device_min": spo2_device_min,
                "spo2_device_max": spo2_device_max,
                "spo2_device_std": spo2_device_std,
                "spo2_device_range": spo2_device_range,
            })

    if not windows:
        raise ValueError(f"OpenOx {target} produced no usable PPG windows. Skipped={skipped}")

    return {
        "X": np.stack(windows, axis=0).astype(np.float32),
        "y": np.asarray(targets, dtype=np.float32).reshape(len(targets), -1),
        "demo": np.stack(demos, axis=0).astype(np.float32),
        "metadata": pd.DataFrame(metadata_rows),
        "skipped": skipped,
    }


def _load_or_build(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec,
        load_cache=True):
    target = normalize_openox_target(target)
    config = get_openox_target_config(target)
    if target_min is None:
        target_min = config["target_min"]
    if target_max is None:
        target_max = config["target_max"]
    cache = _cache_path(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec)
    metadata_path = cache.with_name(cache.stem + "_metadata.csv")
    if load_cache and cache.exists() and metadata_path.exists():
        data = np.load(cache, allow_pickle=False)
        return {
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.float32),
            "demo": data["demo"].astype(np.float32),
            "metadata": pd.read_csv(metadata_path),
            "skipped": {},
            "cache_path": cache,
            "loaded_from_cache": True,
        }

    built = _build_samples(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec)
    np.savez_compressed(cache, X=built["X"], y=built["y"], demo=built["demo"])
    built["metadata"].to_csv(metadata_path, index=False, encoding="utf-8-sig")
    built["cache_path"] = cache
    built["loaded_from_cache"] = False
    return built


def _split_indices(metadata, split_mode, test_size, random_state):
    split_mode = str(split_mode).lower()
    rng = np.random.default_rng(int(random_state))
    n = len(metadata)
    all_indices = np.arange(n)

    if split_mode == "random":
        rng.shuffle(all_indices)
        test_count = max(1, int(round(n * float(test_size))))
        test_idx = all_indices[:test_count]
        train_idx = all_indices[test_count:]
        return np.sort(train_idx), np.sort(test_idx), f"random_test{int(float(test_size) * 100)}_seed{int(random_state)}"

    if split_mode not in ["patient", "encounter"]:
        raise ValueError("openox split_mode must be one of: patient, encounter, random")

    group_column = "patient_id" if split_mode == "patient" else "encounter_id"
    groups = np.asarray(sorted(metadata[group_column].astype(str).unique()))
    rng.shuffle(groups)
    test_count = max(1, int(round(len(groups) * float(test_size))))
    test_groups = set(groups[:test_count])
    test_mask = metadata[group_column].astype(str).isin(test_groups).to_numpy()
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx, f"{split_mode}_test{int(float(test_size) * 100)}_seed{int(random_state)}"


def _apply_input_masks(X_list, modal_enable_dic, ppg_enable_dic):
    X_list = [np.array(x, copy=True) for x in X_list]
    for idx in range(len(X_list)):
        if not int(modal_enable_dic.get(f"modal_{idx + 1}", 1)):
            X_list[idx] = np.zeros_like(X_list[idx])

    for channel_idx in range(X_list[0].shape[1]):
        if not int(ppg_enable_dic.get(f"ppg_{channel_idx + 1}", 1)):
            X_list[0][:, channel_idx] = 0.0
    return X_list


def _safe_scale_name(scale):
    return str(scale).lower().replace(" ", "_").replace("-", "_")


def _detail_from_masks(
        modal_enable_dic,
        ppg_enable_dic,
        target,
        split_label,
        window_sec,
        output_len,
        scale,
        detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec):
    modal_bits = "".join(str(int(modal_enable_dic.get(f"modal_{i}", 1))) for i in range(1, 6))
    ppg_bits = "".join(str(int(ppg_enable_dic.get(f"ppg_{i}", 1))) for i in range(1, 3))
    detail = (
        f"Modal_{modal_bits}_ppg_{ppg_bits}"
        f"_openox_{target}_{split_label}_win{float(window_sec):g}_len{int(output_len)}"
    )
    if str(scale).lower() not in ["none", "raw", "0"]:
        detail += f"_scale{_safe_scale_name(scale)}"
    if bool(detrend):
        detail += f"_detrend{float(detrend_cutoff):g}"
    if str(target).lower() == "spo2":
        source_text = str(spo2_source).lower().replace("_", "")
        range_text = f"r{float(spo2_max_device_range):g}" if float(spo2_max_device_range) > 0 else "rnone"
        stride_text = f"_s{float(spo2_stride_sec):g}" if float(spo2_stride_sec) > 0 else "_sall"
        detail += f"_spo2{source_text}_{str(spo2_aggregation).lower()}_{range_text}{stride_text}"
    elif str(target).lower() == "sao2_spo2":
        source_text = str(spo2_source).lower().replace("_", "")
        range_text = f"r{float(spo2_max_device_range):g}" if float(spo2_max_device_range) > 0 else "rnone"
        detail += (
            f"_spo2{source_text}_{str(spo2_aggregation).lower()}_{range_text}"
            f"_sync{float(window_sec):g}s")
    return detail


def _scale_train_test(X_train, X_test, mode, eps=1e-6):
    mode = str(mode).lower()
    if mode in ["none", "raw", "0"]:
        stats = {
            "mode": "none",
            "mean": np.asarray([0.0], dtype=np.float32),
            "std": np.asarray([1.0], dtype=np.float32),
            "min": np.asarray([0.0], dtype=np.float32),
            "max": np.asarray([1.0], dtype=np.float32),
        }
        return X_train, X_test, stats

    if mode in ["train_global_minmax", "global_minmax", "minmax_global"]:
        x_min = np.asarray([float(np.min(X_train))], dtype=np.float32)
        x_max = np.asarray([float(np.max(X_train))], dtype=np.float32)
        span = np.maximum(x_max - x_min, eps).astype(np.float32)
        X_train_scaled = (X_train - x_min.reshape(1, 1, 1)) / span.reshape(1, 1, 1)
        X_test_scaled = (X_test - x_min.reshape(1, 1, 1)) / span.reshape(1, 1, 1)
        return (
            np.clip(X_train_scaled, 0.0, 1.0).astype(np.float32),
            np.clip(X_test_scaled, 0.0, 1.0).astype(np.float32),
            {
                "mode": "train_global_minmax",
                "mean": x_min,
                "std": span,
                "min": x_min,
                "max": x_max,
            },
        )

    if mode in ["train_global_zscore", "global_zscore", "zscore_global"]:
        mean = np.asarray([float(np.mean(X_train))], dtype=np.float32)
        std = np.asarray([float(np.std(X_train))], dtype=np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return (
            ((X_train - mean.reshape(1, 1, 1)) / std.reshape(1, 1, 1)).astype(np.float32),
            ((X_test - mean.reshape(1, 1, 1)) / std.reshape(1, 1, 1)).astype(np.float32),
            {
                "mode": "train_global_zscore",
                "mean": mean,
                "std": std,
                "min": np.asarray([float(np.min(X_train))], dtype=np.float32),
                "max": np.asarray([float(np.max(X_train))], dtype=np.float32),
            },
        )

    if mode in ["train_channel_zscore", "channel_zscore", "zscore_channel"]:
        mean = np.mean(X_train, axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.std(X_train, axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return (
            ((X_train - mean) / std).astype(np.float32),
            ((X_test - mean) / std).astype(np.float32),
            {
                "mode": "train_channel_zscore",
                "mean": mean.reshape(-1),
                "std": std.reshape(-1),
                "min": np.min(X_train, axis=(0, 2)).astype(np.float32).reshape(-1),
                "max": np.max(X_train, axis=(0, 2)).astype(np.float32).reshape(-1),
            },
        )

    raise ValueError("scale must be one of: train_global_minmax, train_global_zscore, train_channel_zscore, none")


def build_openox_dataset(
        root,
        modal_enable_dic,
        ppg_enable_dic,
        target="so2",
        split_mode="patient",
        test_size=0.2,
        random_state=2,
        window_sec=DEFAULT_OPENOX_WINDOW_SEC,
        output_len=0,
        sampling_rate=DEFAULT_OPENOX_SAMPLING_RATE,
        normalize="window_minmax",
        detrend=True,
        detrend_cutoff=0.3,
        scale="none",
        target_min=None,
        target_max=None,
        spo2_source="continuous_2hz",
        spo2_aggregation="median",
        spo2_max_device_range=10.0,
        spo2_stride_sec=30.0,
        spectral_step=5.0,
        spectral_sigma=25.0,
        spectral_min=600.0,
        spectral_max=1000.0,
        load_cache=True,
        max_samples=0):
    """Build OpenOx red/IR PPG data for SaO2, SpO2, or synchronized dual targets."""
    target = normalize_openox_target(target)
    target_config = get_openox_target_config(target)
    if target_min is None:
        target_min = target_config["target_min"]
    if target_max is None:
        target_max = target_config["target_max"]
    root = _resolve_root(root)
    output_len = _resolve_output_len(window_sec, output_len, sampling_rate)
    effective_sampling_rate = float(output_len) / float(window_sec)
    effective_detrend = bool(detrend) and not _is_acdc_mode(normalize)
    built = _load_or_build(
        root,
        target,
        window_sec,
        output_len,
        normalize,
        target_min,
        target_max,
        effective_detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec,
        load_cache=load_cache)

    X = built["X"]
    y = built["y"]
    demo = built["demo"]
    metadata = built["metadata"].copy()

    if int(max_samples) > 0 and len(y) > int(max_samples):
        keep = np.arange(int(max_samples))
        X = X[keep]
        y = y[keep]
        demo = demo[keep]
        metadata = metadata.iloc[keep].reset_index(drop=True)

    train_idx, test_idx, split_label = _split_indices(metadata, split_mode, test_size, random_state)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("OpenOx split produced an empty train or test set.")

    y_train = y[train_idx].astype(np.float32)
    y_test = y[test_idx].astype(np.float32)
    y_train_norm = ((y_train - target_min) / (target_max - target_min)).astype(np.float32)
    y_test_norm = ((y_test - target_min) / (target_max - target_min)).astype(np.float32)
    y_train_norm = np.clip(y_train_norm, 0.0, 1.0)
    y_test_norm = np.clip(y_test_norm, 0.0, 1.0)

    X_train_ppg = X[train_idx].astype(np.float32)
    X_test_ppg = X[test_idx].astype(np.float32)
    X_train_ppg, X_test_ppg, scale_stats = _scale_train_test(X_train_ppg, X_test_ppg, scale)
    X_train_demo = demo[train_idx].astype(np.float32)
    X_test_demo = demo[test_idx].astype(np.float32)

    X_train_list = [
        X_train_ppg,
        np.zeros((X_train_ppg.shape[0], 6, 52), dtype=np.float32),
        X_train_demo,
        np.zeros((X_train_ppg.shape[0], 6, 8), dtype=np.float32),
        np.zeros((X_train_ppg.shape[0], 1), dtype=np.float32),
    ]
    X_test_list = [
        X_test_ppg,
        np.zeros((X_test_ppg.shape[0], 6, 52), dtype=np.float32),
        X_test_demo,
        np.zeros((X_test_ppg.shape[0], 6, 8), dtype=np.float32),
        np.zeros((X_test_ppg.shape[0], 1), dtype=np.float32),
    ]

    X_train_list = _apply_input_masks(X_train_list, modal_enable_dic, ppg_enable_dic)
    X_test_list = _apply_input_masks(X_test_list, modal_enable_dic, ppg_enable_dic)
    H, wavelengths = build_red_ir_observation_matrix(
        spectral_min=spectral_min,
        spectral_max=spectral_max,
        spectral_step=spectral_step,
        sigma=spectral_sigma)

    target_name = target_config["target_name"]
    target_names = target_config.get("target_names", [target_name])
    target_units = target_config.get("target_units", [target_config["unit"]])
    target_count = len(target_names)
    dataset_name = f"openox_{target}"
    metadata_train = metadata.iloc[train_idx].reset_index(drop=True)
    metadata_test = metadata.iloc[test_idx].reset_index(drop=True)
    if str(split_mode).lower() in ["patient", "encounter"]:
        validation_group_name = "patient_id" if str(split_mode).lower() == "patient" else "encounter_id"
        validation_group = metadata_train[validation_group_name].astype(str).to_numpy()
    else:
        validation_group_name = None
        validation_group = None
    group_counts = metadata.groupby("patient_id").size().to_dict()
    detail = _detail_from_masks(
        modal_enable_dic,
        ppg_enable_dic,
        target,
        split_label,
        window_sec,
        output_len,
        scale,
        effective_detrend,
        detrend_cutoff,
        spo2_source,
        spo2_aggregation,
        spo2_max_device_range,
        spo2_stride_sec)

    return SimpleNamespace(
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        Y_train=y_train_norm,
        Y_test=y_test_norm,
        BG_Min_Max=np.asarray(
            [[target_min] * target_count, [target_max] * target_count],
            dtype=np.float32),
        detail_bin=detail,
        modal_enable_dic=modal_enable_dic.copy(),
        ppg_enable_dic=ppg_enable_dic.copy(),
        observation_matrix=H,
        wavelengths=wavelengths,
        dataset_name=dataset_name,
        target_name=target_name,
        target_unit=target_config["unit"],
        target_names=target_names,
        target_units=target_units,
        target_loss_weights=[1.0, 0.2] if target_count == 2 else None,
        enable_error_grid=False,
        split_mode=split_mode,
        random_state=random_state,
        test_size=test_size,
        openox_window_sec=float(window_sec),
        openox_output_len=int(output_len),
        openox_sampling_rate=float(effective_sampling_rate),
        openox_preprocess=str(normalize),
        openox_detrend=bool(effective_detrend),
        openox_detrend_cutoff=float(detrend_cutoff),
        openox_spo2_source=str(spo2_source).lower(),
        openox_spo2_aggregation=str(spo2_aggregation).lower(),
        openox_spo2_max_device_range=float(spo2_max_device_range),
        openox_spo2_stride_sec=float(spo2_stride_sec),
        openox_scale=scale_stats["mode"],
        openox_scale_mean=scale_stats["mean"],
        openox_scale_std=scale_stats["std"],
        openox_scale_min=scale_stats["min"],
        openox_scale_max=scale_stats["max"],
        subject_counts=group_counts,
        validation_group=validation_group,
        validation_group_name=validation_group_name,
        feature_names=FEATURE_NAMES,
        demo_feature_names=DEMO_FEATURE_NAMES,
        metadata=metadata,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
        cache_path=built["cache_path"],
        loaded_from_cache=built["loaded_from_cache"],
        skipped=built["skipped"])
