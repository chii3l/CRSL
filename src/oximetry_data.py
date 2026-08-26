from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


SUBJECT_IDS = ["100001", "100002", "100003", "100004", "100005", "100006"]
FEATURE_NAMES = ["left-r", "left-g", "left-b", "right-r", "right-g", "right-b"]
SPO2_TARGET_ROWS = {
    "spo2_1": [0],
    "sp02_1": [0],
    "spo2_2": [1],
    "sp02_2": [1],
    "spo2_4": [3],
    "sp02_4": [3],
    "spo2_5": [4],
    "sp02_5": [4],
    "mean": [0, 1, 3, 4],
}


def build_rgb_gaussian_observation_matrix(
        spectral_min=430.0,
        spectral_max=680.0,
        spectral_step=5.0,
        sigma=35.0):
    """Approximate smartphone RGB channel responses with normalized Gaussians."""
    if spectral_step <= 0:
        raise ValueError("spectral_step must be positive")
    wavelengths = np.arange(
        float(spectral_min),
        float(spectral_max) + float(spectral_step) * 0.5,
        float(spectral_step),
        dtype=np.float32)
    centers = [620.0, 530.0, 470.0, 620.0, 530.0, 470.0]
    rows = []
    sigma = float(sigma)
    for center in centers:
        row = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
        row = np.clip(row, 0.0, None)
        denom = row.sum()
        if denom > 0:
            row = row / denom
        rows.append(row.astype(np.float32))
    return np.stack(rows, axis=0), wavelengths


def _parse_test_subject(test_subject):
    test_subject = str(test_subject).lower()
    if test_subject in ["last", "default"]:
        return SUBJECT_IDS[-1]
    if test_subject not in SUBJECT_IDS:
        raise ValueError(
            f"Unknown oximetry test subject: {test_subject}. "
            f"Valid values: {', '.join(SUBJECT_IDS)}, last")
    return test_subject


def _load_preprocessed(root):
    h5_path = Path(root) / "data" / "preprocessed" / "all_uw_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Oximetry H5 file not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        data = f["dataset"][:].astype(np.float32)
        groundtruth = f["groundtruth"][:].astype(np.float32)
    return data, groundtruth


def _target_from_groundtruth(groundtruth, target_name):
    key = str(target_name).lower()
    if key not in SPO2_TARGET_ROWS:
        valid = ", ".join(sorted(SPO2_TARGET_ROWS))
        raise ValueError(f"Unknown SpO2 target '{target_name}'. Valid values: {valid}")
    rows = SPO2_TARGET_ROWS[key]
    selected = groundtruth[rows]
    selected = np.where(selected > 0, selected, np.nan)
    valid = np.isfinite(selected)
    counts = valid.sum(axis=0)
    summed = np.where(valid, selected, 0.0).sum(axis=0)
    target = np.full(selected.shape[1], np.nan, dtype=np.float32)
    np.divide(summed, counts, out=target, where=counts > 0)
    return target.astype(np.float32)


def _valid_signal_length(signal):
    valid = np.where(np.any(signal != 0, axis=0))[0]
    if len(valid) == 0:
        return 0
    return int(valid[-1]) + 1


def _valid_target_length(target):
    valid = np.where(np.isfinite(target) & (target > 0))[0]
    if len(valid) == 0:
        return 0
    return int(valid[-1]) + 1


def _normalize_recording(signal, eps=1e-6):
    mins = signal.min(axis=1, keepdims=True)
    maxs = signal.max(axis=1, keepdims=True)
    return (signal - mins) / np.maximum(maxs - mins, eps)


def _normalize_window(window, mode, eps=1e-6):
    mode = str(mode).lower()
    if mode in ["none", "raw"]:
        return window
    if mode in ["window_minmax", "window", "minmax"]:
        mins = window.min(axis=1, keepdims=True)
        maxs = window.max(axis=1, keepdims=True)
        return (window - mins) / np.maximum(maxs - mins, eps)
    raise ValueError("normalize must be one of: window_minmax, recording_minmax, none")


def _apply_input_masks(X_list, modal_enable_dic, ppg_enable_dic):
    X_list = [np.array(x, copy=True) for x in X_list]
    for idx in range(len(X_list)):
        if not int(modal_enable_dic.get(f"modal_{idx + 1}", 1)):
            X_list[idx] = np.zeros_like(X_list[idx])

    channel_to_mask = {
        "ppg_1": 0,
        "ppg_2": 1,
        "ppg_3": 2,
        "ppg_4": 3,
        "ppg_5": 4,
        "ppg_6": 5,
    }
    for key, channel_idx in channel_to_mask.items():
        if not int(ppg_enable_dic.get(key, 1)):
            X_list[0][:, channel_idx] = 0.0
            X_list[3][:, channel_idx] = 0.0
    return X_list


def _detail_from_masks(modal_enable_dic, ppg_enable_dic):
    modal_bits = "".join(str(int(modal_enable_dic.get(f"modal_{i}", 1))) for i in range(1, 6))
    ppg_bits = "".join(str(int(ppg_enable_dic.get(f"ppg_{i}", 1))) for i in range(1, 7))
    return f"Modal_{modal_bits}_ppg_{ppg_bits}"


def build_oximetry_dataset(
        root,
        modal_enable_dic,
        ppg_enable_dic,
        split_mode="subject",
        test_subject="100006",
        random_test_size=0.2,
        random_state=2,
        window_sec=10,
        stride_sec=10,
        target="spo2_5",
        normalize="window_minmax",
        target_min=70.0,
        target_max=100.0,
        spectral_step=5.0,
        rgb_sigma=35.0,
        spectral_min=430.0,
        spectral_max=680.0):
    """Build smartphone RGB-PPG windows using Masimo Radical-7 by default."""
    data, groundtruth = _load_preprocessed(root)
    fps = 30
    window_size = int(round(float(window_sec) * fps))
    stride = int(round(float(stride_sec) * fps))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    split_mode = str(split_mode).lower()
    if split_mode not in ["subject", "random"]:
        raise ValueError("split_mode must be one of: subject, random")
    test_subject = _parse_test_subject(test_subject)
    test_index = SUBJECT_IDS.index(test_subject)
    all_windows = []
    all_labels = []
    all_subjects = []
    subject_counts = {}

    for subject_index, subject_id in enumerate(SUBJECT_IDS):
        signal = data[subject_index]
        target_values = _target_from_groundtruth(groundtruth[subject_index], target)
        signal_len = _valid_signal_length(signal)
        target_len = _valid_target_length(target_values)
        clip_seconds = min(signal_len // fps, target_len)
        clip_frames = clip_seconds * fps
        if clip_frames < window_size:
            continue

        signal = signal[:, :clip_frames]
        target_values = target_values[:clip_seconds]
        if str(normalize).lower() in ["recording_minmax", "recording"]:
            signal = _normalize_recording(signal)

        windows = []
        labels = []
        for start in range(0, clip_frames - window_size + 1, stride):
            end = start + window_size
            label_start = start // fps
            label_end = end // fps
            label_values = target_values[label_start:label_end]
            label_values = label_values[np.isfinite(label_values) & (label_values > 0)]
            if len(label_values) == 0:
                continue
            window = signal[:, start:end]
            window = _normalize_window(window, normalize).astype(np.float32)
            label = float(label_values.mean())
            if not (target_min <= label <= target_max):
                continue
            windows.append(window)
            labels.append(label)

        if not windows:
            continue
        windows = np.stack(windows, axis=0)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        subject_counts[subject_id] = int(len(labels))
        all_windows.append(windows)
        all_labels.append(labels)
        all_subjects.extend([subject_id] * len(labels))

    if not all_windows:
        raise ValueError("Oximetry dataset produced no windows.")

    all_windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    all_subjects = np.asarray(all_subjects)

    if split_mode == "subject":
        test_mask = all_subjects == test_subject
        train_mask = ~test_mask
        split_label = f"test_{test_subject}"
    else:
        if not (0.0 < float(random_test_size) < 1.0):
            raise ValueError("random_test_size must be between 0 and 1")
        rng = np.random.default_rng(int(random_state))
        indices = np.arange(len(all_labels))
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * float(random_test_size))))
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        train_mask = np.zeros(len(indices), dtype=bool)
        test_mask = np.zeros(len(indices), dtype=bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        split_label = f"random_test{int(float(random_test_size) * 100)}_seed{int(random_state)}"

    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Oximetry dataset split produced an empty train or test set.")

    X_train_ppg = all_windows[train_mask].astype(np.float32)
    X_test_ppg = all_windows[test_mask].astype(np.float32)
    y_train = all_labels[train_mask].astype(np.float32)
    y_test = all_labels[test_mask].astype(np.float32)
    if split_mode == "subject":
        validation_group = all_subjects[train_mask].astype(str)
        validation_group_name = "subject_id"
    else:
        validation_group = None
        validation_group_name = None

    y_train_norm = ((y_train - target_min) / (target_max - target_min)).astype(np.float32)
    y_test_norm = ((y_test - target_min) / (target_max - target_min)).astype(np.float32)
    y_train_norm = np.clip(y_train_norm, 0.0, 1.0)
    y_test_norm = np.clip(y_test_norm, 0.0, 1.0)

    def zero_modal(shape):
        return np.zeros((shape[0],) + shape[1:], dtype=np.float32)

    X_train_list = [
        X_train_ppg,
        np.zeros((X_train_ppg.shape[0], 6, 52), dtype=np.float32),
        np.zeros((X_train_ppg.shape[0], 9), dtype=np.float32),
        np.zeros((X_train_ppg.shape[0], 6, 8), dtype=np.float32),
        np.zeros((X_train_ppg.shape[0], 1), dtype=np.float32),
    ]
    X_test_list = [
        X_test_ppg,
        np.zeros((X_test_ppg.shape[0], 6, 52), dtype=np.float32),
        np.zeros((X_test_ppg.shape[0], 9), dtype=np.float32),
        np.zeros((X_test_ppg.shape[0], 6, 8), dtype=np.float32),
        np.zeros((X_test_ppg.shape[0], 1), dtype=np.float32),
    ]

    X_train_list = _apply_input_masks(X_train_list, modal_enable_dic, ppg_enable_dic)
    X_test_list = _apply_input_masks(X_test_list, modal_enable_dic, ppg_enable_dic)
    H, wavelengths = build_rgb_gaussian_observation_matrix(
        spectral_min=spectral_min,
        spectral_max=spectral_max,
        spectral_step=spectral_step,
        sigma=rgb_sigma)

    detail = (
        f"{_detail_from_masks(modal_enable_dic, ppg_enable_dic)}"
        f"_oximetry_{split_label}_target_{str(target).lower()}"
        f"_win{int(window_sec)}_stride{int(stride_sec)}"
    )
    return SimpleNamespace(
        X_train_list=X_train_list,
        X_test_list=X_test_list,
        Y_train=y_train_norm,
        Y_test=y_test_norm,
        BG_Min_Max=np.asarray([[target_min], [target_max]], dtype=np.float32),
        detail_bin=detail,
        modal_enable_dic=modal_enable_dic.copy(),
        ppg_enable_dic=ppg_enable_dic.copy(),
        observation_matrix=H,
        wavelengths=wavelengths,
        dataset_name="oximetry",
        target_name="SpO2",
        target_unit="%",
        target_source=str(target).lower(),
        reference_device=(
            "Masimo Radical-7 Rainbow II"
            if str(target).lower() in ["spo2_5", "sp02_5"] else str(target)),
        enable_error_grid=False,
        subject_counts=subject_counts,
        validation_group=validation_group,
        validation_group_name=validation_group_name,
        split_mode=split_mode,
        test_subject=test_subject,
        random_test_size=random_test_size,
        random_state=random_state,
        feature_names=FEATURE_NAMES)
