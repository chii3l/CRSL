# Run traditional machine-learning baselines with optional H-derived spectral features.
#
# This script treats the proposed H module as a measurement-aware feature
# extractor for non-differentiable regressors such as RF and SVR.

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.run_ablation import build_base_params, build_dataset  # noqa: E402
from src.run_paper_baselines import build_arg_parser, make_dataset_args  # noqa: E402


ML_MODELS = ["rf", "svr", "linear_svr", "extratrees", "gbr"]
FEATURE_SETS = ["raw", "h", "raw_h"]


def save_results_incremental(all_rows, output_dir, reason=""):
    if not all_rows:
        return

    result_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "ml_h_baseline_results.csv"
    xlsx_path = output_dir / "ml_h_baseline_results.xlsx"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            result_df.to_excel(writer, sheet_name="all_results", index=False)
            if "fold" in result_df.columns:
                means = result_df[result_df["fold"].astype(str) == "mean"].copy()
            else:
                means = pd.DataFrame()
            means.to_excel(writer, sheet_name="mean_results", index=False)
    except PermissionError:
        print(f"  Warning: could not update {xlsx_path} because it is open or locked.")

    if reason:
        print(f"  Saved partial ML results ({reason}) -> {csv_path}")


def _safe_stats(values, axis=-1):
    values = np.asarray(values, dtype=np.float32)
    q25 = np.quantile(values, 0.25, axis=axis)
    q75 = np.quantile(values, 0.75, axis=axis)
    return [
        values.mean(axis=axis),
        values.std(axis=axis),
        values.min(axis=axis),
        values.max(axis=axis),
        np.median(values, axis=axis),
        q25,
        q75,
        q75 - q25,
        np.sqrt(np.mean(values ** 2, axis=axis)),
    ]


def raw_ppg_features(x):
    """Handcrafted time-domain features from PPG windows with shape (N, C, T)."""
    x = np.asarray(x, dtype=np.float32)
    n_samples, n_channels, n_steps = x.shape
    features = []
    names = []

    stat_names = ["mean", "std", "min", "max", "median", "q25", "q75", "iqr", "rms"]
    for stat_name, stat_value in zip(stat_names, _safe_stats(x, axis=2)):
        features.append(stat_value)
        names.extend([f"raw_{stat_name}_ch{idx + 1}" for idx in range(n_channels)])

    diff = np.diff(x, axis=2)
    diff_stats = [
        np.mean(np.abs(diff), axis=2),
        np.std(diff, axis=2),
        np.max(np.abs(diff), axis=2),
    ]
    diff_names = ["absdiff_mean", "diff_std", "absdiff_max"]
    for stat_name, stat_value in zip(diff_names, diff_stats):
        features.append(stat_value)
        names.extend([f"raw_{stat_name}_ch{idx + 1}" for idx in range(n_channels)])

    slope = (x[:, :, -1] - x[:, :, 0]) / max(n_steps - 1, 1)
    features.append(slope)
    names.extend([f"raw_slope_ch{idx + 1}" for idx in range(n_channels)])

    pair_corrs = []
    pair_names = []
    centered = x - x.mean(axis=2, keepdims=True)
    denom = np.sqrt(np.sum(centered ** 2, axis=2)).clip(min=1e-8)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            corr = np.sum(centered[:, i] * centered[:, j], axis=1) / (denom[:, i] * denom[:, j])
            pair_corrs.append(corr[:, None])
            pair_names.append(f"raw_corr_ch{i + 1}_ch{j + 1}")
    if pair_corrs:
        features.append(np.concatenate(pair_corrs, axis=1))
        names.extend(pair_names)

    return np.concatenate([f.reshape(n_samples, -1) for f in features], axis=1), names


def gaussian_basis(num_basis, spectral_dim):
    grid = np.linspace(0.0, 1.0, int(spectral_dim), dtype=np.float32)[None, :]
    centers = np.linspace(0.0, 1.0, int(num_basis), dtype=np.float32)[:, None]
    width = max(1.0 / max(int(num_basis) - 1, 1), 0.08)
    basis = np.exp(-0.5 * ((grid - centers) / width) ** 2).astype(np.float32) + 1e-4
    basis /= np.maximum(basis.sum(axis=1, keepdims=True), 1e-8)
    return basis


def h_spectral_features(x, h_matrix, wavelengths=None, num_basis=16, ridge_alpha=1e-3):
    """H-guided spectral proxy features for non-neural regressors.

    The latent spectrum is represented by non-negative combinations of fixed
    Gaussian bases. Coefficients are estimated by ridge projection from each
    PPG observation vector.
    """
    x = np.asarray(x, dtype=np.float32)
    h_matrix = np.asarray(h_matrix, dtype=np.float32)
    n_samples, n_channels, _ = x.shape
    if h_matrix.shape[0] != n_channels:
        raise ValueError(f"H channels {h_matrix.shape[0]} do not match PPG channels {n_channels}")

    basis = gaussian_basis(num_basis, h_matrix.shape[1])
    mixing = h_matrix @ basis.T
    gram = mixing.T @ mixing + float(ridge_alpha) * np.eye(num_basis, dtype=np.float32)
    projector = np.linalg.solve(gram, mixing.T).astype(np.float32)
    coeff = np.einsum("kc,nct->nkt", projector, x).astype(np.float32)
    coeff = np.clip(coeff, 0.0, None)
    recon = np.einsum("ck,nkt->nct", mixing, coeff).astype(np.float32)
    residual = x - recon

    features = []
    names = []

    coeff_stat_names = ["mean", "std", "min", "max", "median", "q25", "q75", "iqr", "rms"]
    for stat_name, stat_value in zip(coeff_stat_names, _safe_stats(coeff, axis=2)):
        features.append(stat_value)
        names.extend([f"h_coeff_{stat_name}_b{idx + 1}" for idx in range(num_basis)])

    coeff_diff = np.diff(coeff, axis=2)
    coeff_diff_features = [
        np.mean(np.abs(coeff_diff), axis=2),
        np.std(coeff_diff, axis=2),
    ]
    for stat_name, stat_value in zip(["absdiff_mean", "diff_std"], coeff_diff_features):
        features.append(stat_value)
        names.extend([f"h_coeff_{stat_name}_b{idx + 1}" for idx in range(num_basis)])

    residual_features = [
        np.mean(np.abs(residual), axis=2),
        np.sqrt(np.mean(residual ** 2, axis=2)),
        np.mean(residual, axis=2),
        np.std(residual, axis=2),
    ]
    for stat_name, stat_value in zip(["mae", "rmse", "bias", "std"], residual_features):
        features.append(stat_value)
        names.extend([f"h_recon_{stat_name}_ch{idx + 1}" for idx in range(n_channels)])

    mean_coeff = coeff.mean(axis=2)
    mean_spectrum = mean_coeff @ basis
    if wavelengths is None:
        wavelengths = np.arange(h_matrix.shape[1], dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(1, -1)
    spec_mass = np.maximum(mean_spectrum.sum(axis=1, keepdims=True), 1e-8)
    centroid = (mean_spectrum * wavelengths).sum(axis=1, keepdims=True) / spec_mass
    spread = np.sqrt(((wavelengths - centroid) ** 2 * mean_spectrum).sum(axis=1, keepdims=True) / spec_mass)
    spec_smooth = np.mean(np.diff(mean_spectrum, axis=1) ** 2, axis=1, keepdims=True)
    features.extend([centroid, spread, spec_smooth])
    names.extend(["h_spectrum_centroid", "h_spectrum_spread", "h_spectrum_smoothness"])

    return np.concatenate([f.reshape(n_samples, -1) for f in features], axis=1), names


def build_features(x, feature_set, h_matrix, wavelengths, num_basis, ridge_alpha):
    raw_features, raw_names = raw_ppg_features(x)
    h_features, h_names = h_spectral_features(
        x,
        h_matrix,
        wavelengths=wavelengths,
        num_basis=num_basis,
        ridge_alpha=ridge_alpha)
    if feature_set == "raw":
        return raw_features, raw_names
    if feature_set == "h":
        return h_features, h_names
    if feature_set == "raw_h":
        return np.concatenate([raw_features, h_features], axis=1), raw_names + h_names
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_regressor(name, seed):
    name = str(name).lower()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt")
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt")
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.03, gamma="scale"))
    if name == "linear_svr":
        return make_pipeline(StandardScaler(), LinearSVR(C=1.0, epsilon=0.03, random_state=seed, max_iter=20000))
    if name == "gbr":
        return GradientBoostingRegressor(
            random_state=seed,
            n_estimators=800,
            learning_rate=0.02,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8)
    raise ValueError(f"Unknown ML model: {name}")


def inverse_target(y_norm, min_max):
    y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
    min_max = np.asarray(min_max, dtype=np.float32).reshape(-1)
    target_min, target_max = float(min_max[0]), float(min_max[-1])
    return y_norm * (target_max - target_min) + target_min


def as_target_matrix(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def target_ranges(min_max, n_targets):
    ranges = np.asarray(min_max, dtype=np.float32)
    if ranges.ndim == 1:
        if ranges.size != 2 * int(n_targets):
            raise ValueError(
                f"Expected {2 * int(n_targets)} target-range values, got {ranges.size}")
        ranges = ranges.reshape(2, int(n_targets))
    else:
        ranges = ranges.reshape(ranges.shape[0], -1)

    if ranges.shape[0] == 2 and ranges.shape[1] >= int(n_targets):
        target_min = ranges[0, :int(n_targets)]
        target_max = ranges[1, :int(n_targets)]
    elif ranges.shape[1] == 2 and ranges.shape[0] >= int(n_targets):
        target_min = ranges[:int(n_targets), 0]
        target_max = ranges[:int(n_targets), 1]
    else:
        raise ValueError(
            f"Target range shape {ranges.shape} is incompatible with {n_targets} targets")

    if np.any(target_max <= target_min):
        raise ValueError(f"Invalid target ranges: min={target_min}, max={target_max}")
    return target_min.astype(np.float32), target_max.astype(np.float32)


def inverse_target_matrix(y_norm, min_max):
    y_norm = as_target_matrix(y_norm)
    target_min, target_max = target_ranges(min_max, y_norm.shape[1])
    return y_norm * (target_max - target_min)[None, :] + target_min[None, :]


def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def add_ml_args(parser):
    parser.add_argument("--ml-models", nargs="+", choices=ML_MODELS, default=["rf", "svr"])
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=["raw", "raw_h"])
    parser.add_argument("--spectral-basis-count", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--ml-run-tag", default="")
    parser.add_argument(
        "--glucose-spo2-target",
        action="store_true",
        help="Treat the cached glucose SpO2 value as a second supervised target.")
    parser.add_argument(
        "--ml-targets",
        nargs="+",
        default=["all"],
        help="Targets to fit after loading the dataset, for example: all, BG, or SpO2.")
    return parser


def describe_targets(dataset_name, provider, dataset_dic, args, n_targets):
    target_names = dataset_dic.get(
        "target_names", getattr(provider, "target_names", None))
    target_units = dataset_dic.get(
        "target_units", getattr(provider, "target_units", None))
    if not target_names:
        target_names = [dataset_dic.get(
            "target_name", getattr(provider, "target_name", "target"))]
    if not target_units:
        target_units = [dataset_dic.get(
            "target_unit", getattr(provider, "target_unit", ""))]
    target_names = [str(value) for value in target_names]
    target_units = [str(value) for value in target_units]
    while len(target_names) < int(n_targets):
        target_names.append(f"target_{len(target_names)}")
    while len(target_units) < int(n_targets):
        target_units.append("")

    if dataset_name == "glucose" and args.glucose_spo2_target and n_targets >= 2:
        target_names[0] = "BG"
        target_names[1] = "SpO2"
        target_units[0] = "mmol/L"
        target_units[1] = "%"
        label_sources = ["BG_calibration", "glucose_blood_oxygen_saturation"]
    if dataset_name == "oximetry":
        if str(args.oximetry_target).lower() == "spo2_5":
            label_sources = ["measured_Masimo_Radical7_SpO2_5"]
        else:
            label_sources = [f"measured_groundtruth_{args.oximetry_target}"]
    elif dataset_name == "openox_sao2_spo2":
        label_sources = [
            "bloodgas_SO2",
            f"measured_{args.openox_spo2_source}_{args.openox_spo2_aggregation}"]
    elif dataset_name == "openox_spo2":
        label_sources = [
            f"measured_{args.openox_spo2_source}_"
            f"{args.openox_spo2_aggregation}"]
    elif dataset_name in ["openox_sao2", "openox_so2"]:
        label_sources = ["bloodgas_SO2"]
    elif not (dataset_name == "glucose" and args.glucose_spo2_target and n_targets >= 2):
        label_sources = ["BG_calibration"]
    while len(label_sources) < int(n_targets):
        label_sources.append(label_sources[-1])
    return (
        target_names[:int(n_targets)],
        target_units[:int(n_targets)],
        label_sources[:int(n_targets)])


def select_target_indices(target_names, requested_targets):
    requested = [str(value).strip().lower() for value in requested_targets]
    if not requested or "all" in requested:
        return list(range(len(target_names)))
    name_to_index = {
        str(name).strip().lower(): index
        for index, name in enumerate(target_names)
    }
    missing = [name for name in requested if name not in name_to_index]
    if missing:
        raise ValueError(
            f"Requested ML targets {missing} are unavailable; loaded targets are {target_names}")
    return list(dict.fromkeys(name_to_index[name] for name in requested))


def run(args):
    output_root = Path("outputs") / datetime.now().strftime("%Y-%m-%d")
    run_tag = args.ml_run_tag or f"ml_h_{datetime.now().strftime('%H-%M-%S')}"
    output_dir = output_root / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for dataset_name in args.datasets:
        dataset_args = make_dataset_args(args, dataset_name)
        dataset_args.glucose_aux_spo2 = bool(
            dataset_name == "glucose" and args.glucose_spo2_target)
        dataset_args.glucose_keep_spo2_input = False
        provider, dataset_dic = build_dataset(dataset_args)
        base_params = build_base_params(provider, dataset_args)
        from src.data_process_workingcopy import build_observation_matrix

        if "observation_matrix" in base_params:
            h_matrix = np.asarray(base_params["observation_matrix"], dtype=np.float32)
            wavelengths = np.asarray(base_params.get("wavelengths", np.arange(h_matrix.shape[1])), dtype=np.float32)
        else:
            h_matrix, wavelengths = build_observation_matrix(
                base_params["spectral_excel_path"],
                spectral_min=base_params.get("spectral_min", None),
                spectral_max=base_params.get("spectral_max", None),
                spectral_step=base_params.get("spectral_step", 5.0),
                normalize=base_params.get("spectral_normalize", "sum"),
                legacy_channel_order=base_params.get("legacy_channel_order", None),
                use_abs=base_params.get("use_abs", True))

        x_train_ppg = np.asarray(dataset_dic["X_train"][0], dtype=np.float32)
        x_test_ppg = np.asarray(dataset_dic["X_test"][0], dtype=np.float32)
        y_train = as_target_matrix(dataset_dic["Y_train"])
        y_test_norm = as_target_matrix(dataset_dic["Y_test"])
        if y_train.shape[1] != y_test_norm.shape[1]:
            raise ValueError(
                f"Train/test target dimensions differ: {y_train.shape} vs {y_test_norm.shape}")
        y_test_true = inverse_target_matrix(y_test_norm, dataset_dic["BG_Min_Max"])
        target_min, target_max = target_ranges(
            dataset_dic["BG_Min_Max"], y_train.shape[1])
        target_names, target_units, label_sources = describe_targets(
            dataset_name, provider, dataset_dic, args, y_train.shape[1])
        selected_target_indices = select_target_indices(
            target_names, args.ml_targets)
        target_text = ", ".join(
            f"{target_names[index]} ({target_units[index]})"
            if target_units[index] else target_names[index]
            for index in selected_target_indices)

        print(
            f"\nDataset={dataset_name} | targets={target_text} "
            f"| sources={label_sources} | train={len(y_train)} test={len(y_test_true)} "
            f"| PPG={x_train_ppg.shape[1:]} | H={h_matrix.shape}")

        feature_cache = {}
        for feature_set in args.feature_sets:
            x_train_feat, feature_names = build_features(
                x_train_ppg,
                feature_set,
                h_matrix,
                wavelengths,
                args.spectral_basis_count,
                args.ridge_alpha)
            x_test_feat, _ = build_features(
                x_test_ppg,
                feature_set,
                h_matrix,
                wavelengths,
                args.spectral_basis_count,
                args.ridge_alpha)
            feature_cache[feature_set] = (x_train_feat, x_test_feat, feature_names)
            feature_file = output_dir / f"{dataset_name}_{feature_set}_feature_names.txt"
            feature_file.write_text("\n".join(feature_names), encoding="utf-8")
            print(f"  features={feature_set}: dim={x_train_feat.shape[1]}")

        train_indices = np.arange(len(y_train))
        validation_group = dataset_dic.get("validation_group", None)
        if validation_group is not None:
            groups = np.asarray(validation_group).astype(str)
            if len(groups) != len(train_indices):
                raise ValueError(
                    "validation_group length must match ML training samples: "
                    f"{len(groups)} vs {len(train_indices)}")
            splitter = GroupShuffleSplit(
                n_splits=int(args.n_split),
                random_state=2,
                test_size=0.25)
            split_indices = list(splitter.split(train_indices, groups=groups))
            print(
                f"  ML fit split uses disjoint {dataset_dic.get('validation_group_name', 'group')} "
                f"groups (n={len(np.unique(groups))}).")
        else:
            splitter = ShuffleSplit(
                n_splits=int(args.n_split),
                random_state=2,
                test_size=0.25)
            split_indices = list(splitter.split(train_indices))

        for model_name in args.ml_models:
            for feature_set in args.feature_sets:
                x_train_feat, x_test_feat, _ = feature_cache[feature_set]
                fold_rows = {target_idx: [] for target_idx in selected_target_indices}
                for fold_idx, (fit_idx, _) in enumerate(split_indices):
                    for target_idx in selected_target_indices:
                        target_name = target_names[target_idx]
                        model = make_regressor(model_name, seed=2 + fold_idx)
                        model.fit(x_train_feat[fit_idx], y_train[fit_idx, target_idx])
                        pred_norm = np.asarray(model.predict(x_test_feat), dtype=np.float32)
                        pred = (
                            pred_norm * (target_max[target_idx] - target_min[target_idx])
                            + target_min[target_idx])
                        metrics = compute_metrics(y_test_true[:, target_idx], pred)
                        row = {
                            "dataset": dataset_name,
                            "target": target_name,
                            "target_unit": target_units[target_idx],
                            "label_source": label_sources[target_idx],
                            "target_index": target_idx,
                            "model": model_name,
                            "feature_set": feature_set,
                            "fold": fold_idx,
                            "n_train_fit": int(len(fit_idx)),
                            "n_test": int(len(y_test_true)),
                            "feature_dim": int(x_train_feat.shape[1]),
                            **metrics,
                        }
                        fold_rows[target_idx].append(row)
                        all_rows.append(row)
                        print(
                            f"  {model_name:10s} {feature_set:5s} {target_name:6s} "
                            f"fold={fold_idx} RMSE={metrics['RMSE']:.4f} "
                            f"MAE={metrics['MAE']:.4f} R2={metrics['R2']:.4f}")
                        save_results_incremental(
                            all_rows,
                            output_dir,
                            reason=(
                                f"{dataset_name}/{model_name}/{feature_set}/"
                                f"{target_name}/fold{fold_idx}"))

                for target_idx in selected_target_indices:
                    target_name = target_names[target_idx]
                    fold_df = pd.DataFrame(fold_rows[target_idx])
                    numeric_df = fold_df.drop(columns=["fold"]).select_dtypes(include=[np.number])
                    mean_row = numeric_df.mean().to_dict()
                    std_row = numeric_df.std(ddof=0).to_dict()
                    for fold_label, summary_values in [("mean", mean_row), ("std", std_row)]:
                        all_rows.append({
                            "dataset": dataset_name,
                            "target": target_name,
                            "target_unit": target_units[target_idx],
                            "label_source": label_sources[target_idx],
                            "target_index": target_idx,
                            "model": model_name,
                            "feature_set": feature_set,
                            "fold": fold_label,
                            **summary_values,
                        })
                    save_results_incremental(
                        all_rows,
                        output_dir,
                        reason=(
                            f"{dataset_name}/{model_name}/{feature_set}/"
                            f"{target_name}/summary"))

    csv_path = output_dir / "ml_h_baseline_results.csv"
    xlsx_path = output_dir / "ml_h_baseline_results.xlsx"
    save_results_incremental(all_rows, output_dir)
    print(f"\nSaved ML + H baseline results to: {csv_path}")
    print(f"Saved Excel summary to: {xlsx_path}")


def main():
    parser = add_ml_args(build_arg_parser())
    args = parser.parse_args()
    if args.dry_run:
        print("ML + H plan")
        print("datasets:", args.datasets)
        print("models:", args.ml_models)
        print("feature_sets:", args.feature_sets)
        print("n_split:", args.n_split)
        print("glucose_spo2_target:", args.glucose_spo2_target)
        print("ml_targets:", args.ml_targets)
        return
    run(args)


if __name__ == "__main__":
    main()
