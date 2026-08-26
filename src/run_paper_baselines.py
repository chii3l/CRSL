# Run the final paper baseline matrix on glucose and oximetry datasets.
#
# This script intentionally keeps a small, stable experiment surface:
# ResNet, GRU/BiLSTM/LSTM-Attention, ShallowConvNet, PaPaGei, TSLANet, CSFM, and Medformer,
# each as baseline and baseline + H.

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.run_ablation import (  # noqa: E402
    DEFAULT_DATA_PATH,
    OPENOX_DATASET_TARGETS,
    REPO_ROOT,
    SRC_DIR,
    build_base_params,
    build_dataset,
    get_ablation_specs,
)
from src.data_train_torch import Training_Process  # noqa: E402


BASELINE_PAIRS = {
    "resnet": ("A", "H"),
    "gru": ("Y", "Z"),
    "bilstm": ("CI", "CJ"),
    "lstm_attention": ("CK", "CL"),
    "shallowconvnet": ("BM", "BN"),
    "tslanet": ("CC", "CD"),
    "csfm": ("BY", "BZ"),
    "medformer": ("CA", "CB"),
}

PAPAGEI_PAIRS = {
    "scratch": ("CG", "CH"),
    "pretrained": ("BW", "BX"),
}

DEFAULT_BASELINES = [
    "resnet",
    "gru",
    "bilstm",
    "lstm_attention",
    "shallowconvnet",
    "papagei",
    "tslanet",
    "csfm",
    "medformer",
]

GLUCOSE_H_UPDATES = {
    "model_name": "physics_spectral",
    "enable_token_moe": True,
    "token_moe_mode": "anonymous",
    "num_experts": 4,
    "token_moe_dropout": 0.1,
    "spectral_fusion_mode": "concat",
    "spectral_generator_mode": "basis",
    "spectral_basis_count": 16,
    "spectral_basis_dropout": 0.1,
    "lambda_obs": 1.0,
    "lambda_smooth": 0.5,
    "lambda_decorr": 0.5,
}

OXIMETRY_H_UPDATES = {
    "model_name": "physics_spectral",
    "enable_token_moe": True,
    "token_moe_mode": "anonymous",
    "num_experts": 4,
    "token_moe_dropout": 0.1,
    "spectral_fusion_mode": "concat",
    "spectral_generator_mode": "basis",
    "spectral_basis_count": 16,
    "spectral_basis_dropout": 0.1,
    "lambda_obs": 0.1,
    "lambda_smooth": 0.01,
    "lambda_decorr": 0.001,
}

OXIMETRY_FOUNDATION_TUNING = {
    "csfm": {
        "initial_lr": 0.0005,
        "weight_decay": 0.0001,
        "dropout_rate": 0.15,
        "ppg_csfm_signal_size": 300,
        "ppg_csfm_patch_size": 25,
        "ppg_csfm_hidden_dim": 128,
        "ppg_csfm_depth": 2,
        "ppg_csfm_heads": 4,
        "ppg_csfm_mlp_dim": 384,
        "ppg_csfm_dim_head": 32,
        "ppg_csfm_freeze_encoder": False,
    },
    "medformer": {
        "initial_lr": 0.0005,
        "weight_decay": 0.0001,
        "dropout_rate": 0.10,
        "ppg_recent_hidden": 128,
        "ppg_recent_layers": 2,
        "ppg_recent_heads": 4,
        "ppg_recent_patch_sizes": "15,30,60",
        "ppg_medformer_stride_sizes": "5,10,20",
        "ppg_medformer_d_ff": 512,
        "ppg_medformer_single_channel": False,
        "ppg_medformer_no_inter_attn": False,
        "ppg_medformer_activation": "gelu",
    },
}

GLUCOSE_FOUNDATION_TUNING = {
    "csfm": {
        "ppg_csfm_signal_size": 300,
        "ppg_csfm_patch_size": 25,
        "ppg_csfm_hidden_dim": 128,
        "ppg_csfm_depth": 2,
        "ppg_csfm_heads": 4,
        "ppg_csfm_mlp_dim": 384,
        "ppg_csfm_dim_head": 32,
        "ppg_csfm_freeze_encoder": False,
    },
    "medformer": {
        "ppg_recent_hidden": 128,
        "ppg_recent_layers": 2,
        "ppg_recent_heads": 4,
        "ppg_recent_patch_sizes": "15,30,60",
    },
}

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run final paper baselines on glucose, oximetry, and OpenOx SO2/SpO2 datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "glucose",
            "oximetry",
            "openox_sao2",
            "openox_so2",
            "openox_spo2",
            "openox_sao2_spo2",
        ],
        default=["glucose", "oximetry"],
        help="Datasets to run.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=DEFAULT_BASELINES,
        default=DEFAULT_BASELINES,
        help="Baseline model families to run.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["base", "h"],
        default=["base", "h"],
        help="Run original model, +H model, or both.")
    parser.add_argument(
        "--papagei-mode",
        choices=["scratch", "pretrained"],
        default="scratch",
        help="Use PaPaGei from scratch or the local pretrained papagei_s.pt weight.")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-split", type=int, default=1)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=2026,
        help="Base seed for model initialization and fold-level training randomness.")
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic PyTorch/CUDA kernels for reproducible comparisons.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--modal-mask", default="10000")
    parser.add_argument("--ppg-mask", default="111111")
    parser.add_argument("--export-samples", type=int, default=16)
    parser.add_argument("--run-tag", default="", help="Optional tag added to each output detail string.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tune-oximetry-foundation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the tuned oximetry profile for CSFM and Medformer.")
    parser.add_argument(
        "--tune-glucose-foundation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the tuned glucose profile for CSFM and Medformer.")

    parser.add_argument("--load-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--spectral-excel", default=str(SRC_DIR / "Spectral Distribution Curves.xlsx"))
    parser.add_argument("--spectral-step", type=float, default=5.0)
    parser.add_argument(
        "--glucose-split",
        choices=["record", "patient", "patient_balanced", "patient_random"],
        default="record",
        help=(
            "Glucose split mode. patient/patient_balanced uses balanced subject holdout; "
            "patient_random keeps the old random subject holdout."))
    parser.add_argument("--glucose-test-size", type=float, default=0.2)
    parser.add_argument("--glucose-random-state", type=int, default=2)
    parser.add_argument("--glucose-patient-column", type=int, default=3)
    parser.add_argument("--glucose-subwindow-sec", type=float, default=0.0)
    parser.add_argument("--glucose-subwindow-stride-sec", type=float, default=1.0)
    parser.add_argument("--glucose-subwindow-sampling-rate", type=float, default=50.0)

    parser.add_argument(
        "--oximetry-root",
        default=str(REPO_ROOT / "data" / "phone_camera"))
    parser.add_argument(
        "--oximetry-split",
        choices=["random", "subject"],
        default="subject",
        help="Default is leave-one-subject-out for the phone-camera SpO2 experiment.")
    parser.add_argument("--oximetry-test-subject", default="100006")
    parser.add_argument("--oximetry-random-test-size", type=float, default=0.2)
    parser.add_argument("--oximetry-random-state", type=int, default=2)
    parser.add_argument("--oximetry-window-sec", type=float, default=10.0)
    parser.add_argument("--oximetry-stride-sec", type=float, default=10.0)
    parser.add_argument(
        "--oximetry-target",
        choices=["mean", "spo2_1", "spo2_2", "spo2_4", "spo2_5"],
        default="spo2_5")
    parser.add_argument(
        "--oximetry-normalize",
        choices=["window_minmax", "recording_minmax", "none"],
        default="window_minmax")
    parser.add_argument("--oximetry-target-min", type=float, default=70.0)
    parser.add_argument("--oximetry-target-max", type=float, default=100.0)
    parser.add_argument("--oximetry-rgb-sigma", type=float, default=35.0)
    parser.add_argument("--oximetry-spectral-min", type=float, default=430.0)
    parser.add_argument("--oximetry-spectral-max", type=float, default=680.0)

    parser.add_argument(
        "--openox-root",
        default=str(REPO_ROOT / "data" / "openox"))
    parser.add_argument("--openox-split", choices=["patient", "encounter", "random"], default="patient")
    parser.add_argument("--openox-test-size", type=float, default=0.2)
    parser.add_argument("--openox-random-state", type=int, default=2)
    parser.add_argument("--openox-window-sec", type=float, default=6.0)
    parser.add_argument("--openox-sampling-rate", type=float, default=50.0)
    parser.add_argument(
        "--openox-output-len",
        type=int,
        default=0,
        help="Use 0 to infer window_sec * sampling_rate.")
    parser.add_argument(
        "--openox-normalize",
        choices=["acdc", "window_minmax", "zscore", "none"],
        default="window_minmax")
    parser.add_argument(
        "--openox-detrend",
        action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--openox-detrend-cutoff", type=float, default=0.3)
    parser.add_argument(
        "--openox-scale",
        choices=["train_global_minmax", "train_global_zscore", "train_channel_zscore", "none"],
        default="none")
    parser.add_argument("--openox-target-min", type=float, default=None)
    parser.add_argument("--openox-target-max", type=float, default=None)
    parser.add_argument("--openox-spo2-source", choices=["continuous_2hz", "pulseoximeter"], default="continuous_2hz")
    parser.add_argument("--openox-spo2-aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--openox-spo2-max-device-range", type=float, default=10.0)
    parser.add_argument("--openox-spo2-stride-sec", type=float, default=30.0)
    parser.add_argument("--openox-spectral-sigma", type=float, default=25.0)
    parser.add_argument("--openox-spectral-min", type=float, default=600.0)
    parser.add_argument("--openox-spectral-max", type=float, default=1000.0)
    parser.add_argument("--openox-max-samples", type=int, default=0)

    parser.add_argument("--glucose-h-obs", type=float, default=GLUCOSE_H_UPDATES["lambda_obs"])
    parser.add_argument("--glucose-h-smooth", type=float, default=GLUCOSE_H_UPDATES["lambda_smooth"])
    parser.add_argument("--glucose-h-decorr", type=float, default=GLUCOSE_H_UPDATES["lambda_decorr"])
    parser.add_argument("--oximetry-h-obs", type=float, default=OXIMETRY_H_UPDATES["lambda_obs"])
    parser.add_argument("--oximetry-h-smooth", type=float, default=OXIMETRY_H_UPDATES["lambda_smooth"])
    parser.add_argument("--oximetry-h-decorr", type=float, default=OXIMETRY_H_UPDATES["lambda_decorr"])
    parser.add_argument(
        "--role-aware-h",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use role-aware H decomposition with synthetic baseline/noise auxiliary supervision for H variants.")
    parser.add_argument("--artifact-prob", type=float, default=0.5)
    parser.add_argument("--artifact-drift-scale", type=float, default=0.10)
    parser.add_argument("--artifact-noise-scale", type=float, default=0.03)
    parser.add_argument(
        "--context-adv-h",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For role-aware H, regularize background spectrum to absorb context and target spectrum to remove it.")
    parser.add_argument("--context-modalities", default="th,demo")
    parser.add_argument("--context-grl-lambda", type=float, default=1.0)
    parser.add_argument("--lambda-bg-context", type=float, default=0.003)
    parser.add_argument("--lambda-target-context-adv", type=float, default=0.003)
    return parser


def make_dataset_args(args, dataset_name):
    return SimpleNamespace(
        dataset=dataset_name,
        load_cache=args.load_cache,
        data_path=args.data_path,
        modal_mask=args.modal_mask,
        ppg_mask=args.ppg_mask,
        n_split=args.n_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_seed=getattr(args, "training_seed", 2026),
        deterministic_training=getattr(args, "deterministic_training", False),
        spectral_excel=args.spectral_excel,
        spectral_step=args.spectral_step,
        export_samples=args.export_samples,
        glucose_split=args.glucose_split,
        glucose_test_size=args.glucose_test_size,
        glucose_random_state=args.glucose_random_state,
        glucose_patient_column=args.glucose_patient_column,
        glucose_subwindow_sec=args.glucose_subwindow_sec,
        glucose_subwindow_stride_sec=args.glucose_subwindow_stride_sec,
        glucose_subwindow_sampling_rate=args.glucose_subwindow_sampling_rate,
        oximetry_root=args.oximetry_root,
        oximetry_split=args.oximetry_split,
        oximetry_test_subject=args.oximetry_test_subject,
        oximetry_random_test_size=args.oximetry_random_test_size,
        oximetry_random_state=args.oximetry_random_state,
        oximetry_window_sec=args.oximetry_window_sec,
        oximetry_stride_sec=args.oximetry_stride_sec,
        oximetry_target=args.oximetry_target,
        oximetry_normalize=args.oximetry_normalize,
        oximetry_target_min=args.oximetry_target_min,
        oximetry_target_max=args.oximetry_target_max,
        oximetry_rgb_sigma=args.oximetry_rgb_sigma,
        oximetry_spectral_min=args.oximetry_spectral_min,
        oximetry_spectral_max=args.oximetry_spectral_max,
        openox_root=args.openox_root,
        openox_split=args.openox_split,
        openox_test_size=args.openox_test_size,
        openox_random_state=args.openox_random_state,
        openox_window_sec=args.openox_window_sec,
        openox_sampling_rate=args.openox_sampling_rate,
        openox_output_len=args.openox_output_len,
        openox_normalize=args.openox_normalize,
        openox_detrend=args.openox_detrend,
        openox_detrend_cutoff=args.openox_detrend_cutoff,
        openox_scale=args.openox_scale,
        openox_target_min=args.openox_target_min,
        openox_target_max=args.openox_target_max,
        openox_spo2_source=args.openox_spo2_source,
        openox_spo2_aggregation=args.openox_spo2_aggregation,
        openox_spo2_max_device_range=args.openox_spo2_max_device_range,
        openox_spo2_stride_sec=args.openox_spo2_stride_sec,
        openox_spectral_sigma=args.openox_spectral_sigma,
        openox_spectral_min=args.openox_spectral_min,
        openox_spectral_max=args.openox_spectral_max,
        openox_max_samples=args.openox_max_samples,
    )


def selected_pairs(args):
    pairs = {}
    for name in args.baselines:
        if name == "papagei":
            pairs[name] = PAPAGEI_PAIRS[args.papagei_mode]
        else:
            pairs[name] = BASELINE_PAIRS[name]
    return pairs


def h_updates_for_dataset(args, dataset_name):
    if dataset_name == "oximetry" or dataset_name in OPENOX_DATASET_TARGETS:
        updates = OXIMETRY_H_UPDATES.copy()
        updates["lambda_obs"] = args.oximetry_h_obs
        updates["lambda_smooth"] = args.oximetry_h_smooth
        updates["lambda_decorr"] = args.oximetry_h_decorr
        if args.role_aware_h:
            updates.update(role_aware_h_updates(args, updates))
        return updates

    updates = GLUCOSE_H_UPDATES.copy()
    updates["lambda_obs"] = args.glucose_h_obs
    updates["lambda_smooth"] = args.glucose_h_smooth
    updates["lambda_decorr"] = args.glucose_h_decorr
    if args.role_aware_h:
        updates.update(role_aware_h_updates(args, updates))
    return updates


def role_aware_h_updates(args, current_updates):
    updates = {
        "ppg_backbone_name": "resnet",
        "enable_token_moe": True,
        "token_moe_mode": "role_aware",
        "role_moe_gate": True,
        "enable_artifact_pretext": True,
        "artifact_prob": args.artifact_prob,
        "artifact_drift_scale": args.artifact_drift_scale,
        "artifact_noise_scale": args.artifact_noise_scale,
        "lambda_obs": min(float(current_updates["lambda_obs"]), 0.3),
        "lambda_smooth": min(float(current_updates["lambda_smooth"]), 0.05),
        "lambda_decorr": min(float(current_updates["lambda_decorr"]), 0.001),
        "lambda_clean_recon": 0.03,
        "lambda_base_artifact": 0.01,
        "lambda_noise_artifact": 0.01,
        "lambda_background_smooth": 0.01,
        "lambda_target_sparse": 0.0001,
        "lambda_role_orth": 0.001,
    }
    if args.context_adv_h:
        updates.update({
            "enable_context_adversarial": True,
            "context_modalities": args.context_modalities,
            "context_grl_lambda": args.context_grl_lambda,
            "context_hidden_dim": 64,
            "lambda_bg_context": args.lambda_bg_context,
            "lambda_target_context_adv": args.lambda_target_context_adv,
        })
    return updates


def tuning_updates_for_dataset(args, dataset_name, baseline_name):
    if dataset_name == "glucose" and args.tune_glucose_foundation:
        return GLUCOSE_FOUNDATION_TUNING.get(baseline_name, {}).copy()
    if dataset_name == "oximetry" and args.tune_oximetry_foundation:
        return OXIMETRY_FOUNDATION_TUNING.get(baseline_name, {}).copy()
    return {}


def make_detail(dataset_name, baseline_name, variant, base_detail, run_tag):
    pieces = ["paper", dataset_name, baseline_name, variant]
    if run_tag:
        pieces.append(str(run_tag))
    pieces.append(base_detail)
    return "_".join(pieces)


def make_run_params(
        base_params,
        specs,
        dataset_name,
        baseline_name,
        variant,
        spec_key,
        args):
    params = base_params.copy()
    params.update(specs[spec_key]["updates"])
    tuning_updates = tuning_updates_for_dataset(args, dataset_name, baseline_name)
    if tuning_updates:
        params.update(tuning_updates)

    if variant == "h":
        params.update(h_updates_for_dataset(args, dataset_name))
        params["spectral_export_samples"] = args.export_samples
        params["spectral_export_batch_size"] = min(args.batch_size, max(args.export_samples, 1))
    else:
        params["model_name"] = 0
        params["enable_token_moe"] = False
        params["lambda_obs"] = 0.0
        params["lambda_smooth"] = 0.0
        params["lambda_decorr"] = 0.0
        params["spectral_export_samples"] = 0

    params["paper_baseline"] = baseline_name
    params["paper_variant"] = variant
    params["ablation_key"] = spec_key
    params["ablation_description"] = specs[spec_key]["description"]
    params["detail"] = make_detail(
        dataset_name,
        baseline_name,
        variant,
        base_params["detail"],
        args.run_tag)
    if tuning_updates:
        params["detail"] = f"{params['detail']}_tuned"
        params["paper_tuning_profile"] = f"{dataset_name}_foundation_tuned"
    else:
        params["paper_tuning_profile"] = ""
    return params


def print_plan(args):
    pairs = selected_pairs(args)
    specs = get_ablation_specs()
    for dataset_name in args.datasets:
        for baseline_name, (base_key, h_key) in pairs.items():
            keys = []
            if "base" in args.variants:
                keys.append(("base", base_key))
            if "h" in args.variants:
                keys.append(("h", h_key))
            for variant, key in keys:
                h_info = ""
                if variant == "h":
                    updates = h_updates_for_dataset(args, dataset_name)
                    h_info = (
                        f" | obs={updates['lambda_obs']} "
                        f"smooth={updates['lambda_smooth']} "
                        f"decorr={updates['lambda_decorr']} "
                        f"moe={updates.get('token_moe_mode', 'anonymous')} "
                        f"context_adv={updates.get('enable_context_adversarial', False)}"
                    )
                tuning_info = ""
                tuning_updates = tuning_updates_for_dataset(args, dataset_name, baseline_name)
                if tuning_updates:
                    tuning_fields = []
                    if "initial_lr" in tuning_updates:
                        tuning_fields.append(f"lr={tuning_updates['initial_lr']}")
                    if "dropout_rate" in tuning_updates:
                        tuning_fields.append(f"dropout={tuning_updates['dropout_rate']}")
                    if "weight_decay" in tuning_updates:
                        tuning_fields.append(f"wd={tuning_updates['weight_decay']}")
                    tuning_info = " | tuned " + (" ".join(tuning_fields) if tuning_fields else "architecture")
                print(
                    f"{dataset_name:8s} | {baseline_name:15s} | {variant:4s} | "
                    f"{key:4s} | {specs[key]['title']}{h_info}{tuning_info}")


def write_manifest_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "dataset",
            "baseline",
            "variant",
            "spec_key",
            "spec_title",
            "save_path",
            "epochs",
            "batch_size",
            "n_split",
            "lambda_obs",
            "lambda_smooth",
            "lambda_decorr",
            "token_moe_mode",
            "enable_artifact_pretext",
            "artifact_prob",
            "artifact_drift_scale",
            "artifact_noise_scale",
            "enable_context_adversarial",
            "context_modalities",
            "context_grl_lambda",
            "lambda_bg_context",
            "lambda_target_context_adv",
            "initial_lr",
            "dropout_rate",
            "weight_decay",
            "tuning_profile",
            "ppg_backbone_name",
            "papagei_mode",
            "glucose_split",
            "glucose_test_size",
            "glucose_random_state",
            "glucose_subwindow_sec",
            "glucose_subwindow_stride_sec",
            "glucose_subwindow_sampling_rate",
            "oximetry_split",
            "oximetry_test_subject",
            "oximetry_random_test_size",
            "oximetry_random_state",
            "openox_split",
            "openox_window_sec",
            "openox_sampling_rate",
            "openox_output_len",
            "openox_normalize",
            "openox_detrend",
            "openox_detrend_cutoff",
            "openox_scale",
            "openox_spo2_source",
            "openox_spo2_aggregation",
            "openox_spo2_max_device_range",
            "openox_spo2_stride_sec",
        ])


def append_manifest(path, args, dataset_name, baseline_name, variant, spec_key, spec_title, params, save_path):
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name,
            baseline_name,
            variant,
            spec_key,
            spec_title,
            save_path,
            args.epochs,
            args.batch_size,
            args.n_split,
            params.get("lambda_obs", ""),
            params.get("lambda_smooth", ""),
            params.get("lambda_decorr", ""),
            params.get("token_moe_mode", ""),
            params.get("enable_artifact_pretext", ""),
            params.get("artifact_prob", ""),
            params.get("artifact_drift_scale", ""),
            params.get("artifact_noise_scale", ""),
            params.get("enable_context_adversarial", ""),
            params.get("context_modalities", ""),
            params.get("context_grl_lambda", ""),
            params.get("lambda_bg_context", ""),
            params.get("lambda_target_context_adv", ""),
            params.get("initial_lr", ""),
            params.get("dropout_rate", ""),
            params.get("weight_decay", ""),
            params.get("paper_tuning_profile", ""),
            params.get("ppg_backbone_name", "resnet"),
            args.papagei_mode,
            args.glucose_split,
            args.glucose_test_size,
            args.glucose_random_state,
            args.glucose_subwindow_sec,
            args.glucose_subwindow_stride_sec,
            args.glucose_subwindow_sampling_rate,
            args.oximetry_split,
            args.oximetry_test_subject,
            args.oximetry_random_test_size,
            args.oximetry_random_state,
            args.openox_split,
            args.openox_window_sec,
            args.openox_sampling_rate,
            args.openox_output_len,
            args.openox_normalize,
            args.openox_detrend,
            args.openox_detrend_cutoff,
            args.openox_scale,
            args.openox_spo2_source,
            args.openox_spo2_aggregation,
            args.openox_spo2_max_device_range,
            args.openox_spo2_stride_sec,
        ])


def run(args):
    if args.gpu == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dry_run:
        print_plan(args)
        return

    specs = get_ablation_specs()
    pairs = selected_pairs(args)
    manifest_path = REPO_ROOT / "outputs" / f"paper_baseline_manifest_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    write_manifest_header(manifest_path)

    for dataset_name in args.datasets:
        dataset_args = make_dataset_args(args, dataset_name)
        provider, dataset_dic = build_dataset(dataset_args)
        base_params = build_base_params(provider, dataset_args)

        for baseline_name, (base_key, h_key) in pairs.items():
            run_items = []
            if "base" in args.variants:
                run_items.append(("base", base_key))
            if "h" in args.variants:
                run_items.append(("h", h_key))

            for variant, spec_key in run_items:
                spec = specs[spec_key]
                params = make_run_params(
                    base_params,
                    specs,
                    dataset_name,
                    baseline_name,
                    variant,
                    spec_key,
                    args)
                print(
                    f"\n===== {dataset_name} | {baseline_name} | {variant} "
                    f"({spec_key}: {spec['title']}) =====")
                print(spec["description"])

                trainer = Training_Process(params, dataset_dic)
                trainer.train_start()
                append_manifest(
                    manifest_path,
                    args,
                    dataset_name,
                    baseline_name,
                    variant,
                    spec_key,
                    spec["title"],
                    params,
                    trainer.save_path)

    print(f"\nPaper baseline manifest saved to: {manifest_path}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
