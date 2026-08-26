# python src\run_ablation.py --experiments A B C --modal-mask 10000 --gpu 0
# python src\run_ablation.py --experiments D E --modal-mask 10000 --gpu 1
# python src\run_ablation.py --experiments R S --modal-mask 10000 --epochs 500 --gpu 0

import argparse
import csv
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DEFAULT_DATA_PATH = str(REPO_ROOT / "data" / "glucose")
OPENOX_DATASET_TARGETS = {
    "openox_sao2": "so2",
    "openox_so2": "so2",
    "openox_spo2": "spo2",
    "openox_sao2_spo2": "sao2_spo2",
}
OPENOX_DRY_TARGETS = {
    "openox_sao2": ("SO2", "%"),
    "openox_so2": ("SO2", "%"),
    "openox_spo2": ("SpO2", "%"),
    "openox_sao2_spo2": ("SaO2_SpO2", "%"),
}
EXTERNAL_SINGLE_TARGET_DATASETS = {
    "oximetry",
    "openox_sao2",
    "openox_so2",
    "openox_spo2",
}

ARCHIVED_EXPERIMENT_GROUPS = {
    "archived_ah": ["A", "H"],
    "archive_ah": ["A", "H"],
    "locked_ah": ["A", "H"],
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for the physics-guided spectral model.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G", "H"],
        help="Experiments to run: individual keys/aliases, the existing H4A0 groups, "
             "or external_h4_resnet/external_h4_selected/external_h4_all for "
             "independently trained single-target oxygen tasks.")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs for every ablation.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--n-split", type=int, default=1, help="Number of ShuffleSplit folds. Use 5 for final reporting.")
    parser.add_argument(
        "--learning-rate",
        "--initial-lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="Override the base initial learning rate (default: 0.001).")
    parser.add_argument(
        "--training-seed",
        type=int,
        default=2026,
        help="Base seed for model initialization, dropout, and DataLoader shuffling. Fold i uses seed+i.")
    parser.add_argument(
        "--deterministic-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic PyTorch/CUDA kernels for reproducible comparisons.")
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable cuDNN algorithm benchmarking. Disabled by default for CUDA stability.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value. Use -1 for CPU-visible run.")
    parser.add_argument(
        "--modal-mask",
        default="11100",
        help="Five-bit modal mask in order PPG, TH, Demo, DF, MI. Example: 10000 for PPG-only.")
    parser.add_argument(
        "--ppg-mask",
        default="111111",
        help="Six-bit PPG channel mask. Example: 111111 uses all PPG channels.")
    parser.add_argument(
        "--load-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load cached arrays from --data-path. Use --no-load-cache to rebuild them from raw files.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Raw data directory if --no-load-cache is used.")
    parser.add_argument(
        "--dataset",
        choices=[
            "glucose",
            "oximetry",
            "openox_sao2",
            "openox_so2",
            "openox_spo2",
            "openox_sao2_spo2",
        ],
        default="glucose",
        help="Dataset to run. openox_so2/openox_sao2 use blood-gas SaO2; "
             "openox_spo2 uses continuous monitor SpO2; openox_sao2_spo2 predicts "
             "synchronized SaO2 and SpO2 from the same red/IR PPG window.")
    parser.add_argument(
        "--glucose-split",
        choices=["record", "patient", "patient_balanced", "patient_random"],
        default="record",
        help=(
            "Glucose split mode. record keeps the original random row split; "
            "patient uses balanced subject holdout; patient_random keeps the old random subject holdout."))
    parser.add_argument("--glucose-test-size", type=float, default=0.2, help="Glucose test fraction.")
    parser.add_argument("--glucose-random-state", type=int, default=2, help="Glucose split random seed.")
    parser.add_argument("--glucose-patient-column", type=int, default=3, help="Column index in data_raw.npy used as glucose subject ID.")
    parser.add_argument(
        "--glucose-subwindow-sec",
        type=float,
        default=0.0,
        help="Cut each glucose PPG record into overlapping subwindows after train/test split. 0 disables.")
    parser.add_argument(
        "--glucose-subwindow-stride-sec",
        type=float,
        default=1.0,
        help="Stride in seconds for glucose subwindowing.")
    parser.add_argument(
        "--glucose-subwindow-sampling-rate",
        type=float,
        default=50.0,
        help="Sampling rate used to convert glucose subwindow seconds to points.")
    parser.add_argument(
        "--glucose-aux-spo2",
        dest="glucose_aux_spo2",
        action="store_true",
        help="Use the blood oxygen saturation field distributed with the private glucose dataset as an auxiliary target.")
    parser.add_argument(
        "--glucose-keep-spo2-input",
        dest="glucose_keep_spo2_input",
        action="store_true",
        help="Keep the SpO2 field in modal_3. By default it is zeroed to avoid target leakage.")
    parser.add_argument(
        "--oximetry-root",
        default=str(REPO_ROOT / "data" / "phone_camera"),
        help="Root directory of ubicomplab/oximetry-phone-cam-data.")
    parser.add_argument(
        "--oximetry-test-subject",
        default="100006",
        help="Held-out oximetry subject. Valid: 100001-100006 or last.")
    parser.add_argument(
        "--oximetry-split",
        choices=["subject", "random"],
        default="subject",
        help="Oximetry split mode. subject is leave-one-subject-out; random mixes windows from all subjects.")
    parser.add_argument("--oximetry-random-test-size", type=float, default=0.2, help="Test fraction for random oximetry split.")
    parser.add_argument("--oximetry-random-state", type=int, default=2, help="Random seed for random oximetry split.")
    parser.add_argument("--oximetry-window-sec", type=float, default=10.0, help="SpO2 window length in seconds.")
    parser.add_argument("--oximetry-stride-sec", type=float, default=10.0, help="SpO2 window stride in seconds; default matches the 10 s non-overlapping window.")
    parser.add_argument(
        "--oximetry-target",
        choices=["mean", "spo2_1", "spo2_2", "spo2_4", "spo2_5"],
        default="spo2_5",
        help="SpO2 label source; spo2_5 is the paper's Masimo Radical-7 reference.")
    parser.add_argument(
        "--oximetry-normalize",
        default="window_minmax",
        choices=["window_minmax", "recording_minmax", "none"],
        help="RGB PPG normalization for the oximetry dataset.")
    parser.add_argument("--oximetry-target-min", type=float, default=70.0, help="Lower bound used to normalize SpO2; the source paper excludes values below 70%.")
    parser.add_argument("--oximetry-target-max", type=float, default=100.0, help="Upper bound used to normalize SpO2.")
    parser.add_argument("--oximetry-rgb-sigma", type=float, default=35.0, help="Gaussian sigma for approximate RGB H.")
    parser.add_argument("--oximetry-spectral-min", type=float, default=430.0, help="Minimum wavelength for approximate RGB H.")
    parser.add_argument("--oximetry-spectral-max", type=float, default=680.0, help="Maximum wavelength for approximate RGB H.")
    parser.add_argument(
        "--openox-root",
        default=str(REPO_ROOT / "data" / "openox"),
        help="Root directory of PhysioNet openox-repo. Either the repo root or 1.1.1 directory is accepted.")
    parser.add_argument(
        "--openox-split",
        choices=["patient", "encounter", "random"],
        default="patient",
        help="OpenOx split mode. patient is recommended for reporting.")
    parser.add_argument("--openox-test-size", type=float, default=0.2, help="OpenOx test fraction.")
    parser.add_argument("--openox-random-state", type=int, default=2, help="OpenOx random seed.")
    parser.add_argument(
        "--openox-window-sec",
        type=float,
        default=6.0,
        help="OpenOx PPG window centered on the label time. Default is 6 s.")
    parser.add_argument(
        "--openox-sampling-rate",
        type=float,
        default=50.0,
        help="Effective OpenOx sampling rate after resampling. Default 50 Hz.")
    parser.add_argument(
        "--openox-output-len",
        type=int,
        default=0,
        help="OpenOx resampled PPG length. Use 0 to infer window_sec * sampling_rate.")
    parser.add_argument(
        "--openox-normalize",
        choices=["acdc", "window_minmax", "zscore", "none"],
        default="window_minmax",
        help="OpenOx Red/IR PPG preprocessing.")
    parser.add_argument(
        "--openox-detrend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove OpenOx Red/IR baseline drift with a high-pass filter before window normalization.")
    parser.add_argument(
        "--openox-detrend-cutoff",
        type=float,
        default=0.3,
        help="High-pass cutoff in Hz for OpenOx baseline drift removal.")
    parser.add_argument(
        "--openox-scale",
        choices=["train_global_minmax", "train_global_zscore", "train_channel_zscore", "none"],
        default="none",
        help="OpenOx post-preprocessing scaling. Default keeps the window-level preprocessing output unchanged.")
    parser.add_argument("--openox-target-min", type=float, default=None, help="Lower bound used to normalize OpenOx targets. Default is task-specific.")
    parser.add_argument("--openox-target-max", type=float, default=None, help="Upper bound used to normalize OpenOx targets. Default is task-specific.")
    parser.add_argument(
        "--openox-spo2-source",
        choices=["continuous_2hz", "pulseoximeter"],
        default="continuous_2hz",
        help="SpO2 label source. continuous_2hz uses time-aligned *_2hz.csv readings; pulseoximeter keeps sample-level labels.")
    parser.add_argument(
        "--openox-spo2-aggregation",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate cleaned OpenOx pulse oximeter device readings for SpO2 labels.")
    parser.add_argument(
        "--openox-spo2-max-device-range",
        type=float,
        default=10.0,
        help="Drop SpO2 groups whose cleaned device readings differ by more than this percentage points. Use <=0 to disable.")
    parser.add_argument(
        "--openox-spo2-stride-sec",
        type=float,
        default=30.0,
        help="Temporal stride for continuous_2hz SpO2 labels. Use 0 or negative to keep every 2 Hz row.")
    parser.add_argument("--openox-spectral-sigma", type=float, default=25.0, help="Gaussian sigma for OpenOx red/IR H.")
    parser.add_argument("--openox-spectral-min", type=float, default=600.0, help="Minimum wavelength for OpenOx red/IR H.")
    parser.add_argument("--openox-spectral-max", type=float, default=1000.0, help="Maximum wavelength for OpenOx red/IR H.")
    parser.add_argument("--openox-max-samples", type=int, default=0, help="Optional cap for quick OpenOx smoke tests.")
    parser.add_argument(
        "--spectral-excel",
        default=str(SRC_DIR / "Spectral Distribution Curves.xlsx"),
        help="Excel file used to build the observation matrix H.")
    parser.add_argument("--spectral-step", type=float, default=5.0, help="Wavelength resampling step for H.")
    parser.add_argument(
        "--export-samples",
        type=int,
        default=16,
        help="Number of test samples exported for spectral visualization in physics-model ablations.")
    parser.add_argument("--dry-run", action="store_true", help="Print ablation configs and exit.")
    return parser


def get_ablation_specs():
    specs = OrderedDict()
    h_low_loss_updates = {
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

    # Archived A/H reference pair. Keep A fixed as H's no-plugin counterpart.
    specs["A"] = {
        "alias": "baseline",
        "title": "A_baseline_multimodal",
        "description": "原始多模态回归基线，不使用光谱升维分支。",
        "updates": {
            "model_name": 0,
            "enable_token_moe": False,
            "enable_target_specific_regression": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["B"] = {
        "alias": "lifting",
        "title": "B_spectral_lifting",
        "description": "加入光谱升维生成分支，但不使用观测一致性和平滑约束。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
        },
    }

    specs["C"] = {
        "alias": "obs",
        "title": "C_lifting_obs",
        "description": "在升维分支上加入 H 回投影观测一致性约束。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": False,
            "lambda_obs": 0.1,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
        },
    }

    specs["D"] = {
        "alias": "smooth",
        "title": "D_lifting_obs_smooth",
        "description": "在观测一致性基础上加入波长维平滑约束。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": False,
            "lambda_obs": 0.1,
            "lambda_smooth": 0.01,
            "lambda_decorr": 0.0,
        },
    }

    specs["E"] = {
        "alias": "full",
        "title": "E_full_moe_decorr",
        "description": "完整模型：光谱升维、观测一致性、平滑、MoE token 重组和去相关。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.1,
            "lambda_smooth": 0.01,
            "lambda_decorr": 0.001,
        },
    }

    specs["F"] = {
        "alias": "aux_only",
        "title": "F_physics_aux_only",
        "description": "保留光谱生成与物理损失，但不把 spectral_feature 输入血糖回归头。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": False,
            "spectral_fusion_mode": "none",
            "lambda_obs": 0.1,
            "lambda_smooth": 0.01,
            "lambda_decorr": 0.0,
        },
    }

    specs["G"] = {
        "alias": "gated",
        "title": "G_full_gated_spectral",
        "description": "完整模型，但 spectral_feature 通过初始接近 0 的可学习门控进入血糖回归头。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "gate",
            "spectral_gate_init": -4.0,
            "lambda_obs": 0.1,
            "lambda_smooth": 0.01,
            "lambda_decorr": 0.001,
        },
    }

    # Archived A/H reference pair. Keep this low-loss H setting fixed.
    specs["H"] = {
        "alias": "low_weight",
        "title": "H_full_low_loss_weight",
        "description": "完整模型，降低观测一致性和平滑损失权重，去相关损失权重设为 0.001。",
        "updates": h_low_loss_updates.copy(),
    }

    role_h_updates = h_low_loss_updates.copy()
    role_h_updates.update({
        "ppg_backbone_name": "resnet",
        "enable_token_moe": True,
        "token_moe_mode": "role_aware",
        "role_moe_gate": True,
        "enable_artifact_pretext": True,
        "artifact_prob": 0.5,
        "artifact_drift_scale": 0.10,
        "artifact_noise_scale": 0.03,
        "lambda_obs": 0.3,
        "lambda_smooth": 0.05,
        "lambda_decorr": 0.001,
        "lambda_clean_recon": 0.03,
        "lambda_base_artifact": 0.01,
        "lambda_noise_artifact": 0.01,
        "baseline_lowpass_kernel": 101,
        "noise_highpass_kernel": 31,
        "enable_background_context_conditioning": True,
        "background_context_hidden_dim": 64,
        "context_modalities": "th,demo,df",
        "lambda_background_smooth": 0.01,
        "lambda_target_smooth": 0.005,
        "lambda_basis_smooth": 0.005,
        "lambda_target_sparse": 0.0001,
        "lambda_role_orth": 0.001,
        "lambda_glucose_component": 0.05,
        "lambda_spo2_component": 0.05,
    })
    specs["CM"] = {
        "alias": "role_h",
        "title": "CM_RoleAware_H_ResNet",
        "description": "Role-aware H with ResNet backbone: decomposes PPG into baseline drift, high-frequency artifact, background spectrum, and target-related spectrum.",
        "updates": role_h_updates,
    }

    context_adv_h_updates = role_h_updates.copy()
    context_adv_h_updates.update({
        "enable_context_adversarial": True,
        "context_modalities": "th,demo,df",
        "context_grl_lambda": 1.0,
        "context_hidden_dim": 64,
        "lambda_bg_context": 0.003,
        "lambda_target_context_adv": 0.003,
    })
    specs["CN"] = {
        "alias": "context_adv_h",
        "title": "CN_ContextAdv_H_ResNet",
        "description": "Role-aware H with context adversarial regularization: background absorbs demographic/T&H context while target spectrum is discouraged from encoding it.",
        "updates": context_adv_h_updates,
    }

    lite_h_updates = role_h_updates.copy()
    lite_h_updates.update({
        "enable_target_specific_regression": True,
        "target_specific_regression_mode": "baseline_residual_plugin",
        "target_spectral_gate_init": -3.0,
        "glucose_spectral_gate_max": 0.30,
        "spo2_spectral_gate_max": 0.15,
        "role_moe_gate_mode": "sigmoid",
        "role_moe_sigmoid_gate_init": 0.2,
        "early_stopping_monitor": "bg",
        "lambda_glucose_component": 0.05,
        "lambda_spo2_component": 0.05,
        "enable_artifact_pretext": False,
        "artifact_prob": 0.0,
        "lambda_clean_recon": 0.0,
        "lambda_base_artifact": 0.0,
        "lambda_noise_artifact": 0.0,
        "enable_context_adversarial": False,
        "lambda_bg_context": 0.0,
        "lambda_target_context_adv": 0.0,
    })
    specs["CO"] = {
        "alias": "lite_h",
        "title": "CO_Lite_H_ResNet",
        "description": "Role-aware spectral H without artifact pretext or context-adversarial regularization.",
        "updates": lite_h_updates,
    }

    h_ablation_base_updates = {
        "model_name": "physics_spectral",
        "ppg_backbone_name": "resnet",
        "filter_structure": [64, 96, 128, 160],
        "dense_structure": [256, 128, 64, 1],
        "dropout_rate": 0.30,
        "weight_decay": 0.00001,
        "spectral_hidden": 64,
    }

    # Match CK/CL exactly when tuning the current Lite-H module. Keep the
    # archived HAB0-HAB7 ContextAdv ablations on their original ResNet setup.
    lstm_attention_h_ablation_base_updates = {
        "model_name": "physics_spectral",
        "ppg_backbone_name": "lstm_attention",
        "filter_structure": [64, 96, 128, 160],
        "dense_structure": [256, 128, 64, 1],
        "dropout_rate": 0.25,
        "weight_decay": 0.0,
        "spectral_hidden": 64,
    }

    def make_h_ablation_updates(extra_updates=None):
        updates = h_ablation_base_updates.copy()
        updates.update(context_adv_h_updates)
        updates.update(h_ablation_base_updates)
        if extra_updates:
            updates.update(extra_updates)
        return updates

    h_ablation_specs = [
        (
            "HAB0",
            "h_ablation_full",
            "HAB0_ContextAdv_H_full",
            "Full current ContextAdv_H on the ResNet1D backbone; matched to the official S setting.",
            {},
        ),
        (
            "HAB1",
            "h_ablation_basic_h",
            "HAB1_basic_H_no_roles",
            "Basic spectral H without role-aware decomposition, artifact pretext, context adversarial regularization, or component supervision.",
            {
                "token_moe_mode": "anonymous",
                "role_moe_gate": False,
                "enable_artifact_pretext": False,
                "baseline_lowpass_kernel": 0,
                "noise_highpass_kernel": 0,
                "lambda_clean_recon": 0.0,
                "lambda_base_artifact": 0.0,
                "lambda_noise_artifact": 0.0,
                "lambda_background_smooth": 0.0,
                "lambda_target_sparse": 0.0,
                "lambda_role_orth": 0.0,
                "enable_context_adversarial": False,
                "lambda_bg_context": 0.0,
                "lambda_target_context_adv": 0.0,
                "enable_aux_spo2_spectrum": False,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
        (
            "HAB2",
            "h_ablation_no_obs",
            "HAB2_no_observation_loss",
            "Full ContextAdv_H but removes the observation reconstruction loss y_hat = H S.",
            {"lambda_obs": 0.0},
        ),
        (
            "HAB3",
            "h_ablation_no_smooth",
            "HAB3_no_spectral_smoothness",
            "Full ContextAdv_H but removes spectral smoothness regularization.",
            {
                "lambda_smooth": 0.0,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_basis_smooth": 0.0,
            },
        ),
        (
            "HAB4",
            "h_ablation_no_decorr",
            "HAB4_no_decorr_or_role_orth",
            "Full ContextAdv_H but removes spectral decorrelation and role orthogonality losses.",
            {"lambda_decorr": 0.0, "lambda_role_orth": 0.0},
        ),
        (
            "HAB5",
            "h_ablation_no_artifact",
            "HAB5_no_artifact_pretext",
            "Full ContextAdv_H but disables synthetic drift/noise pretext supervision.",
            {
                "enable_artifact_pretext": False,
                "lambda_clean_recon": 0.0,
                "lambda_base_artifact": 0.0,
                "lambda_noise_artifact": 0.0,
            },
        ),
        (
            "HAB6",
            "h_ablation_no_context",
            "HAB6_no_context_adversarial",
            "Full ContextAdv_H but removes context background absorption and target adversarial regularization.",
            {
                "enable_context_adversarial": False,
                "lambda_bg_context": 0.0,
                "lambda_target_context_adv": 0.0,
            },
        ),
        (
            "HAB7",
            "h_ablation_no_dual_target_components",
            "HAB7_no_dual_target_spectral_components",
            "Full ContextAdv_H but removes the auxiliary SpO2 spectral branch and component prediction losses.",
            {
                "enable_aux_spo2_spectrum": False,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
    ]
    for key, alias, title, description, extra_updates in h_ablation_specs:
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": make_h_ablation_updates(extra_updates),
        }

    def make_lite_h_ablation_updates(extra_updates=None):
        updates = lite_h_updates.copy()
        updates.update(lstm_attention_h_ablation_base_updates)
        if extra_updates:
            updates.update(extra_updates)
        return updates

    lite_h_ablation_specs = [
        (
            "HL0",
            "lite_h_ablation_full",
            "HL0_LSTM_Attention_Lite_H_full",
            "Full Lite-H on the LSTM-Attention backbone, matched to CK/CL for H-module tuning.",
            {},
        ),
        (
            "HL1",
            "lite_h_ablation_no_obs",
            "HL1_LSTM_Attention_Lite_H_no_observation_loss",
            "LSTM-Attention Lite-H without the observation reconstruction loss y_hat = H S.",
            {"lambda_obs": 0.0},
        ),
        (
            "HL2",
            "lite_h_ablation_no_smooth",
            "HL2_LSTM_Attention_Lite_H_no_spectral_smoothness",
            "LSTM-Attention Lite-H without spectral smoothness regularization.",
            {
                "lambda_smooth": 0.0,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_basis_smooth": 0.0,
            },
        ),
        (
            "HL3",
            "lite_h_ablation_no_decorr",
            "HL3_LSTM_Attention_Lite_H_no_decorr_or_role_orth",
            "LSTM-Attention Lite-H without spectral decorrelation or role orthogonality losses.",
            {"lambda_decorr": 0.0, "lambda_role_orth": 0.0},
        ),
        (
            "HL4",
            "lite_h_ablation_no_target_sparse",
            "HL4_LSTM_Attention_Lite_H_no_target_sparsity",
            "LSTM-Attention Lite-H without target-spectrum sparsity regularization.",
            {"lambda_target_sparse": 0.0},
        ),
        (
            "HL5",
            "lite_h_ablation_no_component_loss",
            "HL5_LSTM_Attention_Lite_H_no_component_prediction_loss",
            "LSTM-Attention Lite-H keeps component spectra but removes glucose and SpO2 component prediction losses.",
            {"lambda_glucose_component": 0.0, "lambda_spo2_component": 0.0},
        ),
        (
            "HL6",
            "lite_h_ablation_no_spo2_branch",
            "HL6_LSTM_Attention_Lite_H_no_aux_spo2_spectrum",
            "LSTM-Attention Lite-H without the auxiliary SpO2 spectral branch.",
            {"enable_aux_spo2_spectrum": False, "lambda_spo2_component": 0.0},
        ),
    ]
    for key, alias, title, description, extra_updates in lite_h_ablation_specs:
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": make_lite_h_ablation_updates(extra_updates),
        }

    lite_h_combination_specs = [
        (
            "HC1",
            "lite_h_combo_no_role_orth",
            "HC1_LSTM_Attention_Lite_H_no_role_orth",
            "Full LSTM-Attention Lite-H with only role orthogonality disabled.",
            {"lambda_role_orth": 0.0},
        ),
        (
            "HC2",
            "lite_h_combo_no_token_decorr",
            "HC2_LSTM_Attention_Lite_H_no_token_decorr",
            "Full LSTM-Attention Lite-H with only token decorrelation disabled.",
            {"lambda_decorr": 0.0},
        ),
        (
            "HC3",
            "lite_h_combo_no_role_orth_no_sparse",
            "HC3_LSTM_Attention_Lite_H_no_role_orth_no_target_sparsity",
            "LSTM-Attention Lite-H without role orthogonality or target-spectrum sparsity.",
            {"lambda_role_orth": 0.0, "lambda_target_sparse": 0.0},
        ),
        (
            "HC4",
            "lite_h_combo_no_role_orth_no_component",
            "HC4_LSTM_Attention_Lite_H_no_role_orth_no_component_loss",
            "LSTM-Attention Lite-H without role orthogonality or component prediction losses.",
            {
                "lambda_role_orth": 0.0,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
        (
            "HC5",
            "lite_h_combo_minimal_regularization",
            "HC5_LSTM_Attention_Lite_H_no_role_orth_no_sparse_no_component",
            "LSTM-Attention Lite-H without role orthogonality, target sparsity, or component prediction losses.",
            {
                "lambda_role_orth": 0.0,
                "lambda_target_sparse": 0.0,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
    ]
    for key, alias, title, description, extra_updates in lite_h_combination_specs:
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": make_lite_h_ablation_updates(extra_updates),
        }

    # Paper-facing ablations use HL4 as the full model. Every variant keeps
    # target sparsity disabled so that the retired L1 shrinkage term cannot
    # confound attribution to the remaining modules.
    hl4_full_updates = make_lite_h_ablation_updates({
        "lambda_target_sparse": 0.0,
    })

    def make_hl4_ablation_updates(extra_updates=None):
        updates = hl4_full_updates.copy()
        if extra_updates:
            updates.update(extra_updates)
        return updates

    hl4_ablation_specs = [
        (
            "H4A0",
            "hl4_ablation_full",
            "H4A0_LSTM_Attention_HL4_full",
            "Full HL4 model with role-aware decomposition and no target-spectrum sparsity.",
            {},
        ),
        (
            "H4A1",
            "hl4_ablation_no_obs",
            "H4A1_HL4_no_observation_loss",
            "HL4 without fixed-H observation reconstruction supervision.",
            {"lambda_obs": 0.0},
        ),
        (
            "H4A2",
            "hl4_ablation_no_smooth",
            "H4A2_HL4_no_spectral_smoothness",
            "HL4 without global, background, target, or basis smoothness losses.",
            {
                "lambda_smooth": 0.0,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_basis_smooth": 0.0,
            },
        ),
        (
            "H4A3",
            "hl4_ablation_no_role_gate",
            "H4A3_HL4_no_role_gate",
            "HL4 keeps role-specific branches but removes sample-adaptive role gating.",
            {"role_moe_gate": False},
        ),
        (
            "H4A4",
            "hl4_ablation_no_role_orth",
            "H4A4_HL4_no_role_orthogonality",
            "HL4 keeps role-specific branches and gates but removes role orthogonality loss.",
            {"lambda_role_orth": 0.0},
        ),
        (
            "H4A5",
            "hl4_ablation_no_background_context",
            "H4A5_HL4_no_background_context_conditioning",
            "HL4 without demographic and environmental FiLM conditioning of the background spectrum.",
            {"enable_background_context_conditioning": False},
        ),
        (
            "H4A6",
            "hl4_ablation_no_component_supervision",
            "H4A6_HL4_no_component_prediction_losses",
            "HL4 keeps glucose and SpO2 spectra but removes their direct component prediction losses.",
            {"lambda_glucose_component": 0.0, "lambda_spo2_component": 0.0},
        ),
        (
            "H4A7",
            "hl4_ablation_no_spo2_spectral_branch",
            "H4A7_HL4_no_aux_spo2_spectral_branch",
            "HL4 without the auxiliary SpO2 spectral component and target-specific residual heads.",
            {
                "enable_aux_spo2_spectrum": False,
                "enable_target_specific_regression": False,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
        (
            "H4A8",
            "hl4_ablation_no_smooth_bases",
            "H4A8_HL4_pointwise_spectral_generator",
            "HL4 replaces learnable smooth spectral bases with pointwise wavelength generation.",
            {
                "spectral_generator_mode": "pointwise",
                "lambda_basis_smooth": 0.0,
            },
        ),
        (
            "H4A9",
            "hl4_ablation_no_token_decorr",
            "H4A9_HL4_no_token_decorrelation",
            "HL4 without token-level decorrelation while retaining role-aware decomposition.",
            {"lambda_decorr": 0.0},
        ),
        (
            "H4A10",
            "hl4_ablation_no_role_decomposition",
            "H4A10_HL4_generic_spectral_moe_no_roles",
            "HL4 replaces role-aware decomposition with an anonymous spectral MoE and a generic spectrum.",
            {
                "token_moe_mode": "anonymous",
                "role_moe_gate": False,
                "enable_aux_spo2_spectrum": False,
                "enable_target_specific_regression": False,
                "enable_background_context_conditioning": False,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_role_orth": 0.0,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
    ]
    for key, alias, title, description, extra_updates in hl4_ablation_specs:
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": make_hl4_ablation_updates(extra_updates),
        }

    oximetry_h_005_updates = h_low_loss_updates.copy()
    oximetry_h_005_updates.update({
        "lambda_obs": 0.05,
        "lambda_smooth": 0.01,
        "lambda_decorr": 0.001,
    })
    specs["OX05"] = {
        "alias": "oximetry_h_obs005",
        "title": "OX05_oximetry_H_obs005",
        "description": "仅用于 oximetry 数据集：RGB 近似 H 的轻观测一致性权重 lambda_obs=0.05。",
        "updates": oximetry_h_005_updates,
    }

    oximetry_h_010_updates = h_low_loss_updates.copy()
    oximetry_h_010_updates.update({
        "lambda_obs": 0.1,
        "lambda_smooth": 0.01,
        "lambda_decorr": 0.001,
    })
    specs["OX10"] = {
        "alias": "oximetry_h_obs010",
        "title": "OX10_oximetry_H_obs010",
        "description": "仅用于 oximetry 数据集：RGB 近似 H 的轻观测一致性权重 lambda_obs=0.1。",
        "updates": oximetry_h_010_updates,
    }

    specs["I"] = {
        "alias": "h_obs_0p01",
        "title": "I_H_obs_0p01",
        "description": "H 结构局部搜索：降低观测一致性权重 lambda_obs=0.01。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.01,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["J"] = {
        "alias": "h_obs_0p05",
        "title": "J_H_obs_0p05",
        "description": "H 结构局部搜索：提高观测一致性权重 lambda_obs=0.05。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.05,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["K"] = {
        "alias": "h_smooth_0p0005",
        "title": "K_H_smooth_0p0005",
        "description": "H 结构局部搜索：降低光谱平滑权重 lambda_smooth=0.0005。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.0005,
            "lambda_decorr": 0.0001,
        },
    }

    specs["L"] = {
        "alias": "h_smooth_0p003",
        "title": "L_H_smooth_0p003",
        "description": "H 结构局部搜索：提高光谱平滑权重 lambda_smooth=0.003。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.003,
            "lambda_decorr": 0.0001,
        },
    }

    specs["M"] = {
        "alias": "h_decorr_0p00005",
        "title": "M_H_decorr_0p00005",
        "description": "H 结构局部搜索：降低 token 去相关权重 lambda_decorr=0.00005。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.00005,
        },
    }

    specs["N"] = {
        "alias": "h_decorr_0p0003",
        "title": "N_H_decorr_0p0003",
        "description": "H 结构局部搜索：提高 token 去相关权重 lambda_decorr=0.0003。",
        "updates": {
            "model_name": "physics_spectral",
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0003,
        },
    }

    specs["O"] = {
        "alias": "ppg_small_baseline",
        "title": "O_PPG_small_regularized_baseline",
        "description": "PPG-only 倾向的小模型正则化基线：降低主干/全连接容量并加入 weight decay。",
        "updates": {
            "model_name": 0,
            "filter_structure": [32, 48, 64, 96],
            "dense_structure": [128, 64, 32, 1],
            "dropout_rate": 0.35,
            "weight_decay": 0.0001,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["P"] = {
        "alias": "ppg_small_h",
        "title": "P_PPG_small_regularized_H",
        "description": "PPG-only 倾向的小模型正则化 H：弱物理约束 + 较小主干和光谱 hidden。",
        "updates": {
            "model_name": "physics_spectral",
            "filter_structure": [32, 48, 64, 96],
            "dense_structure": [128, 64, 32, 1],
            "dropout_rate": 0.35,
            "weight_decay": 0.0001,
            "spectral_hidden": 32,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.2,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["Q"] = {
        "alias": "ppg_small_h_gate",
        "title": "Q_PPG_small_regularized_H_gate",
        "description": "PPG-only 倾向的小模型正则化 H，并用 gate 控制光谱特征进入血糖预测头。",
        "updates": {
            "model_name": "physics_spectral",
            "filter_structure": [32, 48, 64, 96],
            "dense_structure": [128, 64, 32, 1],
            "dropout_rate": 0.35,
            "weight_decay": 0.0001,
            "spectral_hidden": 32,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.2,
            "spectral_fusion_mode": "gate",
            "spectral_gate_init": -4.0,
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["R"] = {
        "alias": "ppg_original_lightreg_baseline",
        "title": "R_PPG_original_lightreg_baseline",
        "description": "PPG-only 原容量轻正则基线：保留原主干容量，仅提高 dropout 并加入很小 weight decay。",
        "updates": {
            "model_name": 0,
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.30,
            "weight_decay": 0.00001,
            "enable_token_moe": False,
            "enable_target_specific_regression": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["S"] = {
        "alias": "ppg_original_lightreg_h",
        "title": "S_PPG_original_lightreg_H",
        "description": "PPG-only 原容量轻正则 H：保留原容量，弱物理约束，轻 dropout/weight decay。",
        "updates": {
            "model_name": "physics_spectral",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.30,
            "weight_decay": 0.00001,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["T"] = {
        "alias": "ppg_large_baseline",
        "title": "T_PPG_large_baseline",
        "description": "PPG-only 更大容量基线：扩大主干和全连接层，使用轻正则。",
        "updates": {
            "model_name": 0,
            "filter_structure": [96, 128, 192, 256],
            "dense_structure": [384, 192, 96, 1],
            "dropout_rate": 0.30,
            "weight_decay": 0.00001,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["U"] = {
        "alias": "ppg_large_h",
        "title": "U_PPG_large_H",
        "description": "PPG-only 更大容量 H：扩大主干/回归头/光谱 hidden，保持弱物理约束。",
        "updates": {
            "model_name": "physics_spectral",
            "filter_structure": [96, 128, 192, 256],
            "dense_structure": [384, 192, 96, 1],
            "dropout_rate": 0.30,
            "weight_decay": 0.00001,
            "spectral_hidden": 96,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["V"] = {
        "alias": "ppg_large_h_gate",
        "title": "V_PPG_large_H_gate",
        "description": "PPG-only 更大容量 H-gate：扩大容量，并用 gate 控制光谱特征进入血糖预测头。",
        "updates": {
            "model_name": "physics_spectral",
            "filter_structure": [96, 128, 192, 256],
            "dense_structure": [384, 192, 96, 1],
            "dropout_rate": 0.30,
            "weight_decay": 0.00001,
            "spectral_hidden": 96,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "gate",
            "spectral_gate_init": -4.0,
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["W"] = {
        "alias": "ppg_tcn_baseline",
        "title": "W_PPG_TCN_baseline",
        "description": "PPG-only TCN 强基线：用 dilated Conv1d/TCN 替换 PPG ResNet 主干。",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "tcn",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["X"] = {
        "alias": "ppg_tcn_h",
        "title": "X_PPG_TCN_H",
        "description": "PPG-only TCN + H：TCN PPG 主干叠加弱物理光谱约束。",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "tcn",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["Y"] = {
        "alias": "ppg_gru_baseline",
        "title": "Y_PPG_GRU_baseline",
        "description": "PPG-only BiGRU 时序基线：用双向 GRU 建模 PPG 时间依赖。",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "gru",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["Z"] = {
        "alias": "ppg_gru_h",
        "title": "Z_PPG_GRU_H",
        "description": "PPG-only BiGRU + H：BiGRU 主干叠加弱物理光谱约束。",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "gru",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["CI"] = {
        "alias": "ppg_bilstm_baseline",
        "title": "CI_PPG_BiLSTM_baseline",
        "description": "PPG-only BiLSTM temporal baseline for sequence-dependent PPG modeling.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "bilstm",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["CJ"] = {
        "alias": "ppg_bilstm_h",
        "title": "CJ_PPG_BiLSTM_H",
        "description": "PPG-only BiLSTM backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "bilstm",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["CK"] = {
        "alias": "ppg_lstm_attention_baseline",
        "title": "CK_PPG_LSTM_Attention_baseline",
        "description": "PPG-only BiLSTM with temporal attention baseline.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "lstm_attention",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["CL"] = {
        "alias": "ppg_lstm_attention_h",
        "title": "CL_PPG_LSTM_Attention_H",
        "description": "PPG-only BiLSTM with temporal attention and the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "lstm_attention",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.25,
            "weight_decay": 0.0,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AA"] = {
        "alias": "ppg_transformer_baseline",
        "title": "AA_PPG_Transformer_baseline",
        "description": "PPG-only Conv-stem Transformer Encoder 基线：先提取局部 PPG 波形，再用自注意力建模全局时间依赖。",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "transformer",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.10,
            "weight_decay": 0.00001,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AB"] = {
        "alias": "ppg_transformer_h",
        "title": "AB_PPG_Transformer_H",
        "description": "PPG-only Conv-stem Transformer Encoder + H：Transformer 主干叠加弱物理光谱约束。",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "transformer",
            "filter_structure": [64, 96, 128, 160],
            "dense_structure": [256, 128, 64, 1],
            "dropout_rate": 0.10,
            "weight_decay": 0.00001,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AC"] = {
        "alias": "ppg_patchtst_baseline",
        "title": "AC_PPG_PatchTST_baseline",
        "description": "PPG-only PatchTST baseline using patch-wise Transformer encoding.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "patchtst",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_patch_len": 16,
            "ppg_patch_stride": 8,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AD"] = {
        "alias": "ppg_patchtst_h",
        "title": "AD_PPG_PatchTST_H",
        "description": "PPG-only PatchTST backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "patchtst",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_patch_len": 16,
            "ppg_patch_stride": 8,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AE"] = {
        "alias": "ppg_timesnet_baseline",
        "title": "AE_PPG_TimesNet_baseline",
        "description": "PPG-only TimesNet baseline with temporal 2D-variation modeling.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "timesnet",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_tsl_top_k": 3,
            "ppg_tsl_num_kernels": 3,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AF"] = {
        "alias": "ppg_timesnet_h",
        "title": "AF_PPG_TimesNet_H",
        "description": "PPG-only TimesNet backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "timesnet",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_tsl_top_k": 3,
            "ppg_tsl_num_kernels": 3,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AG"] = {
        "alias": "ppg_itransformer_baseline",
        "title": "AG_PPG_iTransformer_baseline",
        "description": "PPG-only iTransformer baseline for inter-channel token modeling.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "itransformer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AH"] = {
        "alias": "ppg_itransformer_h",
        "title": "AH_PPG_iTransformer_H",
        "description": "PPG-only iTransformer backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "itransformer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AI"] = {
        "alias": "ppg_timemixer_baseline",
        "title": "AI_PPG_TimeMixer_baseline",
        "description": "PPG-only TimeMixer baseline for multi-scale temporal mixing.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "timemixer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_tsl_down_sampling_layers": 1,
            "ppg_tsl_down_sampling_window": 2,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AJ"] = {
        "alias": "ppg_timemixer_h",
        "title": "AJ_PPG_TimeMixer_H",
        "description": "PPG-only TimeMixer backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "timemixer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_tsl_down_sampling_layers": 1,
            "ppg_tsl_down_sampling_window": 2,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AK"] = {
        "alias": "ppg_timefilter_baseline",
        "title": "AK_PPG_TimeFilter_baseline",
        "description": "PPG-only TimeFilter baseline for graph-style channel-time filtering.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "timefilter",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_patch_len": 20,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AL"] = {
        "alias": "ppg_timefilter_h",
        "title": "AL_PPG_TimeFilter_H",
        "description": "PPG-only TimeFilter backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "timefilter",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_e_layers": 1,
            "ppg_patch_len": 20,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    specs["AM"] = {
        "alias": "ppg_wpmixer_baseline",
        "title": "AM_PPG_WPMixer_baseline",
        "description": "PPG-only WPMixer baseline using wavelet multi-resolution mixing.",
        "updates": {
            "model_name": 0,
            "ppg_backbone_name": "wpmixer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_pred_len": 300,
            "ppg_patch_len": 16,
            "enable_token_moe": False,
            "lambda_obs": 0.0,
            "lambda_smooth": 0.0,
            "lambda_decorr": 0.0,
            "spectral_export_samples": 0,
        },
    }

    specs["AN"] = {
        "alias": "ppg_wpmixer_h",
        "title": "AN_PPG_WPMixer_H",
        "description": "PPG-only WPMixer backbone with the proposed physics-spectral plug-in.",
        "updates": {
            "model_name": "physics_spectral",
            "ppg_backbone_name": "wpmixer",
            "dropout_rate": 0.20,
            "weight_decay": 0.00001,
            "ppg_tsl_d_model": 64,
            "ppg_tsl_pred_len": 300,
            "ppg_patch_len": 16,
            "spectral_hidden": 64,
            "enable_token_moe": True,
            "num_experts": 4,
            "token_moe_dropout": 0.1,
            "spectral_fusion_mode": "concat",
            "lambda_obs": 0.03,
            "lambda_smooth": 0.001,
            "lambda_decorr": 0.0001,
        },
    }

    strong_common = {
        "initial_lr": 0.0005,
        "dropout_rate": 0.10,
        "weight_decay": 0.00001,
        "ppg_tsl_d_model": 128,
        "ppg_tsl_d_ff": 256,
        "ppg_tsl_e_layers": 2,
        "ppg_tsl_n_heads": 8,
    }
    strong_base_common = {
        **strong_common,
        "model_name": 0,
        "enable_token_moe": False,
        "lambda_obs": 0.0,
        "lambda_smooth": 0.0,
        "lambda_decorr": 0.0,
        "spectral_export_samples": 0,
    }
    strong_h_common = {
        **strong_common,
        "model_name": "physics_spectral",
        "spectral_hidden": 128,
        "enable_token_moe": True,
        "num_experts": 4,
        "token_moe_dropout": 0.1,
        "spectral_fusion_mode": "concat",
        "lambda_obs": 0.03,
        "lambda_smooth": 0.001,
        "lambda_decorr": 0.0001,
    }

    strong_specs = [
        ("AO", "ppg_patchtst_strong_baseline", "AO_PPG_PatchTST_strong_baseline",
         "PPG-only PatchTST strong baseline: d_model=128, e_layers=2, dropout=0.10.",
         "patchtst", {"ppg_patch_len": 16, "ppg_patch_stride": 8}),
        ("AP", "ppg_patchtst_strong_h", "AP_PPG_PatchTST_strong_H",
         "PPG-only PatchTST strong backbone with the proposed physics-spectral plug-in.",
         "patchtst", {"ppg_patch_len": 16, "ppg_patch_stride": 8}),
        ("AQ", "ppg_timesnet_strong_baseline", "AQ_PPG_TimesNet_strong_baseline",
         "PPG-only TimesNet strong baseline: d_model=128, e_layers=2, top_k=5, num_kernels=6.",
         "timesnet", {"ppg_tsl_top_k": 5, "ppg_tsl_num_kernels": 6}),
        ("AR", "ppg_timesnet_strong_h", "AR_PPG_TimesNet_strong_H",
         "PPG-only TimesNet strong backbone with the proposed physics-spectral plug-in.",
         "timesnet", {"ppg_tsl_top_k": 5, "ppg_tsl_num_kernels": 6}),
        ("AS", "ppg_itransformer_strong_baseline", "AS_PPG_iTransformer_strong_baseline",
         "PPG-only iTransformer strong baseline: d_model=128, e_layers=2, n_heads=8.",
         "itransformer", {}),
        ("AT", "ppg_itransformer_strong_h", "AT_PPG_iTransformer_strong_H",
         "PPG-only iTransformer strong backbone with the proposed physics-spectral plug-in.",
         "itransformer", {}),
        ("AU", "ppg_timemixer_strong_baseline", "AU_PPG_TimeMixer_strong_baseline",
         "PPG-only TimeMixer strong baseline: d_model=128, e_layers=2, two down-sampling scales.",
         "timemixer", {"ppg_tsl_down_sampling_layers": 2, "ppg_tsl_down_sampling_window": 2}),
        ("AV", "ppg_timemixer_strong_h", "AV_PPG_TimeMixer_strong_H",
         "PPG-only TimeMixer strong backbone with the proposed physics-spectral plug-in.",
         "timemixer", {"ppg_tsl_down_sampling_layers": 2, "ppg_tsl_down_sampling_window": 2}),
        ("AW", "ppg_timefilter_strong_baseline", "AW_PPG_TimeFilter_strong_baseline",
         "PPG-only TimeFilter strong baseline: d_model=128, e_layers=2, patch_len=20.",
         "timefilter", {"ppg_patch_len": 20}),
        ("AX", "ppg_timefilter_strong_h", "AX_PPG_TimeFilter_strong_H",
         "PPG-only TimeFilter strong backbone with the proposed physics-spectral plug-in.",
         "timefilter", {"ppg_patch_len": 20}),
    ]

    for key, alias, title, description, backbone_name, extra_updates in strong_specs:
        is_h = key in ["AP", "AR", "AT", "AV", "AX"]
        updates = (strong_h_common if is_h else strong_base_common).copy()
        updates.update({
            "ppg_backbone_name": backbone_name,
            **extra_updates,
        })
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": updates,
        }

    fedformer_common = {
        "initial_lr": 0.0005,
        "dropout_rate": 0.10,
        "weight_decay": 0.00001,
        "ppg_tsl_d_model": 64,
        "ppg_tsl_d_ff": 128,
        "ppg_tsl_e_layers": 1,
        "ppg_tsl_d_layers": 1,
        "ppg_tsl_n_heads": 4,
        "ppg_tsl_pred_len": 300,
        "ppg_tsl_moving_avg": 25,
        "ppg_tsl_modes": 16,
        "ppg_tsl_mode_select": "random",
        "ppg_tsl_version": "fourier",
    }
    fedformer_base_common = {
        **fedformer_common,
        "model_name": 0,
        "enable_token_moe": False,
        "lambda_obs": 0.0,
        "lambda_smooth": 0.0,
        "lambda_decorr": 0.0,
        "spectral_export_samples": 0,
    }
    fedformer_h_common = {
        **fedformer_common,
        "model_name": "physics_spectral",
        "spectral_hidden": 128,
        "enable_token_moe": True,
        "num_experts": 4,
        "token_moe_dropout": 0.1,
        "spectral_fusion_mode": "concat",
        "lambda_obs": 0.03,
        "lambda_smooth": 0.001,
        "lambda_decorr": 0.0001,
    }
    specs["BI"] = {
        "alias": "ppg_fedformer_baseline",
        "title": "BI_PPG_FEDformer_baseline",
        "description": "PPG-only FEDformer baseline using frequency-enhanced decomposed Transformer forecasting.",
        "updates": {
            **fedformer_base_common,
            "ppg_backbone_name": "fedformer",
        },
    }
    specs["BJ"] = {
        "alias": "ppg_fedformer_h",
        "title": "BJ_PPG_FEDformer_H",
        "description": "PPG-only FEDformer backbone with the proposed physics-spectral plug-in.",
        "updates": {
            **fedformer_h_common,
            "ppg_backbone_name": "fedformer",
        },
    }

    biosignal_common = {
        "initial_lr": 0.001,
        "dropout_rate": 0.25,
        "weight_decay": 0.00001,
    }
    biosignal_base_common = {
        **biosignal_common,
        "model_name": 0,
        "enable_token_moe": False,
        "lambda_obs": 0.0,
        "lambda_smooth": 0.0,
        "lambda_decorr": 0.0,
        "spectral_export_samples": 0,
    }
    biosignal_h_common = {
        **biosignal_common,
        "model_name": "physics_spectral",
        "spectral_hidden": 128,
        "enable_token_moe": True,
        "num_experts": 4,
        "token_moe_dropout": 0.1,
        "spectral_fusion_mode": "concat",
        "lambda_obs": 0.03,
        "lambda_smooth": 0.001,
        "lambda_decorr": 0.0001,
    }
    biosignal_specs = [
        ("BK", "ppg_eegnet_baseline", "BK_PPG_EEGNet_baseline",
         "PPG-only EEGNet-style compact depthwise-separable CNN baseline.",
         "eegnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_temporal_filters": 16,
             "ppg_bio_depth_multiplier": 2,
             "ppg_bio_separable_filters": 64,
             "ppg_bio_kernel_size": 64,
         }),
        ("BL", "ppg_eegnet_h", "BL_PPG_EEGNet_H",
         "PPG-only EEGNet-style backbone with the proposed physics-spectral plug-in.",
         "eegnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_temporal_filters": 16,
             "ppg_bio_depth_multiplier": 2,
             "ppg_bio_separable_filters": 64,
             "ppg_bio_kernel_size": 64,
         }),
        ("BM", "ppg_shallowconvnet_baseline", "BM_PPG_ShallowConvNet_baseline",
         "PPG-only ShallowConvNet/ShallowFBCSPNet-style baseline.",
         "shallowconvnet", {
             "ppg_bio_hidden": 96,
             "ppg_bio_kernel_size": 25,
             "ppg_bio_pool_size": 20,
             "ppg_bio_pool_stride": 5,
         }),
        ("BN", "ppg_shallowconvnet_h", "BN_PPG_ShallowConvNet_H",
         "PPG-only ShallowConvNet-style backbone with the proposed physics-spectral plug-in.",
         "shallowconvnet", {
             "ppg_bio_hidden": 96,
             "ppg_bio_kernel_size": 25,
             "ppg_bio_pool_size": 20,
             "ppg_bio_pool_stride": 5,
         }),
        ("BO", "ppg_deepconvnet_baseline", "BO_PPG_DeepConvNet_baseline",
         "PPG-only DeepConvNet/Deep4Net-style baseline.",
         "deepconvnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_layers": 4,
             "ppg_bio_kernel_size": 7,
         }),
        ("BP", "ppg_deepconvnet_h", "BP_PPG_DeepConvNet_H",
         "PPG-only DeepConvNet-style backbone with the proposed physics-spectral plug-in.",
         "deepconvnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_layers": 4,
             "ppg_bio_kernel_size": 7,
         }),
        ("BQ", "ppg_atcnet_baseline", "BQ_PPG_ATCNet_baseline",
         "PPG-only ATCNet-style attention temporal convolution baseline.",
         "atcnet", {
             "ppg_bio_hidden": 96,
             "ppg_bio_layers": 3,
             "ppg_bio_heads": 4,
             "ppg_bio_kernel_size": 15,
         }),
        ("BR", "ppg_atcnet_h", "BR_PPG_ATCNet_H",
         "PPG-only ATCNet-style backbone with the proposed physics-spectral plug-in.",
         "atcnet", {
             "ppg_bio_hidden": 96,
             "ppg_bio_layers": 3,
             "ppg_bio_heads": 4,
             "ppg_bio_kernel_size": 15,
         }),
        ("BS", "ppg_eegconformer_baseline", "BS_PPG_EEGConformer_baseline",
         "PPG-only EEG Conformer-style CNN-Transformer baseline.",
         "eegconformer", {
             "ppg_bio_hidden": 96,
             "ppg_bio_layers": 2,
             "ppg_bio_heads": 4,
             "ppg_bio_kernel_size": 25,
         }),
        ("BT", "ppg_eegconformer_h", "BT_PPG_EEGConformer_H",
         "PPG-only EEG Conformer-style backbone with the proposed physics-spectral plug-in.",
         "eegconformer", {
             "ppg_bio_hidden": 96,
             "ppg_bio_layers": 2,
             "ppg_bio_heads": 4,
             "ppg_bio_kernel_size": 25,
         }),
        ("BU", "ppg_ecgresnet_baseline", "BU_PPG_ECGResNet_baseline",
         "PPG-only ECG ResNet-style waveform CNN baseline.",
         "ecgresnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_layers": 5,
         }),
        ("BV", "ppg_ecgresnet_h", "BV_PPG_ECGResNet_H",
         "PPG-only ECG ResNet-style backbone with the proposed physics-spectral plug-in.",
         "ecgresnet", {
             "ppg_bio_hidden": 64,
             "ppg_bio_layers": 5,
         }),
    ]

    for key, alias, title, description, backbone_name, extra_updates in biosignal_specs:
        is_h = key in ["BL", "BN", "BP", "BR", "BT", "BV"]
        updates = (biosignal_h_common if is_h else biosignal_base_common).copy()
        updates.update({
            "ppg_backbone_name": backbone_name,
            **extra_updates,
        })
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": updates,
        }

    recent_common = {
        "initial_lr": 0.001,
        "dropout_rate": 0.10,
        "weight_decay": 0.00001,
        "filter_structure": [64, 96, 128, 160],
        "ppg_recent_hidden": 128,
        "ppg_recent_layers": 3,
        "ppg_recent_heads": 4,
        "ppg_recent_patch_len": 16,
        "ppg_recent_patch_stride": 8,
        "ppg_recent_kernel_size": 15,
    }
    recent_base_common = {
        **recent_common,
        "model_name": 0,
        "enable_token_moe": False,
        "lambda_obs": 0.0,
        "lambda_smooth": 0.0,
        "lambda_decorr": 0.0,
        "spectral_export_samples": 0,
    }
    recent_h_common = {
        **recent_common,
        **h_low_loss_updates,
        "spectral_hidden": 128,
    }
    recent_specs = [
        ("BW", "ppg_papagei_baseline", "BW_PPG_PaPaGei_baseline",
         "PPG-only official PaPaGei-S ResNet1D-MoE backbone adapted to the local six-channel PPG format.",
         "papagei", {
             "ppg_papagei_base_filters": 32,
             "ppg_papagei_kernel_size": 3,
             "ppg_papagei_stride": 2,
             "ppg_papagei_groups": 1,
             "ppg_papagei_blocks": 18,
             "ppg_papagei_embedding_dim": 512,
             "ppg_papagei_experts": 3,
             "ppg_papagei_variant": "s",
             "ppg_papagei_channel_mode": "shared",
             "ppg_papagei_target_len": 0,
             "ppg_papagei_weight_path": str(REPO_ROOT / "weights" / "papagei_s.pt"),
             "ppg_papagei_freeze_encoder": False,
         }),
        ("BX", "ppg_papagei_h", "BX_PPG_PaPaGei_H",
         "PPG-only official PaPaGei-S ResNet1D-MoE backbone with the proposed physics-spectral plug-in.",
         "papagei", {
             "ppg_papagei_base_filters": 32,
             "ppg_papagei_kernel_size": 3,
             "ppg_papagei_stride": 2,
             "ppg_papagei_groups": 1,
             "ppg_papagei_blocks": 18,
             "ppg_papagei_embedding_dim": 512,
             "ppg_papagei_experts": 3,
             "ppg_papagei_variant": "s",
             "ppg_papagei_channel_mode": "shared",
             "ppg_papagei_target_len": 0,
             "ppg_papagei_weight_path": str(REPO_ROOT / "weights" / "papagei_s.pt"),
             "ppg_papagei_freeze_encoder": False,
         }),
        ("BY", "ppg_csfm_baseline", "BY_PPG_CSFM_baseline",
         "PPG-only official Cardiac-Sensing-FM Transformer encoder adapted to six-channel PPG, tuned for small PPG regression.",
         "csfm", {
             "ppg_csfm_variant": "Tiny",
             "ppg_csfm_signal_size": 300,
             "ppg_csfm_patch_size": 15,
             "ppg_csfm_channel_index": 12,
             "ppg_csfm_hidden_dim": 192,
             "ppg_csfm_depth": 3,
             "ppg_csfm_heads": 4,
             "ppg_csfm_mlp_dim": 512,
             "ppg_csfm_dim_head": 32,
             "ppg_csfm_freeze_encoder": False,
             "dropout_rate": 0.05,
         }),
        ("BZ", "ppg_csfm_h", "BZ_PPG_CSFM_H",
         "PPG-only official Cardiac-Sensing-FM Transformer encoder with the proposed physics-spectral plug-in, tuned for small PPG regression.",
         "csfm", {
             "ppg_csfm_variant": "Tiny",
             "ppg_csfm_signal_size": 300,
             "ppg_csfm_patch_size": 15,
             "ppg_csfm_channel_index": 12,
             "ppg_csfm_hidden_dim": 192,
             "ppg_csfm_depth": 3,
             "ppg_csfm_heads": 4,
             "ppg_csfm_mlp_dim": 512,
             "ppg_csfm_dim_head": 32,
             "ppg_csfm_freeze_encoder": False,
             "dropout_rate": 0.05,
         }),
        ("CA", "ppg_medformer_baseline", "CA_PPG_Medformer_baseline",
         "PPG-only official Medformer multi-granularity patching Transformer baseline, tuned for six-channel PPG.",
         "medformer", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 2,
             "ppg_recent_heads": 4,
             "ppg_recent_patch_sizes": "10,20,30",
             "ppg_medformer_stride_sizes": "5,10,15",
             "ppg_medformer_d_ff": 384,
             "ppg_medformer_single_channel": True,
             "ppg_medformer_no_inter_attn": False,
             "ppg_medformer_activation": "gelu",
             "dropout_rate": 0.05,
         }),
        ("CB", "ppg_medformer_h", "CB_PPG_Medformer_H",
         "PPG-only official Medformer multi-granularity backbone with the proposed physics-spectral plug-in, tuned for six-channel PPG.",
         "medformer", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 2,
             "ppg_recent_heads": 4,
             "ppg_recent_patch_sizes": "10,20,30",
             "ppg_medformer_stride_sizes": "5,10,15",
             "ppg_medformer_d_ff": 384,
             "ppg_medformer_single_channel": True,
             "ppg_medformer_no_inter_attn": False,
             "ppg_medformer_activation": "gelu",
             "dropout_rate": 0.05,
         }),
        ("CC", "ppg_tslanet_baseline", "CC_PPG_TSLANet_baseline",
         "PPG-only official TSLANet ASB/ICB encoder adapted to the local PPG format.",
         "tslanet", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 3,
             "ppg_recent_patch_len": 16,
             "ppg_recent_patch_stride": 8,
             "ppg_tslanet_mlp_ratio": 3.0,
             "ppg_tslanet_asb": True,
             "ppg_tslanet_icb": True,
             "ppg_tslanet_adaptive_filter": True,
         }),
        ("CD", "ppg_tslanet_h", "CD_PPG_TSLANet_H",
         "PPG-only official TSLANet ASB/ICB encoder with the proposed physics-spectral plug-in.",
         "tslanet", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 3,
             "ppg_recent_patch_len": 16,
             "ppg_recent_patch_stride": 8,
             "ppg_tslanet_mlp_ratio": 3.0,
             "ppg_tslanet_asb": True,
             "ppg_tslanet_icb": True,
             "ppg_tslanet_adaptive_filter": True,
         }),
        ("CE", "ppg_moment_baseline", "CE_PPG_MOMENT_baseline",
         "PPG-only official MOMENT RevIN + patch embedding encoder baseline, tuned to preserve six-channel PPG information.",
         "moment", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 2,
             "ppg_recent_heads": 4,
             "ppg_recent_patch_len": 16,
             "ppg_recent_patch_stride": 8,
             "ppg_moment_d_ff": 512,
             "ppg_moment_patch_dropout": 0.05,
             "ppg_moment_revin_affine": True,
             "ppg_moment_add_positional_embedding": True,
             "ppg_moment_value_embedding_bias": False,
             "ppg_moment_orth_gain": 1.41,
             "ppg_moment_mask_ratio": 0.0,
             "ppg_moment_channel_reduction": "concat",
             "ppg_moment_pad_end": True,
             "dropout_rate": 0.05,
         }),
        ("CF", "ppg_moment_h", "CF_PPG_MOMENT_H",
         "PPG-only official MOMENT RevIN + patch embedding encoder with the proposed physics-spectral plug-in, tuned to preserve six-channel PPG information.",
         "moment", {
             "ppg_recent_hidden": 128,
             "ppg_recent_layers": 2,
             "ppg_recent_heads": 4,
             "ppg_recent_patch_len": 16,
             "ppg_recent_patch_stride": 8,
             "ppg_moment_d_ff": 512,
             "ppg_moment_patch_dropout": 0.05,
             "ppg_moment_revin_affine": True,
             "ppg_moment_add_positional_embedding": True,
             "ppg_moment_value_embedding_bias": False,
             "ppg_moment_orth_gain": 1.41,
             "ppg_moment_mask_ratio": 0.0,
             "ppg_moment_channel_reduction": "concat",
             "ppg_moment_pad_end": True,
             "dropout_rate": 0.05,
         }),
        ("CG", "ppg_papagei_scratch_baseline", "CG_PPG_PaPaGei_Scratch_baseline",
         "PPG-only official PaPaGei-S ResNet1D-MoE backbone trained from scratch with direct six-channel adaptation.",
         "papagei", {
             "ppg_papagei_base_filters": 32,
             "ppg_papagei_kernel_size": 3,
             "ppg_papagei_stride": 2,
             "ppg_papagei_groups": 1,
             "ppg_papagei_blocks": 18,
             "ppg_papagei_embedding_dim": 256,
             "ppg_papagei_experts": 3,
             "ppg_papagei_variant": "s",
             "ppg_papagei_channel_mode": "direct",
             "ppg_papagei_target_len": 0,
             "ppg_papagei_freeze_encoder": False,
         }),
        ("CH", "ppg_papagei_scratch_h", "CH_PPG_PaPaGei_Scratch_H",
         "PPG-only official PaPaGei-S ResNet1D-MoE backbone trained from scratch with the proposed physics-spectral plug-in.",
         "papagei", {
             "ppg_papagei_base_filters": 32,
             "ppg_papagei_kernel_size": 3,
             "ppg_papagei_stride": 2,
             "ppg_papagei_groups": 1,
             "ppg_papagei_blocks": 18,
             "ppg_papagei_embedding_dim": 256,
             "ppg_papagei_experts": 3,
             "ppg_papagei_variant": "s",
             "ppg_papagei_channel_mode": "direct",
             "ppg_papagei_target_len": 0,
             "ppg_papagei_freeze_encoder": False,
         }),
    ]

    for key, alias, title, description, backbone_name, extra_updates in recent_specs:
        is_h = key in ["BX", "BZ", "CB", "CD", "CF", "CH"]
        updates = (recent_h_common if is_h else recent_base_common).copy()
        updates.update({
            "ppg_backbone_name": backbone_name,
            **extra_updates,
        })
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": updates,
        }

    # Official comparison pairs use the current Lite_H plug-in for every +H model.
    # Keep the backbone-specific capacity/configuration, but remove the artifact
    # pretext and context-adversarial branches from the role-aware spectral H module.
    context_adv_comparison_h_keys = [
        "S", "Z", "BN", "BX", "BZ", "CB", "CD", "CH",
        "AD", "AP", "CJ", "CL",
    ]
    preserve_exact_keys = {
        "initial_lr",
        "filter_structure",
        "dense_structure",
        "dropout_rate",
        "weight_decay",
        "spectral_hidden",
    }
    for key in context_adv_comparison_h_keys:
        if key not in specs:
            continue
        original_updates = specs[key]["updates"].copy()
        updated = original_updates.copy()
        updated.update(lite_h_updates)
        for preserve_key in preserve_exact_keys:
            if preserve_key in original_updates:
                updated[preserve_key] = original_updates[preserve_key]
        for preserve_key, preserve_value in original_updates.items():
            if preserve_key.startswith("ppg_"):
                updated[preserve_key] = preserve_value
        specs[key]["updates"] = updated
        specs[key]["title"] = (
            specs[key]["title"]
            .replace("_ContextAdv_H", "_Lite_H")
            .replace("_H", "_Lite_H")
        )
        specs[key]["description"] = (
            specs[key]["description"] +
            " Uses the current Lite_H plug-in with role-aware spectral decomposition "
            "but without artifact pretext or context-adversarial regularization.")

    # Cross-backbone validation of the final H4A0 setting. Build these from
    # each backbone's existing +H configuration after the common Lite-H merge,
    # then apply the sole HL4 change: remove target-spectrum L1 shrinkage.
    h4a0_backbone_specs = [
        (
            "H4R", "h4a0_resnet", "H4R_ResNet1D_H4A0",
            "ResNet1D with the final H4A0 role-aware spectral plug-in.", "S"),
        (
            "H4G", "h4a0_gru", "H4G_GRU_H4A0",
            "GRU with the final H4A0 role-aware spectral plug-in.", "Z"),
        (
            "H4B", "h4a0_bilstm", "H4B_BiLSTM_H4A0",
            "BiLSTM with the final H4A0 role-aware spectral plug-in.", "CJ"),
        (
            "H4C", "h4a0_csfm", "H4C_CSFM_H4A0",
            "CSFM with the final H4A0 role-aware spectral plug-in.", "BZ"),
        (
            "H4P", "h4a0_patchtst_strong", "H4P_PatchTST_strong_H4A0",
            "PatchTST-strong with the final H4A0 role-aware spectral plug-in.", "AP"),
        (
            "H4S", "h4a0_shallowconvnet", "H4S_ShallowConvNet_H4A0",
            "ShallowConvNet with the final H4A0 role-aware spectral plug-in.", "BN"),
        (
            "H4PG", "h4a0_papagei_scratch", "H4PG_PaPaGei_scratch_H4A0",
            "PaPaGei scratch backbone with the final H4A0 role-aware spectral plug-in.", "CH"),
        (
            "H4PW", "h4a0_papagei_pretrained", "H4PW_PaPaGei_pretrained_H4A0",
            "Pretrained PaPaGei-S backbone with the final H4A0 role-aware spectral plug-in.", "BX"),
        (
            "H4T", "h4a0_tslanet", "H4T_TSLANet_H4A0",
            "TSLANet with the final H4A0 role-aware spectral plug-in.", "CD"),
        (
            "H4M", "h4a0_medformer", "H4M_Medformer_H4A0",
            "Medformer with the final H4A0 role-aware spectral plug-in.", "CB"),
    ]
    for key, alias, title, description, source_h_key in h4a0_backbone_specs:
        updates = specs[source_h_key]["updates"].copy()
        updates["lambda_target_sparse"] = 0.0
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": updates,
        }

    # Compact paper ablation on the final ResNet1D configuration. Each variant
    # removes one conceptual block from H4R so the table maps directly to the
    # method diagram instead of splitting individual regularization weights.
    resnet_h4_compact_specs = [
        (
            "H4R1",
            "resnet_h4_no_obs",
            "H4R1_ResNet1D_H4A0_no_observation_H",
            "ResNet1D H4A0 without fixed-H observation consistency.",
            {"lambda_obs": 0.0},
        ),
        (
            "H4R2",
            "resnet_h4_no_smooth_basis",
            "H4R2_ResNet1D_H4A0_no_smooth_spectral_bases",
            "ResNet1D H4A0 with pointwise spectral generation and no spectral smoothness losses.",
            {
                "spectral_generator_mode": "pointwise",
                "lambda_smooth": 0.0,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_basis_smooth": 0.0,
            },
        ),
        (
            "H4R3",
            "resnet_h4_no_roles",
            "H4R3_ResNet1D_H4A0_no_role_decomposition",
            "ResNet1D H4A0 with the role-aware decomposition replaced by an anonymous spectral MoE.",
            {
                "token_moe_mode": "anonymous",
                "role_moe_gate": False,
                "enable_aux_spo2_spectrum": False,
                "enable_target_specific_regression": False,
                "enable_background_context_conditioning": False,
                "lambda_background_smooth": 0.0,
                "lambda_target_smooth": 0.0,
                "lambda_role_orth": 0.0,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
        (
            "H4R4",
            "resnet_h4_no_context",
            "H4R4_ResNet1D_H4A0_no_background_context",
            "ResNet1D H4A0 without context-conditioned background FiLM.",
            {"enable_background_context_conditioning": False},
        ),
        (
            "H4R5",
            "resnet_h4_no_spo2_component",
            "H4R5_ResNet1D_H4A0_no_aux_spo2_component",
            "ResNet1D H4A0 without the auxiliary SpO2 spectral component and target-specific residual heads.",
            {
                "enable_aux_spo2_spectrum": False,
                "enable_target_specific_regression": False,
                "lambda_glucose_component": 0.0,
                "lambda_spo2_component": 0.0,
            },
        ),
        (
            "H4R6",
            "resnet_h4_no_bn_branches",
            "H4R6_ResNet1D_H4A0_no_baseline_noise_branches",
            "ResNet1D H4A0 without the observation-domain baseline drift and high-frequency residual branches.",
            {"enable_observation_artifact_branches": False},
        ),
    ]
    for key, alias, title, description, extra_updates in resnet_h4_compact_specs:
        updates = specs["H4R"]["updates"].copy()
        updates.update(extra_updates)
        specs[key] = {
            "alias": alias,
            "title": title,
            "description": description,
            "updates": updates,
        }

    return specs


def parse_binary_mask(mask, expected_len, prefix):
    mask = str(mask).strip()
    if len(mask) != expected_len or any(ch not in "01" for ch in mask):
        raise ValueError(f"{prefix} must be a {expected_len}-bit string containing only 0/1, got: {mask}")
    return {f"{prefix}_{i + 1}": int(mask[i]) for i in range(expected_len)}


def resolve_requested_experiments(requested):
    specs = get_ablation_specs()
    alias_to_key = {}
    for key, spec in specs.items():
        alias_to_key[key.lower()] = key
        alias_to_key[spec["alias"].lower()] = key
        alias_to_key[spec["title"].lower()] = key

    group_aliases = {
        **ARCHIVED_EXPERIMENT_GROUPS,
        "h_search": ["I", "J", "K", "L", "M", "N"],
        "local_search": ["I", "J", "K", "L", "M", "N"],
        "weight_search": ["I", "J", "K", "L", "M", "N"],
        "ppg_reg": ["O", "P", "Q"],
        "small_reg": ["O", "P", "Q"],
        "ppg_capacity": ["R", "S", "T", "U", "V"],
        "large": ["T", "U", "V"],
        "large_reg": ["T", "U", "V"],
        "ppg_tcn": ["W", "X"],
        "tcn": ["W", "X"],
        "ppg_sequence": ["Y", "Z", "CI", "CJ", "CK", "CL", "AA", "AB"],
        "sequence": ["Y", "Z", "CI", "CJ", "CK", "CL", "AA", "AB"],
        "gru": ["Y", "Z"],
        "bilstm": ["CI", "CJ"],
        "lstm": ["CI", "CJ"],
        "lstm_attention": ["CK", "CL"],
        "bilstm_attention": ["CK", "CL"],
        "rnn": ["Y", "Z", "CI", "CJ", "CK", "CL"],
        "transformer": ["AA", "AB"],
        "ppg_patchtst": ["AC", "AD"],
        "patchtst": ["AC", "AD"],
        "ppg_timesnet": ["AE", "AF"],
        "timesnet": ["AE", "AF"],
        "ppg_itransformer": ["AG", "AH"],
        "itransformer": ["AG", "AH"],
        "ppg_timemixer": ["AI", "AJ"],
        "timemixer": ["AI", "AJ"],
        "ppg_timefilter": ["AK", "AL"],
        "timefilter": ["AK", "AL"],
        "ppg_wpmixer": ["AM", "AN"],
        "wpmixer": ["AM", "AN"],
        "ppg_plug": ["R", "S", "W", "X", "Y", "Z", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL"],
        "ppg_plug_all": ["R", "S", "W", "X", "Y", "Z", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN"],
        "ppg_latest": ["AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN"],
        "ppg_strong": ["AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX"],
        "ppg_strong_transformer": ["AO", "AP", "AQ", "AR", "AS", "AT"],
        "ppg_strong_mixer": ["AU", "AV", "AW", "AX"],
        "ppg_fedformer": ["BI", "BJ"],
        "fedformer": ["BI", "BJ"],
        "ppg_biosignal": ["BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV"],
        "biosignal": ["BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV"],
        "ppg_eegnet": ["BK", "BL"],
        "eegnet": ["BK", "BL"],
        "ppg_shallowconvnet": ["BM", "BN"],
        "shallowconvnet": ["BM", "BN"],
        "ppg_deepconvnet": ["BO", "BP"],
        "deepconvnet": ["BO", "BP"],
        "ppg_atcnet": ["BQ", "BR"],
        "atcnet": ["BQ", "BR"],
        "ppg_eegconformer": ["BS", "BT"],
        "eegconformer": ["BS", "BT"],
        "ppg_ecgresnet": ["BU", "BV"],
        "ecgresnet": ["BU", "BV"],
        "ppg_recent": ["BW", "BX", "BY", "BZ", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH"],
        "recent": ["BW", "BX", "BY", "BZ", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH"],
        "top_recent": ["BW", "BX", "BY", "BZ", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH"],
        "ppg_foundation": ["BW", "BX", "CG", "CH", "BY", "BZ", "CE", "CF"],
        "foundation": ["BW", "BX", "CG", "CH", "BY", "BZ", "CE", "CF"],
        "ppg_papagei": ["BW", "BX", "CG", "CH"],
        "papagei": ["BW", "BX", "CG", "CH"],
        "ppg_papagei_pretrained": ["BW", "BX"],
        "papagei_pretrained": ["BW", "BX"],
        "ppg_papagei_scratch": ["CG", "CH"],
        "papagei_scratch": ["CG", "CH"],
        "ppg_csfm": ["BY", "BZ"],
        "csfm": ["BY", "BZ"],
        "cardiac_sensing_fm": ["BY", "BZ"],
        "ppg_medformer": ["CA", "CB"],
        "medformer": ["CA", "CB"],
        "ppg_tslanet": ["CC", "CD"],
        "tslanet": ["CC", "CD"],
        "ppg_moment": ["CE", "CF"],
        "moment": ["CE", "CF"],
        "oximetry_h_search": ["OX05", "OX10"],
        "ox_h_search": ["OX05", "OX10"],
        "oximetry_light_h": ["OX05", "OX10"],
        "role_h": ["CM"],
        "role_aware_h": ["CM"],
        "artifact_h": ["CM"],
        "context_adv_h": ["CN"],
        "context_adversarial_h": ["CN"],
        "role_context_h": ["CN"],
        "lite_h": ["CO"],
        "h_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "h_ablate": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "h_lite_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "lite_h_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "current_h_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "lstm_attention_h_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "lstm_attn_h_ablation": ["HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6"],
        "h_combo_ablation": ["HC1", "HC2", "HC3", "HC4", "HC5"],
        "h_combination_ablation": ["HC1", "HC2", "HC3", "HC4", "HC5"],
        "lite_h_combo": ["HC1", "HC2", "HC3", "HC4", "HC5"],
        "hl4_ablation": ["H4A0", "H4A1", "H4A2", "H4A3", "H4A4", "H4A5", "H4A6", "H4A7", "H4A8", "H4A9", "H4A10"],
        "final_h_ablation": ["H4A0", "H4A1", "H4A2", "H4A3", "H4A4", "H4A5", "H4A6", "H4A7", "H4A8", "H4A9", "H4A10"],
        "role_h_ablation": ["H4A0", "H4A3", "H4A4", "H4A10"],
        "h4a0_backbones": ["R", "H4R", "Y", "H4G", "CI", "H4B", "BY", "H4C", "AO", "H4P"],
        "h4a0_other_backbones": ["R", "H4R", "Y", "H4G", "CI", "H4B", "BY", "H4C", "AO", "H4P"],
        "h4a0_cnn_rnn": ["R", "H4R", "Y", "H4G", "CI", "H4B"],
        "h4a0_transformers": ["BY", "H4C", "AO", "H4P"],
        "h4a0_biosignal_recent": ["BM", "H4S", "CG", "H4PG", "CC", "H4T", "BY", "H4C", "CA", "H4M"],
        "h4a0_paper_backbones": ["R", "H4R", "BM", "H4S", "Y", "H4G", "CI", "H4B", "CK", "H4A0", "CG", "H4PG", "AO", "H4P", "CC", "H4T", "BY", "H4C", "CA", "H4M"],
        "h4a0_all_pairs": ["R", "H4R", "BM", "H4S", "Y", "H4G", "CI", "H4B", "CK", "H4A0", "CG", "H4PG", "AO", "H4P", "CC", "H4T", "BY", "H4C", "CA", "H4M"],
        "h4a0_paper_gpu0": ["R", "H4R", "BM", "H4S", "Y", "H4G", "CI", "H4B", "CG", "H4PG"],
        "h4a0_paper_gpu1": ["CK", "H4A0", "AO", "H4P", "CC", "H4T", "BY", "H4C", "CA", "H4M"],
        "external_h4_resnet": ["R", "H4R"],
        "external_h4_selected": ["R", "H4R", "Y", "H4G", "CI", "H4B", "BY", "H4C", "AO", "H4P"],
        "external_h4_all": ["R", "H4R", "BM", "H4S", "Y", "H4G", "CI", "H4B", "CK", "H4A0", "CG", "H4PG", "AO", "H4P", "CC", "H4T", "BY", "H4C", "CA", "H4M"],
        "resnet_h4_compact_ablation": ["R", "H4R", "H4R1", "H4R2", "H4R3", "H4R4", "H4R5", "H4R6"],
        "resnet_h4_compact_missing": ["H4R1", "H4R2", "H4R3", "H4R4", "H4R5", "H4R6"],
        "context_h_ablation": ["HAB0", "HAB1", "HAB2", "HAB3", "HAB4", "HAB5", "HAB6", "HAB7"],
        "old_context_h_ablation": ["HAB0", "HAB1", "HAB2", "HAB3", "HAB4", "HAB5", "HAB6", "HAB7"],
    }

    selected = []
    for item in requested:
        group = group_aliases.get(item.lower())
        if group is not None:
            for key in group:
                if key not in selected:
                    selected.append(key)
            continue

        key = alias_to_key.get(item.lower())
        if key is None:
            valid = ", ".join([f"{k}/{v['alias']}" for k, v in specs.items()])
            valid = valid + ", archived_ah, h_search, ppg_reg, ppg_capacity, ppg_tcn, ppg_sequence, ppg_plug, ppg_strong, ppg_fedformer, ppg_biosignal, ppg_recent, ppg_papagei_pretrained, ppg_papagei_scratch, oximetry_h_search, context_adv_h"
            raise ValueError(f"Unknown ablation experiment: {item}. Valid values: {valid}")
        if key not in selected:
            selected.append(key)
    return selected


def build_dataset(args):
    modal_enable_dic = parse_binary_mask(args.modal_mask, 5, "modal")
    ppg_enable_dic = parse_binary_mask(args.ppg_mask, 6, "ppg")

    if args.dataset == "oximetry":
        from src.oximetry_data import build_oximetry_dataset

        provider = build_oximetry_dataset(
            args.oximetry_root,
            modal_enable_dic=modal_enable_dic,
            ppg_enable_dic=ppg_enable_dic,
            split_mode=args.oximetry_split,
            test_subject=args.oximetry_test_subject,
            random_test_size=args.oximetry_random_test_size,
            random_state=args.oximetry_random_state,
            window_sec=args.oximetry_window_sec,
            stride_sec=args.oximetry_stride_sec,
            target=args.oximetry_target,
            normalize=args.oximetry_normalize,
            target_min=args.oximetry_target_min,
            target_max=args.oximetry_target_max,
            spectral_step=args.spectral_step,
            rgb_sigma=args.oximetry_rgb_sigma,
            spectral_min=args.oximetry_spectral_min,
            spectral_max=args.oximetry_spectral_max)
        dataset_dic = {
            "X_train": provider.X_train_list,
            "X_test": provider.X_test_list,
            "Y_train": provider.Y_train,
            "Y_test": provider.Y_test,
            "BG_Min_Max": provider.BG_Min_Max,
            "dataset_name": provider.dataset_name,
            "target_name": provider.target_name,
            "target_unit": provider.target_unit,
            "target_source": getattr(provider, "target_source", args.oximetry_target),
            "reference_device": getattr(provider, "reference_device", ""),
            "validation_group": getattr(provider, "validation_group", None),
            "validation_group_name": getattr(provider, "validation_group_name", None),
        }
        print(
            f"Oximetry dataset loaded: train={len(provider.Y_train)}, "
            f"test={len(provider.Y_test)}, split={provider.split_mode}, "
            f"target={provider.target_source}, reference={provider.reference_device}, "
            f"held_out={provider.test_subject if provider.split_mode == 'subject' else '-'}, "
            f"subject_windows={provider.subject_counts}")
        return provider, dataset_dic

    if args.dataset in OPENOX_DATASET_TARGETS:
        from src.openox_data import build_openox_dataset

        openox_target = OPENOX_DATASET_TARGETS[args.dataset]
        provider = build_openox_dataset(
            args.openox_root,
            modal_enable_dic=modal_enable_dic,
            ppg_enable_dic=ppg_enable_dic,
            target=openox_target,
            split_mode=args.openox_split,
            test_size=args.openox_test_size,
            random_state=args.openox_random_state,
            window_sec=args.openox_window_sec,
            output_len=args.openox_output_len,
            sampling_rate=args.openox_sampling_rate,
            normalize=args.openox_normalize,
            detrend=args.openox_detrend,
            detrend_cutoff=args.openox_detrend_cutoff,
            scale=args.openox_scale,
            target_min=args.openox_target_min,
            target_max=args.openox_target_max,
            spo2_source=args.openox_spo2_source,
            spo2_aggregation=args.openox_spo2_aggregation,
            spo2_max_device_range=args.openox_spo2_max_device_range,
            spo2_stride_sec=args.openox_spo2_stride_sec,
            spectral_step=args.spectral_step,
            spectral_sigma=args.openox_spectral_sigma,
            spectral_min=args.openox_spectral_min,
            spectral_max=args.openox_spectral_max,
            load_cache=args.load_cache,
            max_samples=args.openox_max_samples)
        dataset_dic = {
            "X_train": provider.X_train_list,
            "X_test": provider.X_test_list,
            "Y_train": provider.Y_train,
            "Y_test": provider.Y_test,
            "BG_Min_Max": provider.BG_Min_Max,
            "dataset_name": provider.dataset_name,
            "target_name": provider.target_name,
            "target_unit": provider.target_unit,
            "target_names": getattr(provider, "target_names", [provider.target_name]),
            "target_units": getattr(provider, "target_units", [provider.target_unit]),
            "validation_group": getattr(provider, "validation_group", None),
            "validation_group_name": getattr(provider, "validation_group_name", None),
        }
        print(
            f"OpenOx dataset loaded: targets={getattr(provider, 'target_names', [provider.target_name])}, "
            f"train={len(provider.Y_train)}, test={len(provider.Y_test)}, "
            f"split={provider.split_mode}, cache={provider.cache_path}, "
            f"from_cache={provider.loaded_from_cache}")
        print(
            f"OpenOx PPG channels={provider.X_train_list[0].shape[1]}, "
            f"length={provider.X_train_list[0].shape[2]}, "
            f"fs={provider.openox_sampling_rate:g}Hz, "
            f"window={provider.openox_window_sec:g}s, "
            f"preprocess={provider.openox_preprocess}, "
            f"detrend={provider.openox_detrend}({provider.openox_detrend_cutoff:g}Hz), "
            f"target_range=[{provider.BG_Min_Max[0, 0]:g}, {provider.BG_Min_Max[1, 0]:g}], "
            f"scale={provider.openox_scale}, "
            f"metadata rows={len(provider.metadata)}")
        if openox_target == "spo2":
            print(
                f"OpenOx SpO2 label={provider.openox_spo2_source}/"
                f"{provider.openox_spo2_aggregation}/"
                f"range<={provider.openox_spo2_max_device_range:g}/"
                f"stride={provider.openox_spo2_stride_sec:g}s")
        return provider, dataset_dic

    from src.load_data import data_provider

    provider = data_provider()
    provider.load_data(
        enable_load_data=args.load_cache,
        file_path=args.data_path,
        split_mode=getattr(args, "glucose_split", "record"),
        test_size=getattr(args, "glucose_test_size", 0.2),
        random_state=getattr(args, "glucose_random_state", 2),
        patient_column=getattr(args, "glucose_patient_column", 3),
        subwindow_sec=getattr(args, "glucose_subwindow_sec", 0.0),
        subwindow_stride_sec=getattr(args, "glucose_subwindow_stride_sec", 1.0),
        subwindow_sampling_rate=getattr(args, "glucose_subwindow_sampling_rate", 50.0))
    spo2_train = None
    spo2_test = None
    if getattr(args, "glucose_aux_spo2", False):
        spo2_train = np.asarray(provider.X_train_list[2][:, 1:2], dtype=np.float32).copy()
        spo2_test = np.asarray(provider.X_test_list[2][:, 1:2], dtype=np.float32).copy()
    provider.input_enable(modal_enable_dic=modal_enable_dic, ppg_enable_dic=ppg_enable_dic)
    if getattr(args, "glucose_aux_spo2", False):
        provider.Y_train = np.concatenate(
            [np.asarray(provider.Y_train, dtype=np.float32).reshape(len(provider.Y_train), -1), spo2_train],
            axis=1)
        provider.Y_test = np.concatenate(
            [np.asarray(provider.Y_test, dtype=np.float32).reshape(len(provider.Y_test), -1), spo2_test],
            axis=1)
        provider.BG_Min_Max = np.asarray([[3.0, 80.0], [30.0, 100.0]], dtype=np.float32)
        provider.target_name = "Glucose_SpO2"
        provider.target_unit = "mmol/L_%"
        provider.target_names = ["BG", "SpO2"]
        provider.target_units = ["mmol/L", "%"]
        provider.enable_aux_spo2 = True
        if not getattr(args, "glucose_keep_spo2_input", False):
            provider.X_train_list[2][:, 1] = 0.0
            provider.X_test_list[2][:, 1] = 0.0
    provider.Input_check()
    provider.Shape_check()
    print(
        f"Glucose dataset loaded: train={len(provider.Y_train)}, "
        f"test={len(provider.Y_test)}, split={provider.split_mode}, "
        f"subjects={len(provider.subject_counts)}")
    if getattr(provider, "split_stats", None):
        print(f"Glucose split stats: {provider.split_stats}")
    if getattr(provider, "glucose_subwindow_info", {}).get("enabled", False):
        print(f"Glucose subwindow info: {provider.glucose_subwindow_info}")

    dataset_dic = {
        "X_train": provider.X_train_list,
        "X_test": provider.X_test_list,
        "Y_train": provider.Y_train,
        "Y_test": provider.Y_test,
        "BG_Min_Max": provider.BG_Min_Max,
        "dataset_name": "glucose",
        "target_name": getattr(provider, "target_name", "BG"),
        "target_unit": getattr(provider, "target_unit", "mmol/L"),
        "target_names": getattr(provider, "target_names", ["BG"]),
        "target_units": getattr(provider, "target_units", ["mmol/L"]),
        "validation_group": getattr(provider, "validation_group", None),
        "validation_group_name": getattr(provider, "validation_group_name", None),
    }
    return provider, dataset_dic


def build_base_params(provider, args):
    dim = 1
    modal_enable_dic = getattr(provider, "modal_enable_dic", {})
    active_modalities = [
        int(modal_enable_dic.get(f"modal_{i}", 1))
        for i in range(1, 6)
    ]
    output_dim = int(np.asarray(provider.Y_train).reshape(len(provider.Y_train), -1).shape[1])
    dense_structure = [256, 128, 64, output_dim]
    target_names = getattr(provider, "target_names", [getattr(provider, "target_name", "BG")])
    target_units = getattr(provider, "target_units", [getattr(provider, "target_unit", "mmol/L")])
    params = {
        "detail": f"ablation_dim={dim}_{provider.detail_bin}",
        "dataset_name": getattr(provider, "dataset_name", "glucose"),
        "target_name": getattr(provider, "target_name", "BG"),
        "target_unit": getattr(provider, "target_unit", "mmol/L"),
        "target_names": target_names,
        "target_units": target_units,
        "primary_target_name": target_names[0],
        "output_dim": output_dim,
        "enable_error_grid": getattr(provider, "enable_error_grid", True),
        "active_modalities": active_modalities,
        "initial_lr": 0.001 if args.learning_rate is None else float(args.learning_rate),
        "_learning_rate_override": args.learning_rate,
        "decay_factor": 0.95,
        "step_size": 10,
        "step_size ": 10,
        "weight_decay": 0.0,
        "dropout_rate": 0.25,
        "n_split": args.n_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_seed": getattr(args, "training_seed", 2026),
        "deterministic_training": getattr(args, "deterministic_training", False),
        "cudnn_benchmark": getattr(args, "cudnn_benchmark", False),
        "EarlyStopping_patience": 50,
        "early_stopping_monitor": "total",
        "dense_structure": dense_structure,
        "enable_PEG_summary_list": [0, 0, 1],
        "enable_PEG_20_40": True,
        "model_name": "physics_spectral",
        "backbone_name": 0,
        "ppg_backbone_name": "resnet",
        "fusion_name": "attention",
        "filter_structure": [64, 96, 128, 160],
        "enable_loop": False,
        "enable_deconv": False if dim == 1 else True,
        "spectral_excel_path": args.spectral_excel,
        "spectral_step": args.spectral_step,
        "spectral_normalize": "sum",
        "legacy_channel_order": ["1200", "1300", "1460", "1550", "665", "905"],
        "use_abs": True,
        "spectral_hidden": 64,
        "spectral_fusion_mode": "concat",
        "spectral_gate_init": -4.0,
        "spectral_generator_mode": "pointwise",
        "spectral_basis_count": 16,
        "spectral_basis_dropout": 0.1,
        "enable_token_moe": True,
        "token_moe_mode": "anonymous",
        "role_moe_gate": True,
        "role_moe_gate_mode": "softmax",
        "role_moe_sigmoid_gate_init": 0.2,
        "enable_artifact_pretext": False,
        "artifact_prob": 0.5,
        "artifact_drift_scale": 0.10,
        "artifact_noise_scale": 0.03,
        "baseline_lowpass_kernel": 0,
        "noise_highpass_kernel": 0,
        "enable_observation_artifact_branches": True,
        "lambda_clean_recon": 0.0,
        "lambda_base_artifact": 0.0,
        "lambda_noise_artifact": 0.0,
        "lambda_background_smooth": 0.0,
        "lambda_target_smooth": 0.0,
        "lambda_basis_smooth": 0.0,
        "lambda_target_sparse": 0.0,
        "lambda_role_orth": 0.0,
        "enable_aux_spo2_spectrum": bool(output_dim >= 2),
        "enable_target_specific_regression": bool(output_dim >= 2),
        "enable_single_target_residual": False,
        "target_specific_regression_mode": "gated_residual",
        "target_spectral_gate_init": -3.0,
        "glucose_spectral_gate_max": 0.30,
        "spo2_spectral_gate_max": 0.15,
        "lambda_glucose_component": 0.0,
        "lambda_spo2_component": 0.0,
        "lambda_primary_component": 0.0,
        "target_loss_weights": getattr(
            provider,
            "target_loss_weights",
            [1.0, 0.2] if output_dim >= 2 else None),
        "enable_context_adversarial": False,
        "enable_background_context_conditioning": False,
        "background_context_hidden_dim": 64,
        "context_modalities": "th,demo,df",
        "context_grl_lambda": 1.0,
        "context_hidden_dim": 64,
        "lambda_bg_context": 0.0,
        "lambda_target_context_adv": 0.0,
        "num_experts": 4,
        "token_moe_dropout": 0.1,
        "lambda_obs": 0.1,
        "lambda_smooth": 0.01,
        "lambda_decorr": 0.001,
        "spectral_export_samples": args.export_samples,
        "spectral_export_batch_size": min(args.batch_size, max(args.export_samples, 1)),
    }
    if getattr(provider, "observation_matrix", None) is not None:
        params["observation_matrix"] = provider.observation_matrix
        params["wavelengths"] = provider.wavelengths
        params["use_abs"] = False
    return params


def adapt_external_single_target_params(params):
    """Instantiate the final role-aware plug-in for independent oxygen targets."""
    dataset_name = str(params.get("dataset_name", "")).lower()
    model_name = str(params.get("model_name", "")).lower()
    role_mode = str(params.get("token_moe_mode", "")).lower()
    regression_mode = str(
        params.get("target_specific_regression_mode", "")).lower()
    is_supported_external_task = (
        dataset_name in EXTERNAL_SINGLE_TARGET_DATASETS and
        int(params.get("output_dim", 1)) == 1)
    uses_final_role_plugin = (
        model_name in {"physics_spectral", "spectral_physics", "physics"} and
        role_mode in {"role_aware", "role", "physical"} and
        bool(params.get("enable_target_specific_regression", False)) and
        regression_mode == "baseline_residual_plugin")
    if not (is_supported_external_task and uses_final_role_plugin):
        return params

    primary_component_weight = float(
        params.get("lambda_primary_component",
                   params.get("lambda_glucose_component", 0.0)))
    if primary_component_weight <= 0:
        primary_component_weight = float(
            params.get("lambda_glucose_component", 0.0))
    params.update({
        "external_task_mode": "single_target_h_core",
        "primary_target_name": params.get(
            "target_names", [params.get("target_name", "target")])[0],
        "enable_aux_spo2_spectrum": False,
        "enable_single_target_residual": True,
        "target_loss_weights": None,
        "enable_background_context_conditioning": False,
        "enable_context_adversarial": False,
        "context_modalities": "",
        "lambda_bg_context": 0.0,
        "lambda_target_context_adv": 0.0,
        "lambda_primary_component": primary_component_weight,
        "lambda_glucose_component": 0.0,
        "lambda_spo2_component": 0.0,
    })
    return params


def make_params(base_params, key, spec):
    oxygen_datasets = ["oximetry"] + list(OPENOX_DATASET_TARGETS.keys())
    if key.startswith("OX") and base_params.get("dataset_name") not in oxygen_datasets:
        raise ValueError(f"{key} is only available with oxygen datasets: {', '.join(oxygen_datasets)}.")
    params = base_params.copy()
    params.update(spec["updates"])
    params = adapt_external_single_target_params(params)
    learning_rate_override = params.pop("_learning_rate_override", None)
    if learning_rate_override is not None:
        params["initial_lr"] = float(learning_rate_override)
    if "dense_structure" in params and params.get("output_dim", 1) > 1:
        dense_structure = list(params["dense_structure"])
        dense_structure[-1] = int(params["output_dim"])
        params["dense_structure"] = dense_structure
    params["detail"] = f"ablation_{spec['title']}_{base_params['detail']}"
    params["ablation_key"] = key
    params["ablation_description"] = spec["description"]
    return params


def print_dry_run(base_params, selected_keys):
    specs = get_ablation_specs()
    print("Ablation plan:")
    for key in selected_keys:
        params = make_params(base_params, key, specs[key])
        print(
            f"{key}: {specs[key]['title']} | "
            f"dataset={params.get('dataset_name', 'glucose')} | "
            f"target={params.get('target_name', 'BG')} | "
            f"targets={params.get('target_names', [params.get('target_name', 'BG')])} | "
            f"out_dim={params.get('output_dim', 1)} | "
            f"model={params['model_name']} | "
            f"obs={params['lambda_obs']} | "
            f"smooth={params['lambda_smooth']} | "
            f"ppg_backbone={params.get('ppg_backbone_name', 'resnet')} | "
            f"spec_fusion={params.get('spectral_fusion_mode', 'concat')} | "
            f"spec_gen={params.get('spectral_generator_mode', '-')} | "
            f"basis={params.get('spectral_basis_count', '-')} | "
            f"active_modalities={params.get('active_modalities', [])} | "
            f"lr={params.get('initial_lr')} | "
            f"dropout={params.get('dropout_rate')} | "
            f"wd={params.get('weight_decay', 0.0)} | "
            f"tsl_d={params.get('ppg_tsl_d_model', '-')} | "
            f"tsl_l={params.get('ppg_tsl_e_layers', '-')} | "
            f"bio_h={params.get('ppg_bio_hidden', '-')} | "
            f"bio_l={params.get('ppg_bio_layers', '-')} | "
            f"recent_h={params.get('ppg_recent_hidden', '-')} | "
            f"recent_l={params.get('ppg_recent_layers', '-')} | "
            f"recent_heads={params.get('ppg_recent_heads', '-')} | "
            f"moment_dff={params.get('ppg_moment_d_ff', '-')} | "
            f"moment_stride={params.get('ppg_recent_patch_stride', '-')} | "
            f"moment_reduction={params.get('ppg_moment_channel_reduction', '-')} | "
            f"csfm={params.get('ppg_csfm_variant', '-')} | "
            f"csfm_dim={params.get('ppg_csfm_hidden_dim', '-')} | "
            f"csfm_depth={params.get('ppg_csfm_depth', '-')} | "
            f"csfm_patch={params.get('ppg_csfm_patch_size', '-')} | "
            f"med_single={params.get('ppg_medformer_single_channel', '-')} | "
            f"med_stride={params.get('ppg_medformer_stride_sizes', '-')} | "
            f"med_inter={not params.get('ppg_medformer_no_inter_attn', False) if 'ppg_medformer_no_inter_attn' in params else '-'} | "
            f"tsla_asb={params.get('ppg_tslanet_asb', '-')} | "
            f"tsla_icb={params.get('ppg_tslanet_icb', '-')} | "
            f"papagei_blocks={params.get('ppg_papagei_blocks', '-')} | "
            f"papagei_mode={params.get('ppg_papagei_channel_mode', '-')} | "
            f"moe={params['enable_token_moe']} | "
            f"moe_mode={params.get('token_moe_mode', '-')} | "
            f"role_gate={params.get('role_moe_gate_mode', '-')} | "
            f"artifact={params.get('enable_artifact_pretext', False)} | "
            f"baseline_lp={params.get('baseline_lowpass_kernel', '-')} | "
            f"noise_hp={params.get('noise_highpass_kernel', '-')} | "
            f"obs_artifact_branches={params.get('enable_observation_artifact_branches', True)} | "
            f"context_adv={params.get('enable_context_adversarial', False)} | "
            f"bg_ctx_cond={params.get('enable_background_context_conditioning', False)} | "
            f"ctx={params.get('context_modalities', '-')} | "
            f"aux_spo2={params.get('enable_aux_spo2_spectrum', False)} | "
            f"target_heads={params.get('enable_target_specific_regression', False)} | "
            f"single_target_residual={params.get('enable_single_target_residual', False)} | "
            f"primary_target={params.get('primary_target_name', '-')} | "
            f"task_mode={params.get('external_task_mode', 'native')} | "
            f"target_head_mode={params.get('target_specific_regression_mode', '-')} | "
            f"target_gate_init={params.get('target_spectral_gate_init', '-')} | "
            f"target_gate_max={params.get('glucose_spectral_gate_max', '-') if params.get('target_specific_regression_mode') == 'bounded_gated_residual' else 'unbounded'}/"
            f"{params.get('spo2_spectral_gate_max', '-') if params.get('target_specific_regression_mode') == 'bounded_gated_residual' else 'unbounded'} | "
            f"compG={params.get('lambda_glucose_component', 0.0)} | "
            f"compSpO2={params.get('lambda_spo2_component', 0.0)} | "
            f"compPrimary={params.get('lambda_primary_component', 0.0)} | "
            f"role_orth={params.get('lambda_role_orth', 0.0)} | "
            f"target_sparse={params.get('lambda_target_sparse', 0.0)} | "
            f"smooth={params.get('lambda_smooth', 0.0)}/"
            f"{params.get('lambda_background_smooth', 0.0)}/"
            f"{params.get('lambda_target_smooth', 0.0)}/"
            f"{params.get('lambda_basis_smooth', 0.0)} | "
            f"decorr={params['lambda_decorr']} | "
            f"early_stop={params.get('early_stopping_monitor', 'total')} | "
            f"epochs={params['epochs']} | folds={params['n_split']} | "
            f"train_seed={params.get('training_seed')} | "
            f"deterministic={params.get('deterministic_training', False)} | "
            f"cudnn_benchmark={params.get('cudnn_benchmark', False)}")


def write_manifest_row(manifest_path, row, write_header=False):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time",
        "dataset_name",
        "target_name",
        "target_unit",
        "target_names",
        "target_units",
        "primary_target_name",
        "output_dim",
        "target_loss_weights",
        "external_task_mode",
        "key",
        "title",
        "description",
        "save_path",
        "active_modalities",
        "model_name",
        "ppg_backbone_name",
        "initial_lr",
        "filter_structure",
        "dense_structure",
        "dropout_rate",
        "weight_decay",
        "spectral_hidden",
        "spectral_generator_mode",
        "spectral_basis_count",
        "spectral_basis_dropout",
        "ppg_tsl_d_model",
        "ppg_tsl_d_ff",
        "ppg_tsl_e_layers",
        "ppg_tsl_d_layers",
        "ppg_tsl_n_heads",
        "ppg_patch_len",
        "ppg_patch_stride",
        "ppg_tsl_pred_len",
        "ppg_tsl_modes",
        "ppg_bio_hidden",
        "ppg_bio_layers",
        "ppg_bio_heads",
        "ppg_bio_kernel_size",
        "ppg_recent_hidden",
        "ppg_recent_layers",
        "ppg_recent_heads",
        "ppg_recent_patch_len",
        "ppg_recent_patch_stride",
        "ppg_recent_patch_sizes",
        "ppg_recent_kernel_size",
        "ppg_moment_d_ff",
        "ppg_moment_patch_dropout",
        "ppg_moment_revin_affine",
        "ppg_moment_add_positional_embedding",
        "ppg_moment_value_embedding_bias",
        "ppg_moment_orth_gain",
        "ppg_moment_mask_ratio",
        "ppg_moment_channel_reduction",
        "ppg_moment_pad_end",
        "ppg_csfm_variant",
        "ppg_csfm_signal_size",
        "ppg_csfm_patch_size",
        "ppg_csfm_channel_index",
        "ppg_csfm_hidden_dim",
        "ppg_csfm_depth",
        "ppg_csfm_heads",
        "ppg_csfm_mlp_dim",
        "ppg_csfm_dim_head",
        "ppg_csfm_weight_path",
        "ppg_csfm_freeze_encoder",
        "ppg_medformer_d_ff",
        "ppg_medformer_stride_sizes",
        "ppg_medformer_single_channel",
        "ppg_medformer_no_inter_attn",
        "ppg_medformer_activation",
        "ppg_tslanet_mlp_ratio",
        "ppg_tslanet_asb",
        "ppg_tslanet_icb",
        "ppg_tslanet_adaptive_filter",
        "ppg_papagei_base_filters",
        "ppg_papagei_kernel_size",
        "ppg_papagei_stride",
        "ppg_papagei_groups",
        "ppg_papagei_blocks",
        "ppg_papagei_embedding_dim",
        "ppg_papagei_experts",
        "ppg_papagei_variant",
        "ppg_papagei_channel_mode",
        "ppg_papagei_target_len",
        "ppg_papagei_weight_path",
        "ppg_papagei_freeze_encoder",
        "spectral_fusion_mode",
        "spectral_gate_init",
        "lambda_obs",
        "lambda_smooth",
        "enable_token_moe",
        "token_moe_mode",
        "role_moe_gate",
        "role_moe_gate_mode",
        "role_moe_sigmoid_gate_init",
        "enable_artifact_pretext",
        "artifact_prob",
        "artifact_drift_scale",
        "artifact_noise_scale",
        "baseline_lowpass_kernel",
        "noise_highpass_kernel",
        "enable_observation_artifact_branches",
        "lambda_clean_recon",
        "lambda_base_artifact",
        "lambda_noise_artifact",
        "lambda_background_smooth",
        "lambda_target_smooth",
        "lambda_basis_smooth",
        "lambda_target_sparse",
        "lambda_role_orth",
        "enable_background_context_conditioning",
        "background_context_hidden_dim",
        "enable_context_adversarial",
        "context_modalities",
        "context_grl_lambda",
        "context_hidden_dim",
        "lambda_bg_context",
        "lambda_target_context_adv",
        "enable_aux_spo2_spectrum",
        "enable_target_specific_regression",
        "enable_single_target_residual",
        "target_specific_regression_mode",
        "target_spectral_gate_init",
        "glucose_spectral_gate_max",
        "spo2_spectral_gate_max",
        "lambda_glucose_component",
        "lambda_spo2_component",
        "lambda_primary_component",
        "lambda_decorr",
        "early_stopping_monitor",
        "training_seed",
        "deterministic_training",
        "cudnn_benchmark",
        "epochs",
        "n_split",
    ]
    with manifest_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_ablation(args):
    if args.deterministic_training:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.gpu == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    selected_keys = resolve_requested_experiments(args.experiments)
    if args.dry_run:
        dry_target_name = "BG"
        dry_target_unit = "mmol/L"
        dry_target_names = ["BG"]
        dry_target_units = ["mmol/L"]
        dry_enable_error_grid = True
        dry_output_dim = 1
        if args.dataset == "oximetry":
            dry_target_name = "SpO2"
            dry_target_unit = "%"
            dry_target_names = ["SpO2"]
            dry_target_units = ["%"]
            dry_enable_error_grid = False
        elif args.dataset == "openox_sao2_spo2":
            dry_target_name = "SaO2_SpO2"
            dry_target_unit = "%"
            dry_target_names = ["SaO2", "SpO2"]
            dry_target_units = ["%", "%"]
            dry_enable_error_grid = False
            dry_output_dim = 2
        elif args.dataset in OPENOX_DRY_TARGETS:
            dry_target_name, dry_target_unit = OPENOX_DRY_TARGETS[args.dataset]
            dry_target_names = [dry_target_name]
            dry_target_units = [dry_target_unit]
            dry_enable_error_grid = False
        elif args.dataset == "glucose" and getattr(args, "glucose_aux_spo2", False):
            dry_target_name = "Glucose_SpO2"
            dry_target_unit = "mmol/L_%"
            dry_target_names = ["BG", "SpO2"]
            dry_target_units = ["mmol/L", "%"]
            dry_output_dim = 2
        provider = SimpleNamespace(
            detail_bin=f"Modal_{args.modal_mask}_ppg_{args.ppg_mask}_{args.dataset}",
            modal_enable_dic=parse_binary_mask(args.modal_mask, 5, "modal"),
            dataset_name=args.dataset,
            target_name=dry_target_name,
            target_unit=dry_target_unit,
            target_names=dry_target_names,
            target_units=dry_target_units,
            Y_train=np.zeros((1, dry_output_dim), dtype=np.float32),
            enable_error_grid=dry_enable_error_grid)
        base_params = build_base_params(provider, args)
        print_dry_run(base_params, selected_keys)
        return

    provider, dataset_dic = build_dataset(args)
    base_params = build_base_params(provider, args)

    from src.data_train_torch import Training_Process

    specs = get_ablation_specs()
    manifest_path = REPO_ROOT / "outputs" / f"ablation_manifest_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    for run_index, key in enumerate(selected_keys, start=1):
        spec = specs[key]
        params = make_params(base_params, key, spec)
        print(f"\n===== [{run_index}/{len(selected_keys)}] {key}: {spec['title']} =====")
        print(spec["description"])

        trainer = Training_Process(params, dataset_dic)
        trainer.train_start()

        write_manifest_row(
            manifest_path,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_name": params.get("dataset_name", ""),
                "target_name": params.get("target_name", ""),
                "target_unit": params.get("target_unit", ""),
                "target_names": params.get("target_names", ""),
                "target_units": params.get("target_units", ""),
                "primary_target_name": params.get("primary_target_name", ""),
                "output_dim": params.get("output_dim", ""),
                "target_loss_weights": params.get("target_loss_weights", ""),
                "external_task_mode": params.get("external_task_mode", "native"),
                "key": key,
                "title": spec["title"],
                "description": spec["description"],
                "save_path": trainer.save_path,
                "active_modalities": params.get("active_modalities", ""),
                "model_name": params["model_name"],
                "ppg_backbone_name": params.get("ppg_backbone_name", "resnet"),
                "initial_lr": params.get("initial_lr", ""),
                "filter_structure": params.get("filter_structure", ""),
                "dense_structure": params.get("dense_structure", ""),
                "dropout_rate": params.get("dropout_rate", ""),
                "weight_decay": params.get("weight_decay", ""),
                "spectral_hidden": params.get("spectral_hidden", ""),
                "spectral_generator_mode": params.get("spectral_generator_mode", ""),
                "spectral_basis_count": params.get("spectral_basis_count", ""),
                "spectral_basis_dropout": params.get("spectral_basis_dropout", ""),
                "ppg_tsl_d_model": params.get("ppg_tsl_d_model", ""),
                "ppg_tsl_d_ff": params.get("ppg_tsl_d_ff", ""),
                "ppg_tsl_e_layers": params.get("ppg_tsl_e_layers", ""),
                "ppg_tsl_d_layers": params.get("ppg_tsl_d_layers", ""),
                "ppg_tsl_n_heads": params.get("ppg_tsl_n_heads", ""),
                "ppg_patch_len": params.get("ppg_patch_len", ""),
                "ppg_patch_stride": params.get("ppg_patch_stride", ""),
                "ppg_tsl_pred_len": params.get("ppg_tsl_pred_len", ""),
                "ppg_tsl_modes": params.get("ppg_tsl_modes", ""),
                "ppg_bio_hidden": params.get("ppg_bio_hidden", ""),
                "ppg_bio_layers": params.get("ppg_bio_layers", ""),
                "ppg_bio_heads": params.get("ppg_bio_heads", ""),
                "ppg_bio_kernel_size": params.get("ppg_bio_kernel_size", ""),
                "ppg_recent_hidden": params.get("ppg_recent_hidden", ""),
                "ppg_recent_layers": params.get("ppg_recent_layers", ""),
                "ppg_recent_heads": params.get("ppg_recent_heads", ""),
                "ppg_recent_patch_len": params.get("ppg_recent_patch_len", ""),
                "ppg_recent_patch_stride": params.get("ppg_recent_patch_stride", ""),
                "ppg_recent_patch_sizes": params.get("ppg_recent_patch_sizes", ""),
                "ppg_recent_kernel_size": params.get("ppg_recent_kernel_size", ""),
                "ppg_moment_d_ff": params.get("ppg_moment_d_ff", ""),
                "ppg_moment_patch_dropout": params.get("ppg_moment_patch_dropout", ""),
                "ppg_moment_revin_affine": params.get("ppg_moment_revin_affine", ""),
                "ppg_moment_add_positional_embedding": params.get("ppg_moment_add_positional_embedding", ""),
                "ppg_moment_value_embedding_bias": params.get("ppg_moment_value_embedding_bias", ""),
                "ppg_moment_orth_gain": params.get("ppg_moment_orth_gain", ""),
                "ppg_moment_mask_ratio": params.get("ppg_moment_mask_ratio", ""),
                "ppg_moment_channel_reduction": params.get("ppg_moment_channel_reduction", ""),
                "ppg_moment_pad_end": params.get("ppg_moment_pad_end", ""),
                "ppg_csfm_variant": params.get("ppg_csfm_variant", ""),
                "ppg_csfm_signal_size": params.get("ppg_csfm_signal_size", ""),
                "ppg_csfm_patch_size": params.get("ppg_csfm_patch_size", ""),
                "ppg_csfm_channel_index": params.get("ppg_csfm_channel_index", ""),
                "ppg_csfm_hidden_dim": params.get("ppg_csfm_hidden_dim", ""),
                "ppg_csfm_depth": params.get("ppg_csfm_depth", ""),
                "ppg_csfm_heads": params.get("ppg_csfm_heads", ""),
                "ppg_csfm_mlp_dim": params.get("ppg_csfm_mlp_dim", ""),
                "ppg_csfm_dim_head": params.get("ppg_csfm_dim_head", ""),
                "ppg_csfm_weight_path": params.get("ppg_csfm_weight_path", ""),
                "ppg_csfm_freeze_encoder": params.get("ppg_csfm_freeze_encoder", ""),
                "ppg_medformer_d_ff": params.get("ppg_medformer_d_ff", ""),
                "ppg_medformer_stride_sizes": params.get("ppg_medformer_stride_sizes", ""),
                "ppg_medformer_single_channel": params.get("ppg_medformer_single_channel", ""),
                "ppg_medformer_no_inter_attn": params.get("ppg_medformer_no_inter_attn", ""),
                "ppg_medformer_activation": params.get("ppg_medformer_activation", ""),
                "ppg_tslanet_mlp_ratio": params.get("ppg_tslanet_mlp_ratio", ""),
                "ppg_tslanet_asb": params.get("ppg_tslanet_asb", ""),
                "ppg_tslanet_icb": params.get("ppg_tslanet_icb", ""),
                "ppg_tslanet_adaptive_filter": params.get("ppg_tslanet_adaptive_filter", ""),
                "ppg_papagei_base_filters": params.get("ppg_papagei_base_filters", ""),
                "ppg_papagei_kernel_size": params.get("ppg_papagei_kernel_size", ""),
                "ppg_papagei_stride": params.get("ppg_papagei_stride", ""),
                "ppg_papagei_groups": params.get("ppg_papagei_groups", ""),
                "ppg_papagei_blocks": params.get("ppg_papagei_blocks", ""),
                "ppg_papagei_embedding_dim": params.get("ppg_papagei_embedding_dim", ""),
                "ppg_papagei_experts": params.get("ppg_papagei_experts", ""),
                "ppg_papagei_variant": params.get("ppg_papagei_variant", ""),
                "ppg_papagei_channel_mode": params.get("ppg_papagei_channel_mode", ""),
                "ppg_papagei_target_len": params.get("ppg_papagei_target_len", ""),
                "ppg_papagei_weight_path": params.get("ppg_papagei_weight_path", ""),
                "ppg_papagei_freeze_encoder": params.get("ppg_papagei_freeze_encoder", ""),
                "spectral_fusion_mode": params.get("spectral_fusion_mode", "concat"),
                "spectral_gate_init": params.get("spectral_gate_init", ""),
                "lambda_obs": params["lambda_obs"],
                "lambda_smooth": params["lambda_smooth"],
                "enable_token_moe": params["enable_token_moe"],
                "token_moe_mode": params.get("token_moe_mode", ""),
                "role_moe_gate": params.get("role_moe_gate", ""),
                "role_moe_gate_mode": params.get("role_moe_gate_mode", ""),
                "role_moe_sigmoid_gate_init": params.get(
                    "role_moe_sigmoid_gate_init", ""),
                "enable_artifact_pretext": params.get("enable_artifact_pretext", ""),
                "artifact_prob": params.get("artifact_prob", ""),
                "artifact_drift_scale": params.get("artifact_drift_scale", ""),
                "artifact_noise_scale": params.get("artifact_noise_scale", ""),
                "baseline_lowpass_kernel": params.get("baseline_lowpass_kernel", ""),
                "noise_highpass_kernel": params.get("noise_highpass_kernel", ""),
                "enable_observation_artifact_branches": params.get(
                    "enable_observation_artifact_branches", ""),
                "lambda_clean_recon": params.get("lambda_clean_recon", ""),
                "lambda_base_artifact": params.get("lambda_base_artifact", ""),
                "lambda_noise_artifact": params.get("lambda_noise_artifact", ""),
                "lambda_background_smooth": params.get("lambda_background_smooth", ""),
                "lambda_target_smooth": params.get("lambda_target_smooth", ""),
                "lambda_basis_smooth": params.get("lambda_basis_smooth", ""),
                "lambda_target_sparse": params.get("lambda_target_sparse", ""),
                "lambda_role_orth": params.get("lambda_role_orth", ""),
                "enable_background_context_conditioning": params.get("enable_background_context_conditioning", ""),
                "background_context_hidden_dim": params.get("background_context_hidden_dim", ""),
                "enable_context_adversarial": params.get("enable_context_adversarial", ""),
                "context_modalities": params.get("context_modalities", ""),
                "context_grl_lambda": params.get("context_grl_lambda", ""),
                "context_hidden_dim": params.get("context_hidden_dim", ""),
                "lambda_bg_context": params.get("lambda_bg_context", ""),
                "lambda_target_context_adv": params.get("lambda_target_context_adv", ""),
                "enable_aux_spo2_spectrum": params.get("enable_aux_spo2_spectrum", ""),
                "enable_target_specific_regression": params.get(
                    "enable_target_specific_regression", ""),
                "enable_single_target_residual": params.get(
                    "enable_single_target_residual", ""),
                "target_specific_regression_mode": params.get(
                    "target_specific_regression_mode", ""),
                "target_spectral_gate_init": params.get(
                    "target_spectral_gate_init", ""),
                "glucose_spectral_gate_max": params.get(
                    "glucose_spectral_gate_max", ""),
                "spo2_spectral_gate_max": params.get(
                    "spo2_spectral_gate_max", ""),
                "lambda_glucose_component": params.get("lambda_glucose_component", ""),
                "lambda_spo2_component": params.get("lambda_spo2_component", ""),
                "lambda_primary_component": params.get("lambda_primary_component", ""),
                "lambda_decorr": params["lambda_decorr"],
                "early_stopping_monitor": params.get(
                    "early_stopping_monitor", "total"),
                "training_seed": params.get("training_seed", ""),
                "deterministic_training": params.get("deterministic_training", ""),
                "cudnn_benchmark": params.get("cudnn_benchmark", ""),
                "epochs": params["epochs"],
                "n_split": params["n_split"],
            },
            write_header=(run_index == 1),
        )

    print(f"\nAblation manifest saved to: {manifest_path}")


def main():
    args = build_arg_parser().parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
