import io
import copy
import os
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.models import MultiModalModel, PhysicsSpectralMultiModalModel

try:
    from torchinfo import summary
except ImportError:
    summary = None


class MultiInputDataset(Dataset):
    def __init__(self, x_OP, x_TH, x_DF, x_Demo, x_MI, labels):
        self.x_OP = x_OP
        self.x_TH = x_TH
        self.x_DF = x_DF
        self.x_Demo = x_Demo
        self.x_MI = x_MI
        self.labels = labels

    def __len__(self):
        return len(self.x_OP)  # 数据集的大小

    def __getitem__(self, idx):
        x_OP = self.x_OP[idx]
        x_TH = self.x_TH[idx]
        x_DF = self.x_DF[idx]
        x_Demo = self.x_Demo[idx]
        x_MI = self.x_MI[idx]
        label = self.labels[idx]
        
        return (x_OP, x_TH, x_DF, x_Demo, x_MI), label


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 当验证损失不再提升时，等待多少个 epoch 后停止训练
            verbose (bool): 是否打印详细的 early stopping 信息
            delta (float): 损失的最小变化，达到这个数值才认为有改进
            path (str): 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if self.verbose:
            #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存验证集损失下降的模型'''
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Traning_and_Evaluation():
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val 
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = []
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_summary = []
        pass

    def __array_to_tensor(self, X, Y):
        import numpy as np
        x_OP = torch.from_numpy(X[0].astype(np.float32))
        x_TH = torch.from_numpy(X[1].astype(np.float32))
        x_Demo = torch.from_numpy(X[2].astype(np.float32))
        x_DF = torch.from_numpy(X[3].astype(np.float32))
        x_MI = torch.from_numpy(X[4].astype(np.float32))
        labels = torch.from_numpy(Y.astype(np.float32))
        return x_OP, x_TH, x_DF, x_Demo, x_MI, labels

    def __is_physics_spectral_model(self, train_para):
        model_name = str(train_para.get('model_name', 0)).lower()
        return model_name in ['physics_spectral', 'spectral_physics', 'physics']

    def __make_artifact_corruption(self, batch_inputs, train_para):
        x_OP = batch_inputs[0]
        prob = float(train_para.get('artifact_prob', 0.5))
        drift_scale = float(train_para.get('artifact_drift_scale', 0.10))
        noise_scale = float(train_para.get('artifact_noise_scale', 0.03))
        zeros = torch.zeros_like(x_OP)
        if prob <= 0 or (drift_scale <= 0 and noise_scale <= 0):
            return batch_inputs, {
                "clean_ppg": x_OP,
                "baseline_target": zeros,
                "noise_target": zeros,
            }

        batch_size, channels, n_steps = x_OP.shape
        device = x_OP.device
        dtype = x_OP.dtype
        sample_mask = (torch.rand(batch_size, 1, 1, device=device, dtype=dtype) < prob).to(dtype)
        channel_std = x_OP.std(dim=-1, keepdim=True).clamp_min(1e-4)

        t = torch.linspace(0.0, 1.0, steps=n_steps, device=device, dtype=dtype)[None, None, :]
        cycles = torch.empty(batch_size, channels, 1, device=device, dtype=dtype).uniform_(0.2, 1.8)
        phase = torch.empty(batch_size, channels, 1, device=device, dtype=dtype).uniform_(0.0, 2.0 * np.pi)
        drift_amp = torch.empty(batch_size, channels, 1, device=device, dtype=dtype).uniform_(0.2, 1.0)
        drift_amp = drift_amp * channel_std * drift_scale
        drift = drift_amp * torch.sin(2.0 * np.pi * cycles * t + phase)
        slope = torch.empty(batch_size, channels, 1, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        drift = drift + slope * channel_std * drift_scale * (t - 0.5)

        raw_noise = torch.randn_like(x_OP) * channel_std * noise_scale
        kernel = int(train_para.get('artifact_noise_lowpass_kernel', 15))
        kernel = max(3, kernel + (1 - kernel % 2))
        if n_steps >= kernel:
            flat_noise = raw_noise.reshape(batch_size * channels, 1, n_steps)
            low_noise = F.avg_pool1d(flat_noise, kernel_size=kernel, stride=1, padding=kernel // 2)
            high_noise = (flat_noise - low_noise).reshape_as(raw_noise)
        else:
            high_noise = raw_noise

        baseline_target = drift * sample_mask
        noise_target = high_noise * sample_mask
        corrupted_inputs = list(batch_inputs)
        corrupted_inputs[0] = x_OP + baseline_target + noise_target
        return corrupted_inputs, {
            "clean_ppg": x_OP,
            "baseline_target": baseline_target,
            "noise_target": noise_target,
        }

    def __build_observation_matrix(self, train_para):
        H, _ = self.__build_observation_data(train_para)
        return H

    def __build_observation_data(self, train_para):
        observation_matrix = train_para.get('observation_matrix', None)
        if observation_matrix is not None:
            H = np.asarray(observation_matrix, dtype=np.float32)
            wavelengths = train_para.get('wavelengths', None)
            if wavelengths is None:
                wavelengths = np.arange(H.shape[1], dtype=np.float32)
            return H, np.asarray(wavelengths, dtype=np.float32)

        spectral_excel_path = train_para.get('spectral_excel_path', 'Spectral Distribution Curves.xlsx')
        if not os.path.exists(spectral_excel_path):
            raise FileNotFoundError(
                f"spectral_excel_path does not exist: {spectral_excel_path}. "
                "Set parameters_dic['spectral_excel_path'] or pass observation_matrix directly.")

        from src.data_process_workingcopy import build_observation_matrix
        H, wavelengths = build_observation_matrix(
            spectral_excel_path,
            spectral_min=train_para.get('spectral_min', None),
            spectral_max=train_para.get('spectral_max', None),
            spectral_step=train_para.get('spectral_step', 5.0),
            normalize=train_para.get('spectral_normalize', 'sum'),
            legacy_channel_order=train_para.get('legacy_channel_order', None),
            use_abs=train_para.get('use_abs', True))
        return H.astype(np.float32), wavelengths.astype(np.float32)

    def __create_model(self, train_para, x_OP_shape=None):
        if x_OP_shape is None and getattr(self, 'input_shapes', None):
            x_OP_shape = self.input_shapes[0]
        filter_structure = train_para.get('filter_structure', [64,96,128,160])
        dense_structure = list(train_para.get('dense_structure', [256,128,64,1]))
        output_dim = int(train_para.get('output_dim', dense_structure[-1]))
        if dense_structure[-1] != output_dim:
            dense_structure[-1] = output_dim
        dropout_rate = train_para.get('dropout_rate', 0.25)
        enable_deconv = train_para.get('enable_deconv', True)
        fusion_name = train_para.get('fusion_name', 'attention')
        ppg_backbone_name = train_para.get('ppg_backbone_name', train_para.get('backbone_name', 'resnet'))
        if str(ppg_backbone_name).lower() in ['0', 'none']:
            ppg_backbone_name = 'resnet'
        ppg_seq_len = train_para.get('ppg_seq_len', x_OP_shape[-1] if x_OP_shape is not None else 300)

        if self.__is_physics_spectral_model(train_para):
            H = self.__build_observation_matrix(train_para)
            if x_OP_shape is not None and H.shape[0] != x_OP_shape[0]:
                raise ValueError(f"Observation matrix channels {H.shape[0]} do not match PPG channels {x_OP_shape[0]}")
            return PhysicsSpectralMultiModalModel(
                filter_structure,
                dense_structure,
                dropout_rate,
                observation_matrix=H,
                fusion_name=fusion_name,
                enable_deconv=enable_deconv,
                spectral_hidden=train_para.get('spectral_hidden', 64),
                enable_token_moe=train_para.get('enable_token_moe', False),
                num_experts=train_para.get('num_experts', 4),
                token_moe_dropout=train_para.get('token_moe_dropout', 0.1),
                spectral_fusion_mode=train_para.get('spectral_fusion_mode', 'concat'),
                spectral_gate_init=train_para.get('spectral_gate_init', -4.0),
                spectral_generator_mode=train_para.get('spectral_generator_mode', 'pointwise'),
                spectral_basis_count=train_para.get('spectral_basis_count', 16),
                spectral_basis_dropout=train_para.get('spectral_basis_dropout', dropout_rate),
                token_moe_mode=train_para.get('token_moe_mode', 'anonymous'),
                role_moe_gate=train_para.get('role_moe_gate', True),
                role_moe_gate_mode=train_para.get('role_moe_gate_mode', 'softmax'),
                role_moe_sigmoid_gate_init=train_para.get(
                    'role_moe_sigmoid_gate_init', 0.2),
                ppg_backbone_name=ppg_backbone_name,
                ppg_backbone_config=train_para,
                ppg_seq_len=ppg_seq_len,
                active_modalities=train_para.get('active_modalities', None),
                enable_context_adversarial=train_para.get('enable_context_adversarial', False),
                context_modalities=train_para.get('context_modalities', None),
                context_grl_lambda=train_para.get('context_grl_lambda', 1.0),
                context_hidden_dim=train_para.get('context_hidden_dim', 64),
                baseline_lowpass_kernel=train_para.get('baseline_lowpass_kernel', 0),
                noise_highpass_kernel=train_para.get('noise_highpass_kernel', 0),
                enable_observation_artifact_branches=train_para.get(
                    'enable_observation_artifact_branches', True),
                output_dim=output_dim,
                enable_aux_spo2_spectrum=train_para.get('enable_aux_spo2_spectrum', False),
                enable_target_specific_regression=train_para.get(
                    'enable_target_specific_regression', True),
                enable_single_target_residual=train_para.get(
                    'enable_single_target_residual', False),
                primary_target_name=train_para.get(
                    'primary_target_name',
                    train_para.get('target_name', 'target')),
                target_specific_regression_mode=train_para.get(
                    'target_specific_regression_mode', 'gated_residual'),
                target_spectral_gate_init=train_para.get(
                    'target_spectral_gate_init', -3.0),
                glucose_spectral_gate_max=train_para.get(
                    'glucose_spectral_gate_max', 0.30),
                spo2_spectral_gate_max=train_para.get(
                    'spo2_spectral_gate_max', 0.15),
                enable_background_context_conditioning=train_para.get('enable_background_context_conditioning', False),
                background_context_hidden_dim=train_para.get(
                    'background_context_hidden_dim',
                    train_para.get('context_hidden_dim', 64)))

        return MultiModalModel(
            filter_structure,
            dense_structure,
            dropout_rate,
            fusion_name=fusion_name,
            enable_deconv=enable_deconv,
            ppg_backbone_name=ppg_backbone_name,
            ppg_backbone_config=train_para,
            ppg_seq_len=ppg_seq_len,
            ppg_input_channels=x_OP_shape[0] if x_OP_shape is not None else 6,
            active_modalities=train_para.get('active_modalities', None))

    def __get_bg_output(self, outputs):
        return outputs["bg"] if isinstance(outputs, dict) else outputs

    def __decorrelation_loss(self, tokens):
        if tokens is None:
            return None
        batch_size, num_tokens = tokens.shape[:2]
        if num_tokens <= 1:
            return torch.zeros((), device=tokens.device)

        token_vectors = tokens.flatten(start_dim=2)
        token_vectors = token_vectors - token_vectors.mean(dim=2, keepdim=True)
        token_vectors = F.normalize(token_vectors, p=2, dim=2, eps=1e-8)
        corr = torch.bmm(token_vectors, token_vectors.transpose(1, 2))
        eye = torch.eye(num_tokens, device=tokens.device).unsqueeze(0)
        off_diag = corr * (1.0 - eye)
        return torch.sum(off_diag ** 2) / (batch_size * num_tokens * (num_tokens - 1))

    def __spectral_absolute_smooth_loss(self, spectrum, wavelength_dim=1):
        """Absolute first-difference metric retained for diagnostics only."""
        if spectrum is None:
            return torch.zeros((), device=self.__device)
        spectral_values = torch.movedim(spectrum, wavelength_dim, -1)
        if spectral_values.size(-1) < 2:
            return torch.zeros((), device=spectrum.device, dtype=spectrum.dtype)
        return torch.mean(
            (spectral_values[..., 1:] - spectral_values[..., :-1]) ** 2)

    def __spectral_smooth_loss(self, spectrum, wavelength_dim=1):
        """Amplitude-invariant curvature penalty along the wavelength axis."""
        if spectrum is None:
            return torch.zeros((), device=self.__device)
        spectral_values = torch.movedim(spectrum, wavelength_dim, -1)
        if spectral_values.size(-1) < 3:
            return torch.zeros((), device=spectrum.device, dtype=spectrum.dtype)
        spectral_rms = torch.sqrt(
            torch.mean(spectral_values ** 2, dim=-1, keepdim=True) + 1e-8)
        normalized = spectral_values / spectral_rms
        curvature = (
            normalized[..., 2:] -
            2.0 * normalized[..., 1:-1] +
            normalized[..., :-2])
        return torch.mean(curvature ** 2)

    def __spectral_rms(self, spectrum):
        if spectrum is None:
            return torch.zeros((), device=self.__device)
        return torch.sqrt(torch.mean(spectrum ** 2) + 1e-8)

    def __orthogonal_loss(self, left, right):
        if left is None or right is None:
            return torch.zeros((), device=self.__device)
        left_vec = left.flatten(start_dim=1)
        right_vec = right.flatten(start_dim=1)
        left_vec = F.normalize(left_vec - left_vec.mean(dim=1, keepdim=True), dim=1, eps=1e-8)
        right_vec = F.normalize(right_vec - right_vec.mean(dim=1, keepdim=True), dim=1, eps=1e-8)
        return torch.mean(torch.sum(left_vec * right_vec, dim=1) ** 2)

    def __physics_loss(self, outputs, batch_inputs, batch_labels, criterion, train_para, artifact_targets=None):
        bg_output = self.__get_bg_output(outputs)
        target_weights = train_para.get('target_loss_weights', None)
        if (
                target_weights is not None and
                bg_output.ndim == 2 and
                batch_labels.ndim == 2 and
                bg_output.shape == batch_labels.shape):
            weights = torch.as_tensor(target_weights, dtype=bg_output.dtype, device=bg_output.device)
            weights = weights[:bg_output.size(1)].view(1, -1)
            loss_bg = torch.mean((bg_output - batch_labels) ** 2 * weights)
        else:
            loss_bg = criterion(bg_output, batch_labels)
        total_loss = loss_bg

        loss_obs = torch.zeros((), device=batch_labels.device)
        loss_smooth = torch.zeros((), device=batch_labels.device)
        loss_smooth_abs = torch.zeros((), device=batch_labels.device)
        spectrum_rms = torch.zeros((), device=batch_labels.device)
        loss_background_smooth = torch.zeros((), device=batch_labels.device)
        loss_target_smooth = torch.zeros((), device=batch_labels.device)
        loss_basis_smooth = torch.zeros((), device=batch_labels.device)
        loss_decorr = torch.zeros((), device=batch_labels.device)
        loss_bg_context = torch.zeros((), device=batch_labels.device)
        loss_target_context = torch.zeros((), device=batch_labels.device)
        loss_glucose_component = torch.zeros((), device=batch_labels.device)
        loss_spo2_component = torch.zeros((), device=batch_labels.device)
        weighted_obs = torch.zeros((), device=batch_labels.device)
        weighted_smooth = torch.zeros((), device=batch_labels.device)
        weighted_background_smooth = torch.zeros((), device=batch_labels.device)
        weighted_target_smooth = torch.zeros((), device=batch_labels.device)
        weighted_basis_smooth = torch.zeros((), device=batch_labels.device)
        weighted_decorr = torch.zeros((), device=batch_labels.device)
        weighted_bg_context = torch.zeros((), device=batch_labels.device)
        weighted_target_context = torch.zeros((), device=batch_labels.device)
        weighted_glucose_component = torch.zeros((), device=batch_labels.device)
        weighted_spo2_component = torch.zeros((), device=batch_labels.device)

        if isinstance(outputs, dict):
            lambda_obs = train_para.get('lambda_obs', 0.0)
            lambda_smooth = train_para.get('lambda_smooth', 0.0)
            lambda_decorr = train_para.get('lambda_decorr', 0.0)

            if lambda_obs > 0:
                loss_obs = criterion(outputs["y_hat"], batch_inputs[0])
                weighted_obs = lambda_obs * loss_obs
                total_loss = total_loss + weighted_obs

            spectrum = outputs.get("spectrum")
            if spectrum is not None:
                smooth_source = spectrum if lambda_smooth > 0 else spectrum.detach()
                loss_smooth = self.__spectral_smooth_loss(smooth_source)
                loss_smooth_abs = self.__spectral_absolute_smooth_loss(
                    spectrum.detach())
                spectrum_rms = self.__spectral_rms(spectrum.detach())
            if lambda_smooth > 0:
                weighted_smooth = lambda_smooth * loss_smooth
                total_loss = total_loss + weighted_smooth

            if lambda_decorr > 0:
                token_loss = self.__decorrelation_loss(outputs.get("tokens"))
                if token_loss is not None:
                    loss_decorr = token_loss
                    weighted_decorr = lambda_decorr * loss_decorr
                    total_loss = total_loss + weighted_decorr

            if artifact_targets is not None:
                lambda_clean = train_para.get('lambda_clean_recon', 0.0)
                lambda_base = train_para.get('lambda_base_artifact', 0.0)
                lambda_noise = train_para.get('lambda_noise_artifact', 0.0)

                if lambda_clean > 0 and outputs.get("clean_y_hat") is not None:
                    total_loss = total_loss + lambda_clean * criterion(outputs["clean_y_hat"], artifact_targets["clean_ppg"])
                if lambda_base > 0 and outputs.get("baseline_hat") is not None:
                    total_loss = total_loss + lambda_base * criterion(outputs["baseline_hat"], artifact_targets["baseline_target"])
                if lambda_noise > 0 and outputs.get("noise_hat") is not None:
                    total_loss = total_loss + lambda_noise * criterion(outputs["noise_hat"], artifact_targets["noise_target"])

            # Spectral structure constraints are independent of artifact pretext.
            lambda_bg_smooth = train_para.get('lambda_background_smooth', 0.0)
            lambda_target_smooth = train_para.get('lambda_target_smooth', 0.0)
            lambda_basis_smooth = train_para.get('lambda_basis_smooth', 0.0)
            lambda_target_sparse = train_para.get('lambda_target_sparse', 0.0)
            lambda_role_orth = train_para.get('lambda_role_orth', 0.0)

            background_spectrum = outputs.get("background_spectrum")
            if background_spectrum is not None:
                background_source = (
                    background_spectrum if lambda_bg_smooth > 0
                    else background_spectrum.detach())
                loss_background_smooth = self.__spectral_smooth_loss(
                    background_source)
            if lambda_bg_smooth > 0:
                weighted_background_smooth = (
                    lambda_bg_smooth * loss_background_smooth)
                total_loss = total_loss + weighted_background_smooth

            target_smooth_terms = []
            for target_key in (
                    "target_glucose_spectrum", "target_spo2_spectrum"):
                target_component = outputs.get(target_key)
                if target_component is not None:
                    target_source = (
                        target_component if lambda_target_smooth > 0
                        else target_component.detach())
                    target_smooth_terms.append(
                        self.__spectral_smooth_loss(target_source))
            if target_smooth_terms:
                loss_target_smooth = torch.stack(target_smooth_terms).mean()
            if lambda_target_smooth > 0:
                weighted_target_smooth = lambda_target_smooth * loss_target_smooth
                total_loss = total_loss + weighted_target_smooth

            spectral_bases = outputs.get("spectral_bases")
            if spectral_bases is not None:
                basis_source = (
                    spectral_bases if lambda_basis_smooth > 0
                    else spectral_bases.detach())
                loss_basis_smooth = self.__spectral_smooth_loss(
                    basis_source, wavelength_dim=-1)
            if lambda_basis_smooth > 0:
                weighted_basis_smooth = lambda_basis_smooth * loss_basis_smooth
                total_loss = total_loss + weighted_basis_smooth

            weighted_smooth = (
                weighted_smooth +
                weighted_background_smooth +
                weighted_target_smooth +
                weighted_basis_smooth)

            if lambda_target_sparse > 0 and outputs.get("target_spectrum") is not None:
                total_loss = total_loss + lambda_target_sparse * torch.mean(torch.abs(outputs["target_spectrum"]))
            if lambda_role_orth > 0:
                total_loss = total_loss + lambda_role_orth * self.__orthogonal_loss(
                    outputs.get("background_spectrum"),
                    outputs.get("target_spectrum"))

            lambda_bg_context = train_para.get('lambda_bg_context', 0.0)
            lambda_target_context = train_para.get('lambda_target_context_adv', 0.0)
            context_target = outputs.get("context_target")
            if context_target is not None:
                if lambda_bg_context > 0 and outputs.get("background_context_pred") is not None:
                    loss_bg_context = F.mse_loss(outputs["background_context_pred"], context_target)
                    weighted_bg_context = lambda_bg_context * loss_bg_context
                    total_loss = total_loss + weighted_bg_context
                if lambda_target_context > 0 and outputs.get("target_context_pred") is not None:
                    loss_target_context = F.mse_loss(outputs["target_context_pred"], context_target)
                    weighted_target_context = lambda_target_context * loss_target_context
                    total_loss = total_loss + weighted_target_context

            labels_2d = batch_labels.reshape(batch_labels.size(0), -1)
            lambda_primary_component = train_para.get(
                'lambda_primary_component', 0.0)
            lambda_glucose_component = train_para.get('lambda_glucose_component', 0.0)
            lambda_spo2_component = train_para.get('lambda_spo2_component', 0.0)
            if (
                    labels_2d.size(1) >= 1 and
                    lambda_primary_component > 0 and
                    outputs.get("primary_target_component_pred") is not None):
                loss_glucose_component = F.mse_loss(
                    outputs["primary_target_component_pred"],
                    labels_2d[:, 0:1])
                weighted_glucose_component = (
                    lambda_primary_component * loss_glucose_component)
                total_loss = total_loss + weighted_glucose_component
            elif labels_2d.size(1) >= 1 and lambda_glucose_component > 0 and outputs.get("glucose_component_pred") is not None:
                loss_glucose_component = F.mse_loss(outputs["glucose_component_pred"], labels_2d[:, 0:1])
                weighted_glucose_component = lambda_glucose_component * loss_glucose_component
                total_loss = total_loss + weighted_glucose_component
            if labels_2d.size(1) >= 2 and lambda_spo2_component > 0 and outputs.get("spo2_component_pred") is not None:
                loss_spo2_component = F.mse_loss(outputs["spo2_component_pred"], labels_2d[:, 1:2])
                weighted_spo2_component = lambda_spo2_component * loss_spo2_component
                total_loss = total_loss + weighted_spo2_component

        return total_loss, {
            "bg": loss_bg.detach(),
            "obs": weighted_obs.detach(),
            "smooth": weighted_smooth.detach(),
            "background_smooth": weighted_background_smooth.detach(),
            "target_smooth": weighted_target_smooth.detach(),
            "basis_smooth": weighted_basis_smooth.detach(),
            "decorr": weighted_decorr.detach(),
            "bg_context": weighted_bg_context.detach(),
            "target_context": weighted_target_context.detach(),
            "glucose_component": weighted_glucose_component.detach(),
            "spo2_component": weighted_spo2_component.detach(),
            "obs_raw": loss_obs.detach(),
            "smooth_raw": loss_smooth.detach(),
            "smooth_abs": loss_smooth_abs.detach(),
            "spectrum_rms": spectrum_rms.detach(),
            "background_smooth_raw": loss_background_smooth.detach(),
            "target_smooth_raw": loss_target_smooth.detach(),
            "basis_smooth_raw": loss_basis_smooth.detach(),
            "decorr_raw": loss_decorr.detach(),
            "bg_context_raw": loss_bg_context.detach(),
            "target_context_raw": loss_target_context.detach(),
            "glucose_component_raw": loss_glucose_component.detach(),
            "spo2_component_raw": loss_spo2_component.detach()
        }

    def Train(self, train_para={}):
        self.train_para = train_para.copy()
        training_seed = int(train_para.get(
            'round_seed', train_para.get('training_seed', 2026)))
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        deterministic_training = bool(train_para.get('deterministic_training', False))
        cudnn_benchmark = bool(train_para.get('cudnn_benchmark', False))
        torch.backends.cudnn.deterministic = deterministic_training
        torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic_training
        torch.use_deterministic_algorithms(deterministic_training, warn_only=True)
        print(
            f"CUDA training settings: deterministic={deterministic_training}, "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark}")

        x_OP_train, x_TH_train, x_DF_train, x_Demo_train, x_MI_train, labels_train = self.__array_to_tensor(self.X_train,self.Y_train)
        x_OP_val, x_TH_val, x_DF_val, x_Demo_val, x_MI_val, labels_val = self.__array_to_tensor(self.X_val, self.Y_val)

        self.input_shapes = [
                            tuple(x_OP_train.shape[1:]),  # x_OP_train 的形状，除去 batch_size
                            tuple(x_TH_train.shape[1:]),  # x_TH_train 的形状，除去 cbath_size
                            tuple(x_DF_train.shape[1:]),  # x_DF_train 的形状，除去 batch_size
                            tuple(x_Demo_train.shape[1:]), # x_Demo_train 的形状，除去 batch_size
                            tuple(x_MI_train.shape[1:])    # x_MI_train 的形状，除去 batch_size
                            ]
        
        # 模型参数
        filter_structure = train_para.get('filter_structure', [64,96,128,160])
        dense_structure = train_para.get('dense_structure', [256,128,64,1])
        dropout_rate = train_para.get('dropout_rate', 0.25)
        initial_lr = train_para.get('initial_lr', 0.001)
        EarlyStopping_patience = train_para.get('EarlyStopping_patience', 200)
        early_stopping_monitor = str(
            train_para.get('early_stopping_monitor', 'total')).lower()
        if early_stopping_monitor not in ['total', 'bg']:
            raise ValueError("early_stopping_monitor must be 'total' or 'bg'")
        decay_factor = train_para.get('decay_factor', 0.99)
        step_size = train_para.get('step_size', 10)
        weight_decay = train_para.get('weight_decay', 0.0)
        num_epochs = train_para.get('epochs', 10)
        batch_size = train_para.get('batch_size', 10)
        enable_deconv = train_para.get('enable_deconv', True)
        fusion_name = train_para.get('fusion_name', 'attention')

        # 创建数据集和数据加载器
        train_dataset = MultiInputDataset(x_OP_train, x_TH_train, x_DF_train, x_Demo_train, x_MI_train, labels_train)
        train_generator = torch.Generator()
        train_generator.manual_seed(training_seed)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator)

        val_dataset = MultiInputDataset(x_OP_val, x_TH_val, x_DF_val, x_Demo_val, x_MI_val, labels_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型、优化器和损失函数
        model = self.__create_model(train_para, x_OP_shape=tuple(x_OP_train.shape[1:])).to(self.__device)

        # 检查是否有多个 GPU
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = nn.DataParallel(model)
        
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # 使用 StepLR 来实现学习率衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_factor)

        # 初始化 EarlyStopping
        early_stopping = EarlyStopping(patience=EarlyStopping_patience, verbose=True, path='best_model.pt')

        # 存储训练和验证损失
        self.train_losses = []
        self.val_losses = []
        self.train_bg_losses = []
        self.train_obs_losses = []
        self.train_smooth_losses = []
        self.train_decorr_losses = []
        self.train_bg_context_losses = []
        self.train_target_context_losses = []
        self.train_glucose_component_losses = []
        self.train_spo2_component_losses = []
        self.train_obs_raw_losses = []
        self.train_smooth_raw_losses = []
        self.train_smooth_abs_losses = []
        self.train_spectrum_rms_values = []
        self.train_background_smooth_losses = []
        self.train_target_smooth_losses = []
        self.train_basis_smooth_losses = []
        self.train_decorr_raw_losses = []
        self.train_bg_context_raw_losses = []
        self.train_target_context_raw_losses = []
        self.train_glucose_component_raw_losses = []
        self.train_spo2_component_raw_losses = []
        self.val_bg_losses = []
        self.val_obs_losses = []
        self.val_smooth_losses = []
        self.val_decorr_losses = []
        self.val_bg_context_losses = []
        self.val_target_context_losses = []
        self.val_glucose_component_losses = []
        self.val_spo2_component_losses = []
        self.val_obs_raw_losses = []
        self.val_smooth_raw_losses = []
        self.val_smooth_abs_losses = []
        self.val_spectrum_rms_values = []
        self.val_background_smooth_losses = []
        self.val_target_smooth_losses = []
        self.val_basis_smooth_losses = []
        self.val_decorr_raw_losses = []
        self.val_bg_context_raw_losses = []
        self.val_target_context_raw_losses = []
        self.val_glucose_component_raw_losses = []
        self.val_spo2_component_raw_losses = []
        self.lrs = []  # 存储学习率
    
        # 训练循环
        best_state_dict = None
        for epoch in range(num_epochs):
            start_time = time.time()  # 记录开始时间
            model.train()
            running_loss = 0.0
            running_bg_loss = 0.0
            running_obs_loss = 0.0
            running_smooth_loss = 0.0
            running_decorr_loss = 0.0
            running_bg_context_loss = 0.0
            running_target_context_loss = 0.0
            running_glucose_component_loss = 0.0
            running_spo2_component_loss = 0.0
            running_obs_raw_loss = 0.0
            running_smooth_raw_loss = 0.0
            running_smooth_abs_loss = 0.0
            running_spectrum_rms = 0.0
            running_background_smooth_loss = 0.0
            running_target_smooth_loss = 0.0
            running_basis_smooth_loss = 0.0
            running_decorr_raw_loss = 0.0
            running_bg_context_raw_loss = 0.0
            running_target_context_raw_loss = 0.0
            running_glucose_component_raw_loss = 0.0
            running_spo2_component_raw_loss = 0.0
            for batch_idx, (batch_inputs, batch_labels) in enumerate(train_dataloader):
                # 将输入和标签移动到 GPU 上
                batch_inputs = [i.to(self.__device) for i in batch_inputs]
                batch_labels = batch_labels.to(self.__device)
                artifact_targets = None
                model_inputs = batch_inputs
                if (
                        self.__is_physics_spectral_model(train_para) and
                        bool(train_para.get('enable_artifact_pretext', False))):
                    model_inputs, artifact_targets = self.__make_artifact_corruption(batch_inputs, train_para)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(*model_inputs)
                loss, loss_items = self.__physics_loss(
                    outputs,
                    model_inputs,
                    batch_labels,
                    criterion,
                    train_para,
                    artifact_targets=artifact_targets)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_bg_loss += loss_items["bg"].item()
                running_obs_loss += loss_items["obs"].item()
                running_smooth_loss += loss_items["smooth"].item()
                running_decorr_loss += loss_items["decorr"].item()
                running_bg_context_loss += loss_items["bg_context"].item()
                running_target_context_loss += loss_items["target_context"].item()
                running_glucose_component_loss += loss_items["glucose_component"].item()
                running_spo2_component_loss += loss_items["spo2_component"].item()
                running_obs_raw_loss += loss_items["obs_raw"].item()
                running_smooth_raw_loss += loss_items["smooth_raw"].item()
                running_smooth_abs_loss += loss_items["smooth_abs"].item()
                running_spectrum_rms += loss_items["spectrum_rms"].item()
                running_background_smooth_loss += loss_items["background_smooth"].item()
                running_target_smooth_loss += loss_items["target_smooth"].item()
                running_basis_smooth_loss += loss_items["basis_smooth"].item()
                running_decorr_raw_loss += loss_items["decorr_raw"].item()
                running_bg_context_raw_loss += loss_items["bg_context_raw"].item()
                running_target_context_raw_loss += loss_items["target_context_raw"].item()
                running_glucose_component_raw_loss += loss_items["glucose_component_raw"].item()
                running_spo2_component_raw_loss += loss_items["spo2_component_raw"].item()

            # 计算当前 epoch 的平均训练损失
            avg_train_loss = running_loss / len(train_dataloader)
            avg_train_bg_loss = running_bg_loss / len(train_dataloader)
            avg_train_obs_loss = running_obs_loss / len(train_dataloader)
            avg_train_smooth_loss = running_smooth_loss / len(train_dataloader)
            avg_train_decorr_loss = running_decorr_loss / len(train_dataloader)
            avg_train_bg_context_loss = running_bg_context_loss / len(train_dataloader)
            avg_train_target_context_loss = running_target_context_loss / len(train_dataloader)
            avg_train_glucose_component_loss = running_glucose_component_loss / len(train_dataloader)
            avg_train_spo2_component_loss = running_spo2_component_loss / len(train_dataloader)
            avg_train_obs_raw_loss = running_obs_raw_loss / len(train_dataloader)
            avg_train_smooth_raw_loss = running_smooth_raw_loss / len(train_dataloader)
            avg_train_smooth_abs_loss = running_smooth_abs_loss / len(train_dataloader)
            avg_train_spectrum_rms = running_spectrum_rms / len(train_dataloader)
            avg_train_background_smooth_loss = running_background_smooth_loss / len(train_dataloader)
            avg_train_target_smooth_loss = running_target_smooth_loss / len(train_dataloader)
            avg_train_basis_smooth_loss = running_basis_smooth_loss / len(train_dataloader)
            avg_train_decorr_raw_loss = running_decorr_raw_loss / len(train_dataloader)
            avg_train_bg_context_raw_loss = running_bg_context_raw_loss / len(train_dataloader)
            avg_train_target_context_raw_loss = running_target_context_raw_loss / len(train_dataloader)
            avg_train_glucose_component_raw_loss = running_glucose_component_raw_loss / len(train_dataloader)
            avg_train_spo2_component_raw_loss = running_spo2_component_raw_loss / len(train_dataloader)
            self.train_losses.append(avg_train_loss)  # 记录训练损失
            self.train_bg_losses.append(avg_train_bg_loss)
            self.train_obs_losses.append(avg_train_obs_loss)
            self.train_smooth_losses.append(avg_train_smooth_loss)
            self.train_decorr_losses.append(avg_train_decorr_loss)
            self.train_bg_context_losses.append(avg_train_bg_context_loss)
            self.train_target_context_losses.append(avg_train_target_context_loss)
            self.train_glucose_component_losses.append(avg_train_glucose_component_loss)
            self.train_spo2_component_losses.append(avg_train_spo2_component_loss)
            self.train_obs_raw_losses.append(avg_train_obs_raw_loss)
            self.train_smooth_raw_losses.append(avg_train_smooth_raw_loss)
            self.train_smooth_abs_losses.append(avg_train_smooth_abs_loss)
            self.train_spectrum_rms_values.append(avg_train_spectrum_rms)
            self.train_background_smooth_losses.append(avg_train_background_smooth_loss)
            self.train_target_smooth_losses.append(avg_train_target_smooth_loss)
            self.train_basis_smooth_losses.append(avg_train_basis_smooth_loss)
            self.train_decorr_raw_losses.append(avg_train_decorr_raw_loss)
            self.train_bg_context_raw_losses.append(avg_train_bg_context_raw_loss)
            self.train_target_context_raw_losses.append(avg_train_target_context_raw_loss)
            self.train_glucose_component_raw_losses.append(avg_train_glucose_component_raw_loss)
            self.train_spo2_component_raw_losses.append(avg_train_spo2_component_raw_loss)
        
            # 调整学习率
            scheduler.step()

            # 验证过程按照 batch 进行
            model.eval()
            val_running_loss = 0.0
            val_bg_running_loss = 0.0
            val_obs_running_loss = 0.0
            val_smooth_running_loss = 0.0
            val_decorr_running_loss = 0.0
            val_bg_context_running_loss = 0.0
            val_target_context_running_loss = 0.0
            val_glucose_component_running_loss = 0.0
            val_spo2_component_running_loss = 0.0
            val_obs_raw_running_loss = 0.0
            val_smooth_raw_running_loss = 0.0
            val_smooth_abs_running_loss = 0.0
            val_spectrum_rms_running = 0.0
            val_background_smooth_running_loss = 0.0
            val_target_smooth_running_loss = 0.0
            val_basis_smooth_running_loss = 0.0
            val_decorr_raw_running_loss = 0.0
            val_bg_context_raw_running_loss = 0.0
            val_target_context_raw_running_loss = 0.0
            val_glucose_component_raw_running_loss = 0.0
            val_spo2_component_raw_running_loss = 0.0
            with torch.no_grad():
                for batch_inputs_val, batch_labels_val in val_dataloader:
                    batch_inputs_val = [i.to(self.__device) for i in batch_inputs_val]
                    batch_labels_val = batch_labels_val.to(self.__device)
                    outputs_val = model(*batch_inputs_val)
                    val_loss, val_loss_items = self.__physics_loss(outputs_val, batch_inputs_val, batch_labels_val, criterion, train_para)
                    val_running_loss += val_loss.item()
                    val_bg_running_loss += val_loss_items["bg"].item()
                    val_obs_running_loss += val_loss_items["obs"].item()
                    val_smooth_running_loss += val_loss_items["smooth"].item()
                    val_decorr_running_loss += val_loss_items["decorr"].item()
                    val_bg_context_running_loss += val_loss_items["bg_context"].item()
                    val_target_context_running_loss += val_loss_items["target_context"].item()
                    val_glucose_component_running_loss += val_loss_items["glucose_component"].item()
                    val_spo2_component_running_loss += val_loss_items["spo2_component"].item()
                    val_obs_raw_running_loss += val_loss_items["obs_raw"].item()
                    val_smooth_raw_running_loss += val_loss_items["smooth_raw"].item()
                    val_smooth_abs_running_loss += val_loss_items["smooth_abs"].item()
                    val_spectrum_rms_running += val_loss_items["spectrum_rms"].item()
                    val_background_smooth_running_loss += val_loss_items["background_smooth"].item()
                    val_target_smooth_running_loss += val_loss_items["target_smooth"].item()
                    val_basis_smooth_running_loss += val_loss_items["basis_smooth"].item()
                    val_decorr_raw_running_loss += val_loss_items["decorr_raw"].item()
                    val_bg_context_raw_running_loss += val_loss_items["bg_context_raw"].item()
                    val_target_context_raw_running_loss += val_loss_items["target_context_raw"].item()
                    val_glucose_component_raw_running_loss += val_loss_items["glucose_component_raw"].item()
                    val_spo2_component_raw_running_loss += val_loss_items["spo2_component_raw"].item()

            avg_val_loss = val_running_loss / len(val_dataloader)
            avg_val_bg_loss = val_bg_running_loss / len(val_dataloader)
            avg_val_obs_loss = val_obs_running_loss / len(val_dataloader)
            avg_val_smooth_loss = val_smooth_running_loss / len(val_dataloader)
            avg_val_decorr_loss = val_decorr_running_loss / len(val_dataloader)
            avg_val_bg_context_loss = val_bg_context_running_loss / len(val_dataloader)
            avg_val_target_context_loss = val_target_context_running_loss / len(val_dataloader)
            avg_val_glucose_component_loss = val_glucose_component_running_loss / len(val_dataloader)
            avg_val_spo2_component_loss = val_spo2_component_running_loss / len(val_dataloader)
            avg_val_obs_raw_loss = val_obs_raw_running_loss / len(val_dataloader)
            avg_val_smooth_raw_loss = val_smooth_raw_running_loss / len(val_dataloader)
            avg_val_smooth_abs_loss = val_smooth_abs_running_loss / len(val_dataloader)
            avg_val_spectrum_rms = val_spectrum_rms_running / len(val_dataloader)
            avg_val_background_smooth_loss = val_background_smooth_running_loss / len(val_dataloader)
            avg_val_target_smooth_loss = val_target_smooth_running_loss / len(val_dataloader)
            avg_val_basis_smooth_loss = val_basis_smooth_running_loss / len(val_dataloader)
            avg_val_decorr_raw_loss = val_decorr_raw_running_loss / len(val_dataloader)
            avg_val_bg_context_raw_loss = val_bg_context_raw_running_loss / len(val_dataloader)
            avg_val_target_context_raw_loss = val_target_context_raw_running_loss / len(val_dataloader)
            avg_val_glucose_component_raw_loss = val_glucose_component_raw_running_loss / len(val_dataloader)
            avg_val_spo2_component_raw_loss = val_spo2_component_raw_running_loss / len(val_dataloader)
            self.val_losses.append(avg_val_loss)  # 记录验证损失
            self.val_bg_losses.append(avg_val_bg_loss)
            self.val_obs_losses.append(avg_val_obs_loss)
            self.val_smooth_losses.append(avg_val_smooth_loss)
            self.val_decorr_losses.append(avg_val_decorr_loss)
            self.val_bg_context_losses.append(avg_val_bg_context_loss)
            self.val_target_context_losses.append(avg_val_target_context_loss)
            self.val_glucose_component_losses.append(avg_val_glucose_component_loss)
            self.val_spo2_component_losses.append(avg_val_spo2_component_loss)
            self.val_obs_raw_losses.append(avg_val_obs_raw_loss)
            self.val_smooth_raw_losses.append(avg_val_smooth_raw_loss)
            self.val_smooth_abs_losses.append(avg_val_smooth_abs_loss)
            self.val_spectrum_rms_values.append(avg_val_spectrum_rms)
            self.val_background_smooth_losses.append(avg_val_background_smooth_loss)
            self.val_target_smooth_losses.append(avg_val_target_smooth_loss)
            self.val_basis_smooth_losses.append(avg_val_basis_smooth_loss)
            self.val_decorr_raw_losses.append(avg_val_decorr_raw_loss)
            self.val_bg_context_raw_losses.append(avg_val_bg_context_raw_loss)
            self.val_target_context_raw_losses.append(avg_val_target_context_raw_loss)
            self.val_glucose_component_raw_losses.append(avg_val_glucose_component_raw_loss)
            self.val_spo2_component_raw_losses.append(avg_val_spo2_component_raw_loss)

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.lrs.append(current_lr)
            
            end_time = time.time()  # 记录结束时间
            epoch_duration = end_time - start_time  # 计算耗时
        
            # 打印每个 epoch 的所有信息到一行
            log_message = f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_duration:.2f}s, Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}, Learning Rate: {current_lr:.5f}'
            if self.__is_physics_spectral_model(train_para):
                log_message += f', Train BG/ObsW/SmoothW/DecorrW: {avg_train_bg_loss:.5f}/{avg_train_obs_loss:.5f}/{avg_train_smooth_loss:.5f}/{avg_train_decorr_loss:.5f}'
                log_message += f', Val BG/ObsW/SmoothW/DecorrW: {avg_val_bg_loss:.5f}/{avg_val_obs_loss:.5f}/{avg_val_smooth_loss:.5f}/{avg_val_decorr_loss:.5f}'
                log_message += (
                    f', Train SmoothRel/Abs/RMS: '
                    f'{avg_train_smooth_raw_loss:.2e}/'
                    f'{avg_train_smooth_abs_loss:.2e}/'
                    f'{avg_train_spectrum_rms:.3e}')
                log_message += (
                    f', Val SmoothRel/Abs/RMS: '
                    f'{avg_val_smooth_raw_loss:.2e}/'
                    f'{avg_val_smooth_abs_loss:.2e}/'
                    f'{avg_val_spectrum_rms:.3e}')
                log_message += (
                    f', Train SmBg/Tar/BasisW: '
                    f'{avg_train_background_smooth_loss:.2e}/'
                    f'{avg_train_target_smooth_loss:.2e}/'
                    f'{avg_train_basis_smooth_loss:.2e}')
                log_message += (
                    f', Val SmBg/Tar/BasisW: '
                    f'{avg_val_background_smooth_loss:.2e}/'
                    f'{avg_val_target_smooth_loss:.2e}/'
                    f'{avg_val_basis_smooth_loss:.2e}')
                if train_para.get('enable_context_adversarial', False):
                    log_message += f', Train CtxBgW/CtxTarAdvW: {avg_train_bg_context_loss:.5f}/{avg_train_target_context_loss:.5f}'
                    log_message += f', Val CtxBgW/CtxTarAdvW: {avg_val_bg_context_loss:.5f}/{avg_val_target_context_loss:.5f}'
                if train_para.get('enable_aux_spo2_spectrum', False):
                    log_message += f', Train CompG/Spo2W: {avg_train_glucose_component_loss:.5f}/{avg_train_spo2_component_loss:.5f}'
                    log_message += f', Val CompG/Spo2W: {avg_val_glucose_component_loss:.5f}/{avg_val_spo2_component_loss:.5f}'
                elif train_para.get('enable_single_target_residual', False):
                    component_name = train_para.get(
                        'primary_target_name',
                        train_para.get('target_name', 'target'))
                    log_message += (
                        f', Train Comp{component_name}W: '
                        f'{avg_train_glucose_component_loss:.5f}')
                    log_message += (
                        f', Val Comp{component_name}W: '
                        f'{avg_val_glucose_component_loss:.5f}')

            # 检查 EarlyStopping，并更新模型
            monitored_val_loss = (
                avg_val_bg_loss if early_stopping_monitor == 'bg' else avg_val_loss)
            monitor_label = (
                'Validation BG loss' if early_stopping_monitor == 'bg'
                else 'Validation loss')
            if monitored_val_loss < early_stopping.val_loss_min:
                log_message += f', {monitor_label} decreased ({early_stopping.val_loss_min:.5f} --> {monitored_val_loss:.5f}). Saving model...'
                best_state_dict = copy.deepcopy(model.state_dict())  # 保存最优的模型
            elif early_stopping.counter > 0:
                log_message += f', EarlyStopping counter: {early_stopping.counter} out of {early_stopping.patience}'

            print(log_message)
            
            # 调用 early stopping
            early_stopping(monitored_val_loss, model)
            
            if early_stopping.early_stop:
                print(f"Early stopping at Epoch {epoch+1}")
                break

        # 训练结束
        print("Training complete.")
        torch.cuda.empty_cache()
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        self.model = model

    def model_eval(self, X, Y, batch_size=32, path = None):
        # model = self.model

        train_para = getattr(self, 'train_para', {})
        model = self.__create_model(train_para).to(self.__device)

        # 加载模型的权重
        model.load_state_dict(torch.load(path, map_location=self.__device))
        
        model.eval()

        # 将数据转换为 tensor 并移动到正确的设备
        x_OP, x_TH, x_DF, x_Demo, x_MI, labels = self.__array_to_tensor(X, Y)
        
        print(x_OP.shape)
        print(x_TH.shape)
        print(x_DF.shape)
        print(x_Demo.shape)
        print(x_MI.shape)
        # 获取数据大小
        num_samples = x_OP.size(0)
        
        outputs_list = []
        
        with torch.no_grad():
            # 批量加载数据
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                
                # 获取每个 batch 的数据
                x_OP_batch = x_OP[start_idx:end_idx].to(self.__device)
                x_TH_batch = x_TH[start_idx:end_idx].to(self.__device)
                x_DF_batch = x_DF[start_idx:end_idx].to(self.__device)
                x_Demo_batch = x_Demo[start_idx:end_idx].to(self.__device)
                x_MI_batch = x_MI[start_idx:end_idx].to(self.__device)
                
                # 前向传播
                outputs_batch = model(x_OP_batch, x_TH_batch, x_DF_batch, x_Demo_batch, x_MI_batch)
                outputs_batch = self.__get_bg_output(outputs_batch)
                
                # 将结果转为 NumPy 格式并保存
                outputs_np = outputs_batch.cpu().numpy() if outputs_batch.is_cuda else outputs_batch.numpy()
                outputs_list.append(outputs_np)
        
        # 将所有批次的结果拼接成一个数组
        return np.concatenate(outputs_list, axis=0)

    def model_aux_eval(self, X, Y, batch_size=32, path=None, max_samples=16):
        train_para = getattr(self, 'train_para', {})
        if not self.__is_physics_spectral_model(train_para):
            return None

        H, wavelengths = self.__build_observation_data(train_para)
        model = self.__create_model(train_para).to(self.__device)
        model.load_state_dict(torch.load(path, map_location=self.__device))
        model.eval()

        x_OP, x_TH, x_DF, x_Demo, x_MI, labels = self.__array_to_tensor(X, Y)
        num_samples = min(max_samples, x_OP.size(0))
        if num_samples <= 0:
            return None

        x_OP = x_OP[:num_samples]
        x_TH = x_TH[:num_samples]
        x_DF = x_DF[:num_samples]
        x_Demo = x_Demo[:num_samples]
        x_MI = x_MI[:num_samples]
        labels = labels[:num_samples]

        bg_outputs = []
        spectrum_outputs = []
        y_hat_outputs = []
        clean_y_hat_outputs = []
        baseline_outputs = []
        noise_outputs = []
        background_spectrum_outputs = []
        target_spectrum_outputs = []
        target_glucose_spectrum_outputs = []
        target_spo2_spectrum_outputs = []
        background_coefficient_outputs = []
        target_glucose_coefficient_outputs = []
        target_spo2_coefficient_outputs = []
        glucose_component_outputs = []
        spo2_component_outputs = []
        primary_target_component_outputs = []
        context_target_outputs = []
        background_context_pred_outputs = []
        target_context_pred_outputs = []
        background_context_film_outputs = []
        common_target_outputs = []
        target_spectral_contribution_outputs = []
        target_spectral_gate_outputs = []
        token_outputs = []
        gate_outputs = []
        spectral_gate_outputs = []

        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                x_OP_batch = x_OP[start_idx:end_idx].to(self.__device)
                x_TH_batch = x_TH[start_idx:end_idx].to(self.__device)
                x_DF_batch = x_DF[start_idx:end_idx].to(self.__device)
                x_Demo_batch = x_Demo[start_idx:end_idx].to(self.__device)
                x_MI_batch = x_MI[start_idx:end_idx].to(self.__device)

                outputs = model(x_OP_batch, x_TH_batch, x_DF_batch, x_Demo_batch, x_MI_batch)
                if not isinstance(outputs, dict):
                    return None

                bg_outputs.append(outputs["bg"].cpu().numpy())
                spectrum_outputs.append(outputs["spectrum"].cpu().numpy())
                y_hat_outputs.append(outputs["y_hat"].cpu().numpy())
                if outputs.get("clean_y_hat") is not None:
                    clean_y_hat_outputs.append(outputs["clean_y_hat"].cpu().numpy())
                if outputs.get("baseline_hat") is not None:
                    baseline_outputs.append(outputs["baseline_hat"].cpu().numpy())
                if outputs.get("noise_hat") is not None:
                    noise_outputs.append(outputs["noise_hat"].cpu().numpy())
                if outputs.get("background_spectrum") is not None:
                    background_spectrum_outputs.append(outputs["background_spectrum"].cpu().numpy())
                if outputs.get("target_spectrum") is not None:
                    target_spectrum_outputs.append(outputs["target_spectrum"].cpu().numpy())
                if outputs.get("target_glucose_spectrum") is not None:
                    target_glucose_spectrum_outputs.append(outputs["target_glucose_spectrum"].cpu().numpy())
                if outputs.get("target_spo2_spectrum") is not None:
                    target_spo2_spectrum_outputs.append(outputs["target_spo2_spectrum"].cpu().numpy())
                if outputs.get("background_coefficients") is not None:
                    background_coefficient_outputs.append(outputs["background_coefficients"].cpu().numpy())
                if outputs.get("target_glucose_coefficients") is not None:
                    target_glucose_coefficient_outputs.append(outputs["target_glucose_coefficients"].cpu().numpy())
                if outputs.get("target_spo2_coefficients") is not None:
                    target_spo2_coefficient_outputs.append(outputs["target_spo2_coefficients"].cpu().numpy())
                if outputs.get("glucose_component_pred") is not None:
                    glucose_component_outputs.append(outputs["glucose_component_pred"].cpu().numpy())
                if outputs.get("spo2_component_pred") is not None:
                    spo2_component_outputs.append(outputs["spo2_component_pred"].cpu().numpy())
                if outputs.get("primary_target_component_pred") is not None:
                    primary_target_component_outputs.append(
                        outputs["primary_target_component_pred"].cpu().numpy())
                if outputs.get("context_target") is not None:
                    context_target_outputs.append(outputs["context_target"].cpu().numpy())
                if outputs.get("background_context_pred") is not None:
                    background_context_pred_outputs.append(outputs["background_context_pred"].cpu().numpy())
                if outputs.get("target_context_pred") is not None:
                    target_context_pred_outputs.append(outputs["target_context_pred"].cpu().numpy())
                if outputs.get("background_context_film") is not None:
                    background_context_film_outputs.append(outputs["background_context_film"].cpu().numpy())
                if outputs.get("common_target_pred") is not None:
                    common_target_outputs.append(outputs["common_target_pred"].cpu().numpy())
                if outputs.get("target_spectral_contribution") is not None:
                    target_spectral_contribution_outputs.append(
                        outputs["target_spectral_contribution"].cpu().numpy())
                if outputs.get("target_spectral_gates") is not None:
                    target_gates = outputs["target_spectral_gates"].detach().view(1, -1)
                    target_spectral_gate_outputs.append(
                        target_gates.repeat(end_idx - start_idx, 1).cpu().numpy())
                if outputs.get("tokens") is not None:
                    token_outputs.append(outputs["tokens"].cpu().numpy())
                if outputs.get("gate_weights") is not None:
                    gate_outputs.append(outputs["gate_weights"].cpu().numpy())
                if outputs.get("spectral_gate") is not None:
                    spectral_gate = outputs["spectral_gate"].detach().view(1).cpu().numpy()
                    spectral_gate_outputs.append(np.repeat(spectral_gate, end_idx - start_idx))

        aux = {
            "bg_pred": np.concatenate(bg_outputs, axis=0),
            "bg_true": labels.cpu().numpy(),
            "spectrum": np.concatenate(spectrum_outputs, axis=0),
            "y_hat": np.concatenate(y_hat_outputs, axis=0),
            "y_real": x_OP.cpu().numpy(),
            "H": H,
            "wavelengths": wavelengths
        }
        spectral_generator = getattr(model, "spectral_generator", None)
        if spectral_generator is not None and hasattr(spectral_generator, "basis"):
            aux["spectral_basis"] = spectral_generator.basis().detach().cpu().numpy()
        role_moe = getattr(model, "role_moe", None)
        if role_moe is not None:
            if hasattr(role_moe.background_generator, "basis"):
                aux["background_spectral_basis"] = role_moe.background_generator.basis().detach().cpu().numpy()
            if hasattr(role_moe.target_generator, "basis"):
                aux["target_spectral_basis"] = role_moe.target_generator.basis().detach().cpu().numpy()
                aux["target_glucose_spectral_basis"] = role_moe.target_generator.basis().detach().cpu().numpy()
            if getattr(role_moe, "target_spo2_generator", None) is not None and hasattr(role_moe.target_spo2_generator, "basis"):
                aux["target_spo2_spectral_basis"] = role_moe.target_spo2_generator.basis().detach().cpu().numpy()
        if clean_y_hat_outputs:
            aux["clean_y_hat"] = np.concatenate(clean_y_hat_outputs, axis=0)
        if baseline_outputs:
            aux["baseline_hat"] = np.concatenate(baseline_outputs, axis=0)
        if noise_outputs:
            aux["noise_hat"] = np.concatenate(noise_outputs, axis=0)
        if background_spectrum_outputs:
            aux["background_spectrum"] = np.concatenate(background_spectrum_outputs, axis=0)
        if target_spectrum_outputs:
            aux["target_spectrum"] = np.concatenate(target_spectrum_outputs, axis=0)
        if target_glucose_spectrum_outputs:
            aux["target_glucose_spectrum"] = np.concatenate(target_glucose_spectrum_outputs, axis=0)
            aux["primary_target_spectrum"] = aux["target_glucose_spectrum"]
        if target_spo2_spectrum_outputs:
            aux["target_spo2_spectrum"] = np.concatenate(target_spo2_spectrum_outputs, axis=0)
        if background_coefficient_outputs:
            aux["background_coefficients"] = np.concatenate(background_coefficient_outputs, axis=0)
        if target_glucose_coefficient_outputs:
            aux["target_glucose_coefficients"] = np.concatenate(target_glucose_coefficient_outputs, axis=0)
            aux["primary_target_coefficients"] = aux["target_glucose_coefficients"]
        if target_spo2_coefficient_outputs:
            aux["target_spo2_coefficients"] = np.concatenate(target_spo2_coefficient_outputs, axis=0)
        if glucose_component_outputs:
            aux["glucose_component_pred"] = np.concatenate(glucose_component_outputs, axis=0)
        if spo2_component_outputs:
            aux["spo2_component_pred"] = np.concatenate(spo2_component_outputs, axis=0)
        if primary_target_component_outputs:
            aux["primary_target_component_pred"] = np.concatenate(
                primary_target_component_outputs, axis=0)
        if context_target_outputs:
            aux["context_target"] = np.concatenate(context_target_outputs, axis=0)
        if background_context_pred_outputs:
            aux["background_context_pred"] = np.concatenate(background_context_pred_outputs, axis=0)
        if target_context_pred_outputs:
            aux["target_context_pred"] = np.concatenate(target_context_pred_outputs, axis=0)
        if background_context_film_outputs:
            aux["background_context_film"] = np.concatenate(background_context_film_outputs, axis=0)
        if common_target_outputs:
            aux["common_target_pred"] = np.concatenate(common_target_outputs, axis=0)
        if target_spectral_contribution_outputs:
            aux["target_spectral_contribution"] = np.concatenate(
                target_spectral_contribution_outputs, axis=0)
        if target_spectral_gate_outputs:
            aux["target_spectral_gates"] = np.concatenate(
                target_spectral_gate_outputs, axis=0)
        if token_outputs:
            aux["tokens"] = np.concatenate(token_outputs, axis=0)
        if gate_outputs:
            aux["gate_weights"] = np.concatenate(gate_outputs, axis=0)
        if spectral_gate_outputs:
            aux["spectral_feature_gate"] = np.concatenate(spectral_gate_outputs, axis=0)
        return aux
    
    def get_model_summary(self):
        # 使用 StringIO 来捕获 summary 输出
        buffer = io.StringIO()

        # 确保模型和输入形状已经正确设置
        if self.model is None:
            raise ValueError("Model is not initialized.")
        if not self.input_shapes:
            raise ValueError("Input shapes are not set.")
        if summary is None:
            raise ImportError("torchinfo is required for get_model_summary(). Install it or skip model summary export.")

        # 调用 summary，并将其打印到 buffer
        model_summary = summary(self.model, input_size=[(1,) + shape for shape in self.input_shapes], device="cuda", verbose=0)
        
        # 打印到 StringIO 中
        buffer.write(str(model_summary))

        # 返回 buffer 的内容
        return buffer.getvalue()
    
    def model_save(self, file_path):
        # torch.save(self.model, file_path)
        torch.save(self.model.state_dict(), file_path)
