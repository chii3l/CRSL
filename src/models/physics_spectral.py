import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SelfAttention
from .resnet import ResNet1DBlock, ResNet2DBlock
from .tcn import TCN1DBlock
from .gru import GRUBackbone1D
from .lstm import LSTMBackbone1D
from .lstm_attention import LSTMAttentionBackbone1D
from .patchtst import PatchTSTBackbone1D
from .transformer import TransformerBackbone1D
from .biosignal_factory import BIOSIGNAL_BACKBONE_NAMES, create_biosignal_backbone
from .recent_factory import RECENT_TOP_BACKBONE_NAMES, create_recent_top_backbone
from .timeseries_library import TSL_BACKBONE_MODULES, TimeSeriesLibraryBackbone1D
from .multi_modal import MultiModalModel


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverseFunction.apply(x, lambd)


def _parse_context_modalities(context_modalities):
    if context_modalities is None:
        return ["th", "demo"]
    if isinstance(context_modalities, str):
        values = [item.strip().lower() for item in context_modalities.replace(";", ",").split(",")]
    else:
        values = [str(item).strip().lower() for item in context_modalities]
    alias = {
        "temperature": "th",
        "humidity": "th",
        "temp_humidity": "th",
        "t&h": "th",
        "demographic": "demo",
        "demographics": "demo",
        "subject": "demo",
        "mi": "mi",
        "df": "df",
    }
    parsed = []
    for value in values:
        value = alias.get(value, value)
        if value in ["th", "demo", "df", "mi"] and value not in parsed:
            parsed.append(value)
    return parsed


class SpectralTokenMoE(nn.Module):
    def __init__(self, hidden_channels, num_experts=4, dropout_rate=0.1):
        super(SpectralTokenMoE, self).__init__()
        self.num_experts = num_experts
        kernels = [3, 5, 7, 9]
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            kernel_size = kernels[i % len(kernels)]
            padding = kernel_size // 2
            self.experts.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True)
            ))

        self.gate = nn.Linear(hidden_channels, num_experts)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_logits = self.gate(x.mean(dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=1)
        mixed = torch.sum(expert_outputs * gate_weights[:, :, None, None], dim=1)
        return mixed, expert_outputs, gate_weights


class RoleAwareSpectralMoE(nn.Module):
    """Decompose PPG into observation artifacts and latent spectral components."""

    def __init__(
            self,
            hidden_channels,
            ppg_channels,
            spectral_dim,
            generator_mode="basis",
            num_basis=16,
            dropout_rate=0.1,
            use_gate=True,
            gate_mode="softmax",
            sigmoid_gate_init=0.2,
            baseline_lowpass_kernel=0,
            noise_highpass_kernel=0,
            enable_observation_artifact_branches=True,
            enable_aux_spo2_spectrum=False,
            context_dim=0,
            context_hidden_dim=64,
            context_condition_background=False):
        super(RoleAwareSpectralMoE, self).__init__()
        self.use_gate = bool(use_gate)
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in ["softmax", "sigmoid"]:
            raise ValueError("role gate_mode must be 'softmax' or 'sigmoid'")
        self.enable_aux_spo2_spectrum = bool(enable_aux_spo2_spectrum)
        self.enable_observation_artifact_branches = bool(
            enable_observation_artifact_branches)
        self.ppg_channels = int(ppg_channels)
        self.baseline_lowpass_kernel = self._normalize_kernel(baseline_lowpass_kernel)
        self.noise_highpass_kernel = self._normalize_kernel(noise_highpass_kernel)
        self.context_condition_background = bool(context_condition_background) and int(context_dim or 0) > 0

        def hidden_block(kernel_size):
            padding = int(kernel_size) // 2
            return nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=int(kernel_size), padding=padding, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(float(dropout_rate)))

        def spectrum_generator():
            mode = str(generator_mode).lower()
            if mode in ["pointwise", "conv1x1", "linear"]:
                return PointwiseSpectralGenerator(hidden_channels, spectral_dim)
            if mode in ["basis", "basis_spectral", "basis_generator"]:
                return BasisSpectralGenerator(
                    hidden_channels,
                    spectral_dim,
                    num_basis=num_basis,
                    dropout_rate=dropout_rate)
            raise ValueError("role-aware generator_mode must be 'pointwise' or 'basis'")

        self.baseline_hidden = (
            hidden_block(15) if self.enable_observation_artifact_branches else None)
        self.noise_hidden = (
            hidden_block(3) if self.enable_observation_artifact_branches else None)
        self.background_hidden = hidden_block(9)
        self.target_hidden = hidden_block(5)
        self.target_spo2_hidden = hidden_block(5) if self.enable_aux_spo2_spectrum else None

        self.baseline_head = (
            nn.Conv1d(hidden_channels, ppg_channels, kernel_size=1)
            if self.enable_observation_artifact_branches else None)
        self.noise_head = (
            nn.Conv1d(hidden_channels, ppg_channels, kernel_size=1)
            if self.enable_observation_artifact_branches else None)
        self.background_generator = spectrum_generator()
        self.target_generator = spectrum_generator()
        self.target_spo2_generator = spectrum_generator() if self.enable_aux_spo2_spectrum else None
        spectral_roles = 3 if self.enable_aux_spo2_spectrum else 2
        self.num_roles = spectral_roles + (
            2 if self.enable_observation_artifact_branches else 0)
        self.gate = nn.Linear(hidden_channels, self.num_roles) if self.use_gate else None
        if self.gate is not None and self.gate_mode == "sigmoid":
            gate_init = min(max(float(sigmoid_gate_init), 1e-4), 1.0 - 1e-4)
            gate_bias = torch.logit(torch.tensor(gate_init)).item()
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias)

        self.background_context_film = None
        if self.context_condition_background:
            context_hidden_dim = max(8, int(context_hidden_dim))
            final_layer = nn.Linear(context_hidden_dim, hidden_channels * 2)
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
            self.background_context_film = nn.Sequential(
                nn.LayerNorm(int(context_dim)),
                nn.Linear(int(context_dim), context_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout_rate)),
                final_layer)

    @staticmethod
    def _normalize_kernel(kernel_size):
        kernel_size = int(kernel_size or 0)
        if kernel_size <= 1:
            return 0
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    @staticmethod
    def _moving_average(x, kernel_size):
        if kernel_size <= 1:
            return x
        pad = int(kernel_size) // 2
        padded = F.pad(x, (pad, pad), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=int(kernel_size), stride=1)

    def _condition_background(self, background_hidden, context):
        if self.background_context_film is None or context is None:
            return background_hidden, None
        gamma_beta = self.background_context_film(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return background_hidden * (1.0 + gamma) + beta, gamma_beta

    @staticmethod
    def _generate_with_coefficients(generator, hidden):
        if hasattr(generator, "forward_with_coefficients"):
            return generator.forward_with_coefficients(hidden)
        return generator(hidden), None

    def forward(self, x, context=None):
        baseline_hidden = (
            self.baseline_hidden(x)
            if self.enable_observation_artifact_branches else None)
        noise_hidden = (
            self.noise_hidden(x)
            if self.enable_observation_artifact_branches else None)
        background_hidden = self.background_hidden(x)
        target_hidden = self.target_hidden(x)
        target_spo2_hidden = self.target_spo2_hidden(x) if self.target_spo2_hidden is not None else None
        background_hidden, background_context_film = self._condition_background(background_hidden, context)

        if self.enable_observation_artifact_branches:
            baseline = self.baseline_head(baseline_hidden)
            noise = self.noise_head(noise_hidden)
            if self.baseline_lowpass_kernel > 1:
                baseline = self._moving_average(
                    baseline, self.baseline_lowpass_kernel)
            if self.noise_highpass_kernel > 1:
                noise = noise - self._moving_average(
                    noise, self.noise_highpass_kernel)
        else:
            output_shape = (x.size(0), self.ppg_channels, x.size(-1))
            baseline = x.new_zeros(output_shape)
            noise = x.new_zeros(output_shape)
        background_spectrum, background_coefficients = self._generate_with_coefficients(
            self.background_generator, background_hidden)
        target_glucose_spectrum, target_glucose_coefficients = self._generate_with_coefficients(
            self.target_generator, target_hidden)
        target_spo2_spectrum = None
        target_spo2_coefficients = None
        if self.target_spo2_generator is not None and target_spo2_hidden is not None:
            target_spo2_spectrum, target_spo2_coefficients = self._generate_with_coefficients(
                self.target_spo2_generator, target_spo2_hidden)

        gate_weights = None
        if self.gate is not None:
            gate_logits = self.gate(x.mean(dim=-1))
            if self.gate_mode == "sigmoid":
                gate_weights = torch.sigmoid(gate_logits)
            else:
                gate_weights = torch.softmax(gate_logits, dim=1)
            role_index = 0
            if self.enable_observation_artifact_branches:
                baseline = baseline * gate_weights[:, 0:1, None]
                noise = noise * gate_weights[:, 1:2, None]
                role_index = 2
            background_spectrum = (
                background_spectrum *
                gate_weights[:, role_index:role_index + 1, None])
            if background_coefficients is not None:
                background_coefficients = (
                    background_coefficients *
                    gate_weights[:, role_index:role_index + 1, None])
            role_index += 1
            target_glucose_spectrum = (
                target_glucose_spectrum *
                gate_weights[:, role_index:role_index + 1, None])
            if target_glucose_coefficients is not None:
                target_glucose_coefficients = (
                    target_glucose_coefficients *
                    gate_weights[:, role_index:role_index + 1, None])
            role_index += 1
            if target_spo2_spectrum is not None:
                target_spo2_spectrum = (
                    target_spo2_spectrum *
                    gate_weights[:, role_index:role_index + 1, None])
            if target_spo2_coefficients is not None:
                target_spo2_coefficients = (
                    target_spo2_coefficients *
                    gate_weights[:, role_index:role_index + 1, None])

        token_list = []
        if self.enable_observation_artifact_branches:
            token_list.extend([baseline_hidden, noise_hidden])
        token_list.extend([background_hidden, target_hidden])
        if target_spo2_hidden is not None:
            token_list.append(target_spo2_hidden)
        expert_tokens = torch.stack(token_list, dim=1)
        target_spectrum = target_glucose_spectrum
        if target_spo2_spectrum is not None:
            target_spectrum = target_spectrum + target_spo2_spectrum

        spectral_bases = []
        for generator in (
                self.background_generator,
                self.target_generator,
                self.target_spo2_generator):
            if generator is not None and hasattr(generator, "basis"):
                spectral_bases.append(generator.basis())
        spectral_bases = (
            torch.stack(spectral_bases, dim=0) if spectral_bases else None)

        return {
            "baseline": baseline,
            "noise": noise,
            "background_spectrum": background_spectrum,
            "target_spectrum": target_spectrum,
            "target_glucose_spectrum": target_glucose_spectrum,
            "target_spo2_spectrum": target_spo2_spectrum,
            "background_coefficients": background_coefficients,
            "target_glucose_coefficients": target_glucose_coefficients,
            "target_spo2_coefficients": target_spo2_coefficients,
            "spectral_bases": spectral_bases,
            "tokens": expert_tokens,
            "gate_weights": gate_weights,
            "background_context_film": background_context_film,
        }


def _inverse_softplus(x):
    x = torch.clamp(x, min=1e-6)
    return torch.log(torch.expm1(x))


class PointwiseSpectralGenerator(nn.Module):
    def __init__(self, hidden_channels, spectral_dim):
        super(PointwiseSpectralGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_channels, spectral_dim, kernel_size=1),
            nn.Softplus())

    def forward(self, x):
        return self.net(x)


class BasisSpectralGenerator(nn.Module):
    """Generate spectra as non-negative combinations of smooth learnable bases."""

    def __init__(
            self,
            hidden_channels,
            spectral_dim,
            num_basis=16,
            dropout_rate=0.1):
        super(BasisSpectralGenerator, self).__init__()
        self.num_basis = max(1, int(num_basis))
        self.spectral_dim = int(spectral_dim)
        self.coeff_net = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout_rate)),
            nn.Conv1d(hidden_channels, self.num_basis, kernel_size=1),
            nn.Softplus())
        init_basis = self._make_gaussian_basis(self.num_basis, self.spectral_dim)
        self.basis_logits = nn.Parameter(_inverse_softplus(init_basis))

    @staticmethod
    def _make_gaussian_basis(num_basis, spectral_dim):
        grid = torch.linspace(0.0, 1.0, steps=int(spectral_dim)).unsqueeze(0)
        centers = torch.linspace(0.0, 1.0, steps=int(num_basis)).unsqueeze(1)
        width = max(1.0 / max(int(num_basis) - 1, 1), 0.08)
        basis = torch.exp(-0.5 * ((grid - centers) / width) ** 2) + 1e-4
        return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def basis(self):
        basis = F.softplus(self.basis_logits)
        return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def coefficients(self, x):
        return self.coeff_net(x)

    def forward_with_coefficients(self, x):
        coefficients = self.coeff_net(x)
        spectrum = torch.einsum("bkt,kd->bdt", coefficients, self.basis())
        return spectrum, coefficients

    def forward(self, x):
        spectrum, _ = self.forward_with_coefficients(x)
        return spectrum


class PhysicsSpectralMultiModalModel(nn.Module):
    def __init__(
            self,
            filter_structure,
            dense_structure,
            dropout_rate,
            observation_matrix,
            fusion_name='concat',
            enable_deconv=False,
            spectral_hidden=64,
            enable_token_moe=False,
            num_experts=4,
            token_moe_dropout=0.1,
            token_moe_mode='anonymous',
            role_moe_gate=True,
            role_moe_gate_mode='softmax',
            role_moe_sigmoid_gate_init=0.2,
            spectral_fusion_mode='concat',
            spectral_gate_init=-4.0,
            spectral_generator_mode='pointwise',
            spectral_basis_count=16,
            spectral_basis_dropout=0.1,
            ppg_backbone_name='resnet',
            ppg_backbone_config=None,
            ppg_seq_len=300,
            active_modalities=None,
            enable_context_adversarial=False,
            context_modalities=None,
            context_grl_lambda=1.0,
            context_hidden_dim=64,
            baseline_lowpass_kernel=0,
            noise_highpass_kernel=0,
            enable_observation_artifact_branches=True,
            output_dim=1,
            enable_aux_spo2_spectrum=False,
            enable_target_specific_regression=True,
            enable_single_target_residual=False,
            primary_target_name='target',
            target_specific_regression_mode='gated_residual',
            target_spectral_gate_init=-3.0,
            glucose_spectral_gate_max=0.30,
            spo2_spectral_gate_max=0.15,
            enable_background_context_conditioning=False,
            background_context_hidden_dim=None):
        super(PhysicsSpectralMultiModalModel, self).__init__()

        H = torch.as_tensor(observation_matrix, dtype=torch.float32)
        if H.ndim != 2:
            raise ValueError("observation_matrix must have shape (n_channels, n_wavelengths)")

        self.fusion_name = fusion_name
        self.enable_deconv = enable_deconv
        self.output_dim = int(output_dim)
        self.enable_aux_spo2_spectrum = bool(enable_aux_spo2_spectrum)
        self.enable_single_target_residual = bool(enable_single_target_residual)
        self.primary_target_name = str(primary_target_name)
        self.spectral_fusion_mode = str(spectral_fusion_mode).lower()
        self.context_grl_lambda = float(context_grl_lambda)
        if self.spectral_fusion_mode not in ['concat', 'none', 'gate']:
            raise ValueError("spectral_fusion_mode must be one of: concat, none, gate")
        if active_modalities is None:
            active_modalities = [1, 1, 1, 1, 1]
        if len(active_modalities) != 5:
            raise ValueError("active_modalities must contain 5 values: PPG, TH, Demo, DF, MI")
        self.register_buffer("modal_mask", torch.as_tensor(active_modalities, dtype=torch.float32), persistent=False)
        self.register_buffer("H", H)
        self.context_modalities = _parse_context_modalities(context_modalities)
        context_dims = {"th": 6, "demo": 9, "df": 6, "mi": 1}
        self.context_dim = sum(context_dims[name] for name in self.context_modalities)
        self.enable_background_context_conditioning = (
            bool(enable_background_context_conditioning) and
            self.context_dim > 0)
        if background_context_hidden_dim is None:
            background_context_hidden_dim = context_hidden_dim

        ppg_channels = H.shape[0]
        spectral_dim = H.shape[1]
        feature_width = filter_structure[-1]
        self.base_feature_dim = feature_width * 2 + 6 + 9 + 1
        self.spectral_feature_dim = feature_width if self.spectral_fusion_mode != 'none' else 0
        self.feature_dim = self.base_feature_dim + self.spectral_feature_dim
        self.target_specific_regression_mode = str(target_specific_regression_mode).lower()
        if self.target_specific_regression_mode not in [
                'concat', 'gated_residual', 'bounded_gated_residual',
                'baseline_residual_plugin']:
            raise ValueError(
                "target_specific_regression_mode must be 'concat', "
                "'gated_residual', 'bounded_gated_residual', or "
                "'baseline_residual_plugin'")
        self.use_dual_target_regression = (
            bool(enable_target_specific_regression) and
            self.output_dim == 2 and
            self.enable_aux_spo2_spectrum and
            self.spectral_fusion_mode != 'none')
        self.use_single_target_regression = (
            bool(enable_target_specific_regression) and
            self.enable_single_target_residual and
            self.output_dim == 1 and
            not self.enable_aux_spo2_spectrum and
            self.spectral_fusion_mode != 'none')
        if (
                self.use_single_target_regression and
                self.target_specific_regression_mode != 'baseline_residual_plugin'):
            raise ValueError(
                "Single-target spectral regression currently requires "
                "target_specific_regression_mode='baseline_residual_plugin'")
        self.use_target_specific_regression = (
            self.use_dual_target_regression or
            self.use_single_target_regression)
        self.use_baseline_residual_plugin = (
            self.use_target_specific_regression and
            self.target_specific_regression_mode == 'baseline_residual_plugin')
        self.register_buffer(
            "target_spectral_gate_max",
            torch.tensor(
                [float(glucose_spectral_gate_max), float(spo2_spectral_gate_max)],
                dtype=torch.float32))

        self.baseline_predictor = None
        if self.use_baseline_residual_plugin:
            self.baseline_predictor = MultiModalModel(
                filter_structure,
                dense_structure,
                dropout_rate,
                fusion_name=fusion_name,
                enable_deconv=enable_deconv,
                ppg_backbone_name=ppg_backbone_name,
                ppg_backbone_config=ppg_backbone_config,
                ppg_seq_len=ppg_seq_len,
                ppg_input_channels=ppg_channels,
                active_modalities=active_modalities)

        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(ppg_channels, spectral_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(spectral_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(spectral_hidden, spectral_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(spectral_hidden),
            nn.ReLU(inplace=True)
        )
        self.spectral_generator_mode = str(spectral_generator_mode).lower()
        self.token_moe_mode = str(token_moe_mode).lower()
        if self.spectral_generator_mode in ['pointwise', 'conv1x1', 'linear']:
            self.spectral_generator = PointwiseSpectralGenerator(spectral_hidden, spectral_dim)
        elif self.spectral_generator_mode in ['basis', 'basis_spectral', 'basis_generator']:
            self.spectral_generator = BasisSpectralGenerator(
                spectral_hidden,
                spectral_dim,
                num_basis=spectral_basis_count,
                dropout_rate=spectral_basis_dropout)
        else:
            raise ValueError("spectral_generator_mode must be 'pointwise' or 'basis'")
        self.token_moe = SpectralTokenMoE(
            spectral_hidden,
            num_experts=num_experts,
            dropout_rate=token_moe_dropout
        ) if enable_token_moe and self.token_moe_mode in ['anonymous', 'standard', 'default'] else None
        self.role_moe = RoleAwareSpectralMoE(
            spectral_hidden,
            ppg_channels,
            spectral_dim,
            generator_mode=self.spectral_generator_mode,
            num_basis=spectral_basis_count,
            dropout_rate=token_moe_dropout,
            use_gate=role_moe_gate,
            gate_mode=role_moe_gate_mode,
            sigmoid_gate_init=role_moe_sigmoid_gate_init,
            baseline_lowpass_kernel=baseline_lowpass_kernel,
            noise_highpass_kernel=noise_highpass_kernel,
            enable_observation_artifact_branches=(
                enable_observation_artifact_branches),
            enable_aux_spo2_spectrum=self.enable_aux_spo2_spectrum,
            context_dim=self.context_dim,
            context_hidden_dim=background_context_hidden_dim,
            context_condition_background=self.enable_background_context_conditioning
        ) if enable_token_moe and self.token_moe_mode in ['role_aware', 'role', 'physical'] else None

        self.glucose_component_head = None
        self.spo2_component_head = None
        self.target_component_head = None
        if self.use_dual_target_regression or self.use_single_target_regression:
            def component_head():
                return nn.Sequential(
                    nn.LayerNorm(spectral_dim),
                    nn.Linear(spectral_dim, max(16, spectral_dim // 4)),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(max(16, spectral_dim // 4), 1),
                    nn.Sigmoid())

            if self.use_single_target_regression:
                self.target_component_head = component_head()
            else:
                self.glucose_component_head = component_head()
                self.spo2_component_head = component_head()

        self.enable_context_adversarial = (
            bool(enable_context_adversarial) and
            self.role_moe is not None and
            self.context_dim > 0)
        self.background_context_predictor = None
        self.target_context_predictor = None
        if self.enable_context_adversarial:
            context_hidden_dim = max(8, int(context_hidden_dim))

            def context_predictor():
                return nn.Sequential(
                    nn.LayerNorm(spectral_dim),
                    nn.Linear(spectral_dim, context_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(context_hidden_dim, self.context_dim))

            self.background_context_predictor = context_predictor()
            self.target_context_predictor = context_predictor()

        self.spectral_projection = None
        self.glucose_spectral_projection = None
        self.spo2_spectral_projection = None
        self.target_spectral_projection = None
        self.spectral_gate_logit = None
        if self.spectral_fusion_mode != 'none':
            def create_spectral_projection():
                return nn.Sequential(
                    nn.Linear(spectral_dim, feature_width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate))

            if self.use_single_target_regression:
                self.target_spectral_projection = create_spectral_projection()
            elif self.use_dual_target_regression:
                self.glucose_spectral_projection = create_spectral_projection()
                self.spo2_spectral_projection = create_spectral_projection()
            else:
                self.spectral_projection = create_spectral_projection()
            if self.spectral_fusion_mode == 'gate':
                self.spectral_gate_logit = nn.Parameter(torch.tensor(float(spectral_gate_init)))

        def create_backbone(input_channels, enable_deconv=enable_deconv, backbone_name='resnet'):
            layers = []
            backbone_name = str(backbone_name).lower()
            if backbone_name in ['gru', 'bigru', 'ppg_gru']:
                return GRUBackbone1D(
                    input_channels,
                    filter_structure[-1],
                    dropout_rate=dropout_rate)
            if backbone_name in ['lstm', 'bilstm', 'ppg_lstm', 'ppg_bilstm']:
                return LSTMBackbone1D(
                    input_channels,
                    filter_structure[-1],
                    dropout_rate=dropout_rate)
            if backbone_name in ['lstm_attention', 'lstm_attn', 'bilstm_attention', 'ppg_lstm_attention']:
                return LSTMAttentionBackbone1D(
                    input_channels,
                    filter_structure[-1],
                    dropout_rate=dropout_rate)
            if backbone_name in ['transformer', 'ppg_transformer']:
                return TransformerBackbone1D(
                    input_channels,
                    filter_structure[-1],
                    dropout_rate=dropout_rate)
            if backbone_name in ['patchtst', 'patch_tst', 'ppg_patchtst']:
                return PatchTSTBackbone1D(
                    input_channels,
                    filter_structure[-1],
                    seq_len=ppg_seq_len,
                    dropout_rate=dropout_rate,
                    config=ppg_backbone_config)
            if backbone_name in BIOSIGNAL_BACKBONE_NAMES:
                return create_biosignal_backbone(
                    backbone_name,
                    input_channels,
                    filter_structure[-1],
                    dropout_rate=dropout_rate,
                    config=ppg_backbone_config)
            if backbone_name in RECENT_TOP_BACKBONE_NAMES:
                return create_recent_top_backbone(
                    backbone_name,
                    input_channels,
                    filter_structure[-1],
                    seq_len=ppg_seq_len,
                    dropout_rate=dropout_rate,
                    config=ppg_backbone_config)
            if backbone_name in TSL_BACKBONE_MODULES:
                return TimeSeriesLibraryBackbone1D(
                    backbone_name,
                    input_channels,
                    filter_structure[-1],
                    seq_len=ppg_seq_len,
                    dropout_rate=dropout_rate,
                    config=ppg_backbone_config)
            for i, filters in enumerate(filter_structure):
                if backbone_name in ['tcn', 'ppg_tcn']:
                    layers.append(TCN1DBlock(
                        input_channels,
                        filters,
                        dilation=2 ** i,
                        dropout_rate=dropout_rate))
                    input_channels = filters
                    continue
                stride = 1 if i % 2 == 0 else 2
                input_channels = input_channels if i == 0 else filter_structure[i - 1]
                if enable_deconv:
                    layers.append(ResNet2DBlock(input_channels, filters, stride=stride))
                else:
                    layers.append(ResNet1DBlock(input_channels, filters, stride=stride))
            return nn.Sequential(*layers)

        if self.use_baseline_residual_plugin:
            self.backbone_OP = None
            self.backbone_TH = None
        else:
            self.backbone_OP = create_backbone(
                ppg_channels,
                enable_deconv=enable_deconv,
                backbone_name=ppg_backbone_name)
            self.backbone_TH = create_backbone(6, enable_deconv=False)

        self.sigmoid = nn.Sigmoid()
        self.fc_layers = nn.ModuleList()
        self.fusion = None
        self.common_fusion = None
        self.common_encoder = None
        self.glucose_regression_head = None
        self.spo2_regression_head = None
        self.glucose_common_head = None
        self.spo2_common_head = None
        self.glucose_spectral_residual_head = None
        self.spo2_spectral_residual_head = None
        self.target_spectral_residual_head = None
        self.target_spectral_gate_logits = None

        if self.use_target_specific_regression:
            hidden_dims = [int(width) for width in dense_structure[:-1]]
            if not hidden_dims:
                raise ValueError("dense_structure must contain at least one hidden layer")
            target_hidden_dim = hidden_dims[-1]

            def create_small_head(input_dim, head_dropout=dropout_rate):
                return nn.Sequential(
                    nn.Linear(input_dim, target_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(target_hidden_dim, 1))

            if self.use_single_target_regression:
                self.target_spectral_residual_head = create_small_head(
                    self.spectral_feature_dim)
                nn.init.zeros_(self.target_spectral_residual_head[-1].weight)
                nn.init.zeros_(self.target_spectral_residual_head[-1].bias)
                self.target_spectral_gate_logits = nn.Parameter(torch.full(
                    (1,), float(target_spectral_gate_init)))
            elif self.use_baseline_residual_plugin:
                self.glucose_spectral_residual_head = create_small_head(
                    self.spectral_feature_dim)
                self.spo2_spectral_residual_head = create_small_head(
                    self.spectral_feature_dim)
                for residual_head in [
                        self.glucose_spectral_residual_head,
                        self.spo2_spectral_residual_head]:
                    nn.init.zeros_(residual_head[-1].weight)
                    nn.init.zeros_(residual_head[-1].bias)
                self.target_spectral_gate_logits = nn.Parameter(torch.full(
                    (2,), float(target_spectral_gate_init)))
            else:
                if self.fusion_name == 'attention':
                    self.common_fusion = SelfAttention(input_dim=self.base_feature_dim)
                    common_input_dim = self.base_feature_dim * 2
                elif self.fusion_name == 'concat':
                    common_input_dim = self.base_feature_dim
                else:
                    raise ValueError(f"Unsupported fusion method: {fusion_name}")

                common_dims = hidden_dims[:-1]
                common_layers = []
                previous_dim = common_input_dim
                for width in common_dims:
                    common_layers.extend([
                        nn.Linear(previous_dim, width),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)])
                    previous_dim = width
                self.common_encoder = (
                    nn.Sequential(*common_layers) if common_layers else nn.Identity())

                if self.target_specific_regression_mode == 'concat':
                    target_input_dim = previous_dim + self.spectral_feature_dim
                    self.glucose_regression_head = create_small_head(target_input_dim)
                    self.spo2_regression_head = create_small_head(target_input_dim)
                else:
                    self.glucose_common_head = create_small_head(previous_dim)
                    self.spo2_common_head = create_small_head(previous_dim)
                    self.glucose_spectral_residual_head = create_small_head(
                        self.spectral_feature_dim)
                    self.spo2_spectral_residual_head = create_small_head(
                        self.spectral_feature_dim)
                    self.target_spectral_gate_logits = nn.Parameter(torch.full(
                        (2,), float(target_spectral_gate_init)))
        else:
            if self.fusion_name == 'concat':
                self.fc_layers.append(nn.Linear(self.feature_dim, dense_structure[0]))
            elif self.fusion_name == 'attention':
                self.fusion = SelfAttention(input_dim=self.feature_dim)
                self.fc_layers.append(nn.Linear(self.feature_dim * 2, dense_structure[0]))
            else:
                raise ValueError(f"Unsupported fusion method: {fusion_name}")

            for i in range(len(dense_structure) - 1):
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(dropout_rate))
                self.fc_layers.append(nn.Linear(dense_structure[i], dense_structure[i + 1]))

    @staticmethod
    def _pool_context_signal(x):
        if x.ndim == 3:
            return F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x.reshape(x.size(0), -1)

    def _build_context_target(self, x_TH, x_Demo, x_DF, x_MI):
        parts = []
        if "th" in self.context_modalities:
            parts.append(self._pool_context_signal(x_TH) * self.modal_mask[1])
        if "demo" in self.context_modalities:
            parts.append(x_Demo.reshape(x_Demo.size(0), -1) * self.modal_mask[2])
        if "df" in self.context_modalities:
            parts.append(self._pool_context_signal(x_DF) * self.modal_mask[3])
        if "mi" in self.context_modalities:
            parts.append(x_MI.reshape(x_MI.size(0), -1) * self.modal_mask[4])
        if not parts:
            return None
        return torch.cat(parts, dim=1).detach()

    def forward(self, x_OP, x_TH, x_DF, x_Demo, x_MI):
        spectral_hidden = self.spectral_encoder(x_OP)
        expert_tokens = None
        gate_weights = None
        baseline_hat = None
        noise_hat = None
        clean_y_hat = None
        background_spectrum = None
        target_spectrum = None
        target_glucose_spectrum = None
        target_spo2_spectrum = None
        background_coefficients = None
        target_glucose_coefficients = None
        target_spo2_coefficients = None
        spectral_bases = None
        glucose_component_pred = None
        spo2_component_pred = None
        target_component_pred = None
        background_context_pred = None
        target_context_pred = None
        context_target = None
        background_context_film = None
        if self.role_moe is not None:
            if self.enable_background_context_conditioning or self.enable_context_adversarial:
                context_target = self._build_context_target(x_TH, x_Demo, x_DF, x_MI)
            role_outputs = self.role_moe(
                spectral_hidden,
                context=context_target if self.enable_background_context_conditioning else None)
            background_context_film = role_outputs.get("background_context_film")
            baseline_hat = role_outputs["baseline"]
            noise_hat = role_outputs["noise"]
            background_spectrum = role_outputs["background_spectrum"]
            target_spectrum = role_outputs["target_spectrum"]
            target_glucose_spectrum = role_outputs.get("target_glucose_spectrum")
            target_spo2_spectrum = role_outputs.get("target_spo2_spectrum")
            background_coefficients = role_outputs.get("background_coefficients")
            target_glucose_coefficients = role_outputs.get("target_glucose_coefficients")
            target_spo2_coefficients = role_outputs.get("target_spo2_coefficients")
            spectral_bases = role_outputs.get("spectral_bases")
            spectrum = background_spectrum + target_spectrum
            clean_y_hat = torch.einsum("kl,blt->bkt", self.H, spectrum)
            y_hat = clean_y_hat + baseline_hat + noise_hat
            expert_tokens = role_outputs["tokens"]
            gate_weights = role_outputs["gate_weights"]
            if (
                    self.use_single_target_regression and
                    self.target_component_head is not None and
                    target_glucose_spectrum is not None):
                target_component_pred = self.target_component_head(
                    target_glucose_spectrum.mean(dim=-1))
            elif (
                    self.enable_aux_spo2_spectrum and
                    target_glucose_spectrum is not None and
                    target_spo2_spectrum is not None):
                glucose_component_pred = self.glucose_component_head(target_glucose_spectrum.mean(dim=-1))
                spo2_component_pred = self.spo2_component_head(target_spo2_spectrum.mean(dim=-1))
            if self.enable_context_adversarial:
                if context_target is not None:
                    background_repr = background_spectrum.mean(dim=-1)
                    target_repr = target_spectrum.mean(dim=-1)
                    background_context_pred = self.background_context_predictor(background_repr)
                    target_context_pred = self.target_context_predictor(
                        grad_reverse(target_repr, self.context_grl_lambda))
        elif self.token_moe is not None:
            spectral_hidden, expert_tokens, gate_weights = self.token_moe(spectral_hidden)
            spectrum = self.spectral_generator(spectral_hidden)
            if hasattr(self.spectral_generator, "basis"):
                spectral_bases = self.spectral_generator.basis().unsqueeze(0)
            y_hat = torch.einsum("kl,blt->bkt", self.H, spectrum)
            clean_y_hat = y_hat
        else:
            spectrum = self.spectral_generator(spectral_hidden)
            if hasattr(self.spectral_generator, "basis"):
                spectral_bases = self.spectral_generator.basis().unsqueeze(0)
            y_hat = torch.einsum("kl,blt->bkt", self.H, spectrum)
            clean_y_hat = y_hat

        baseline_logits = None
        base_feature_list = None
        if self.use_baseline_residual_plugin:
            baseline_logits = self.baseline_predictor.forward_logits(
                x_OP, x_TH, x_DF, x_Demo, x_MI)
        else:
            x_OP_feature = self.backbone_OP(x_OP)
            x_TH_feature = self.backbone_TH(x_TH)

            if self.enable_deconv:
                x_OP_feature = F.adaptive_avg_pool2d(
                    x_OP_feature, 1).squeeze(-1).squeeze(-1)
            else:
                x_OP_feature = F.adaptive_avg_pool1d(x_OP_feature, 1).squeeze(-1)

            x_TH_feature = F.adaptive_avg_pool1d(x_TH_feature, 1).squeeze(-1)
            x_DF_feature = F.adaptive_avg_pool1d(x_DF, 1).squeeze(-1)

            x_OP_feature = x_OP_feature * self.modal_mask[0]
            x_TH_feature = x_TH_feature * self.modal_mask[1]
            x_Demo_feature = x_Demo * self.modal_mask[2]
            x_DF_feature = x_DF_feature * self.modal_mask[3]
            x_MI_feature = x_MI * self.modal_mask[4]

            base_feature_list = [
                x_OP_feature,
                x_TH_feature,
                x_DF_feature,
                x_Demo_feature,
                x_MI_feature]
        spectral_gate = None
        glucose_spectral_feature = None
        spo2_spectral_feature = None
        target_spectral_feature = None
        common_target_pred = None
        target_spectral_contribution = None
        target_spectral_gates = None
        if self.spectral_fusion_mode != 'none':
            if (
                    self.use_single_target_regression and
                    target_glucose_spectrum is not None):
                target_spectral_feature = self.target_spectral_projection(
                    target_glucose_spectrum.mean(dim=-1))
                target_spectral_feature = (
                    target_spectral_feature * self.modal_mask[0])
                if self.spectral_fusion_mode == 'gate':
                    spectral_gate = torch.sigmoid(self.spectral_gate_logit)
                    target_spectral_feature = (
                        target_spectral_feature * spectral_gate)
            elif (
                    self.use_dual_target_regression and
                    target_glucose_spectrum is not None and
                    target_spo2_spectrum is not None):
                glucose_spectral_feature = self.glucose_spectral_projection(
                    target_glucose_spectrum.mean(dim=-1))
                spo2_spectral_feature = self.spo2_spectral_projection(
                    target_spo2_spectrum.mean(dim=-1))
                glucose_spectral_feature = glucose_spectral_feature * self.modal_mask[0]
                spo2_spectral_feature = spo2_spectral_feature * self.modal_mask[0]
                if self.spectral_fusion_mode == 'gate':
                    spectral_gate = torch.sigmoid(self.spectral_gate_logit)
                    glucose_spectral_feature = glucose_spectral_feature * spectral_gate
                    spo2_spectral_feature = spo2_spectral_feature * spectral_gate
            else:
                regression_spectrum = target_spectrum if target_spectrum is not None else spectrum
                glucose_spectral_feature = self.spectral_projection(
                    regression_spectrum.mean(dim=-1))
                glucose_spectral_feature = glucose_spectral_feature * self.modal_mask[0]
                if self.spectral_fusion_mode == 'gate':
                    spectral_gate = torch.sigmoid(self.spectral_gate_logit)
                    glucose_spectral_feature = glucose_spectral_feature * spectral_gate

        if self.use_single_target_regression:
            if target_spectral_feature is None:
                raise RuntimeError(
                    "Single-target spectral regression requires a primary target spectrum")
            target_delta = self.target_spectral_residual_head(
                target_spectral_feature)
            target_spectral_gates = torch.sigmoid(
                self.target_spectral_gate_logits)
            regression_output = (
                baseline_logits +
                target_spectral_gates[0] * target_delta)
            common_target_pred = self.sigmoid(baseline_logits)
        elif self.use_dual_target_regression:
            if glucose_spectral_feature is None or spo2_spectral_feature is None:
                raise RuntimeError(
                    "Target-specific regression requires both glucose and SpO2 spectra")
            if self.use_baseline_residual_plugin:
                glucose_delta = self.glucose_spectral_residual_head(
                    glucose_spectral_feature)
                spo2_delta = self.spo2_spectral_residual_head(
                    spo2_spectral_feature)
                target_spectral_gates = torch.sigmoid(
                    self.target_spectral_gate_logits)
                regression_output = baseline_logits + torch.cat([
                    target_spectral_gates[0] * glucose_delta,
                    target_spectral_gates[1] * spo2_delta], dim=1)
                common_target_pred = self.sigmoid(baseline_logits)
            else:
                common_features = torch.cat(base_feature_list, dim=1)
                if self.fusion_name == 'attention':
                    attended = self.common_fusion(common_features.unsqueeze(1)).squeeze(1)
                    common_features = torch.cat([common_features, attended], dim=1)
                common_features = self.common_encoder(common_features)
            if self.target_specific_regression_mode == 'concat':
                glucose_output = self.glucose_regression_head(torch.cat(
                    [common_features, glucose_spectral_feature], dim=1))
                spo2_output = self.spo2_regression_head(torch.cat(
                    [common_features, spo2_spectral_feature], dim=1))
                regression_output = torch.cat([glucose_output, spo2_output], dim=1)
            elif not self.use_baseline_residual_plugin:
                glucose_common = self.glucose_common_head(common_features)
                spo2_common = self.spo2_common_head(common_features)
                glucose_delta = self.glucose_spectral_residual_head(
                    glucose_spectral_feature)
                spo2_delta = self.spo2_spectral_residual_head(
                    spo2_spectral_feature)
                target_spectral_gates = torch.sigmoid(
                    self.target_spectral_gate_logits)
                if self.target_specific_regression_mode == 'bounded_gated_residual':
                    glucose_delta = torch.tanh(glucose_delta)
                    spo2_delta = torch.tanh(spo2_delta)
                    target_spectral_gates = (
                        target_spectral_gates * self.target_spectral_gate_max)
                glucose_output = (
                    glucose_common + target_spectral_gates[0] * glucose_delta)
                spo2_output = (
                    spo2_common + target_spectral_gates[1] * spo2_delta)
                common_logits = torch.cat([glucose_common, spo2_common], dim=1)
                common_target_pred = self.sigmoid(common_logits)
                regression_output = torch.cat([glucose_output, spo2_output], dim=1)
        else:
            feature_list = list(base_feature_list)
            if glucose_spectral_feature is not None:
                feature_list.append(glucose_spectral_feature)
            regression_output = torch.cat(feature_list, dim=1)
            if self.fusion_name == 'attention':
                attended = self.fusion(regression_output.unsqueeze(1)).squeeze(1)
                regression_output = torch.cat([regression_output, attended], dim=1)
            for layer in self.fc_layers:
                regression_output = layer(regression_output)

        bg_prediction = self.sigmoid(regression_output)
        if common_target_pred is not None:
            target_spectral_contribution = bg_prediction - common_target_pred

        return {
            "bg": bg_prediction,
            "spectrum": spectrum,
            "y_hat": y_hat,
            "clean_y_hat": clean_y_hat,
            "baseline_hat": baseline_hat,
            "noise_hat": noise_hat,
            "background_spectrum": background_spectrum,
            "target_spectrum": target_spectrum,
            "target_glucose_spectrum": target_glucose_spectrum,
            "target_spo2_spectrum": target_spo2_spectrum,
            "primary_target_spectrum": target_glucose_spectrum,
            "background_coefficients": background_coefficients,
            "target_glucose_coefficients": target_glucose_coefficients,
            "target_spo2_coefficients": target_spo2_coefficients,
            "primary_target_coefficients": target_glucose_coefficients,
            "spectral_bases": spectral_bases,
            "glucose_component_pred": glucose_component_pred,
            "spo2_component_pred": spo2_component_pred,
            "primary_target_component_pred": target_component_pred,
            "background_context_pred": background_context_pred,
            "target_context_pred": target_context_pred,
            "context_target": context_target,
            "background_context_film": background_context_film,
            "common_target_pred": common_target_pred,
            "target_spectral_contribution": target_spectral_contribution,
            "target_spectral_gates": target_spectral_gates,
            "tokens": expert_tokens,
            "gate_weights": gate_weights,
            "spectral_gate": spectral_gate
        }
