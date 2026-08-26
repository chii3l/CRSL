import importlib
import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


TSL_BACKBONE_MODULES = {
    'patchtst': 'PatchTST',
    'patch_tst': 'PatchTST',
    'timesnet': 'TimesNet',
    'times_net': 'TimesNet',
    'itransformer': 'iTransformer',
    'i_transformer': 'iTransformer',
    'timemixer': 'TimeMixer',
    'time_mixer': 'TimeMixer',
    'timefilter': 'TimeFilter',
    'time_filter': 'TimeFilter',
    'wpmixer': 'WPMixer',
    'wp_mixer': 'WPMixer',
    'fedformer': 'FEDformer',
    'fed_former': 'FEDformer',
}

TSL_FORECAST_BACKBONES = {
    'FEDformer',
    'WPMixer',
}


def _prepare_timeseries_library_imports():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tsl_root = os.path.join(repo_root, 'external_baselines', 'Time-Series-Library')
    if not os.path.isdir(tsl_root):
        raise ImportError(f"Time-Series-Library is not found: {tsl_root}")
    if tsl_root not in sys.path:
        sys.path.insert(0, tsl_root)

    # PatchTST/iTransformer import optional Reformer/einops symbols even when
    # the selected FullAttention path does not use them.
    has_reformer = 'reformer_pytorch' in sys.modules
    if not has_reformer:
        try:
            has_reformer = importlib.util.find_spec('reformer_pytorch') is not None
        except ValueError:
            has_reformer = False
    if not has_reformer:
        reformer_stub = types.ModuleType('reformer_pytorch')

        class LSHSelfAttention(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x, *args, **kwargs):
                return x

        reformer_stub.LSHSelfAttention = LSHSelfAttention
        sys.modules['reformer_pytorch'] = reformer_stub

    has_einops = 'einops' in sys.modules
    if not has_einops:
        try:
            has_einops = importlib.util.find_spec('einops') is not None
        except ValueError:
            has_einops = False
    if not has_einops:
        einops_stub = types.ModuleType('einops')
        einops_layers_stub = types.ModuleType('einops.layers')
        einops_layers_torch_stub = types.ModuleType('einops.layers.torch')

        def rearrange(x, pattern, **axes_lengths):
            pattern = " ".join(str(pattern).split())
            if pattern in ["b l d -> b d l", "b d l -> b l d"]:
                return x.permute(0, 2, 1).contiguous()
            if pattern == "(b l) dstate -> b dstate l":
                l = int(axes_lengths["l"])
                b = x.shape[0] // l
                return x.reshape(b, l, x.shape[-1]).permute(0, 2, 1).contiguous()
            if pattern == "b n (h d) -> b h n d":
                h = int(axes_lengths["h"])
                b, n, hd = x.shape
                return x.reshape(b, n, h, hd // h).permute(0, 2, 1, 3).contiguous()
            if pattern == "b h n d -> b n (h d)":
                b, h, n, d = x.shape
                return x.permute(0, 2, 1, 3).contiguous().reshape(b, n, h * d)
            if pattern == "b e (h) (w) -> b (h w) e":
                b, e, h, w = x.shape
                return x.permute(0, 2, 3, 1).contiguous().reshape(b, h * w, e)
            raise ImportError(f'einops is required for pattern: {pattern}')

        def repeat(x, pattern, **axes_lengths):
            pattern = " ".join(str(pattern).split())
            if pattern == "n -> d n":
                return x.unsqueeze(0).expand(int(axes_lengths["d"]), -1).contiguous()
            if pattern == "d -> b d":
                return x.unsqueeze(0).expand(int(axes_lengths["b"]), -1).contiguous()
            raise ImportError(f'einops is required for pattern: {pattern}')

        def reduce(x, pattern, reduction, **axes_lengths):
            raise ImportError(f'einops is required for reduce pattern: {pattern}')

        def einsum(*args):
            if len(args) < 2:
                raise TypeError("einsum expects tensors followed by a pattern")
            *tensors, pattern = args
            pattern = " ".join(str(pattern).split())
            equation_map = {
                "b l d, d n -> b l d n": "bld,dn->bldn",
                "b l d, b l n, b l d -> b l d n": "bld,bln,bld->bldn",
                "b d n, b n -> b d": "bdn,bn->bd",
            }
            equation = equation_map.get(pattern)
            if equation is None:
                raise ImportError(f'einops is required for einsum pattern: {pattern}')
            return torch.einsum(equation, *tensors)

        class Rearrange(nn.Module):
            def __init__(self, pattern, **axes_lengths):
                super().__init__()
                self.pattern = pattern
                self.axes_lengths = axes_lengths

            def forward(self, x):
                return rearrange(x, self.pattern, **self.axes_lengths)

        einops_stub.rearrange = rearrange
        einops_stub.repeat = repeat
        einops_stub.reduce = reduce
        einops_stub.einsum = einsum
        einops_layers_torch_stub.Rearrange = Rearrange
        sys.modules['einops'] = einops_stub
        sys.modules['einops.layers'] = einops_layers_stub
        sys.modules['einops.layers.torch'] = einops_layers_torch_stub


def _get_config_value(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class TimeSeriesLibraryBackbone1D(nn.Module):
    def __init__(
            self,
            backbone_name,
            input_channels,
            output_channels,
            seq_len=300,
            dropout_rate=0.0,
            config=None):
        super(TimeSeriesLibraryBackbone1D, self).__init__()
        self.backbone_name = str(backbone_name).lower()
        self.module_name = TSL_BACKBONE_MODULES[self.backbone_name]
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.seq_len = int(seq_len)
        self.is_forecast_backbone = self.module_name in TSL_FORECAST_BACKBONES

        _prepare_timeseries_library_imports()
        module = importlib.import_module(f'models.{self.module_name}')
        model_cls = module.Model
        tsl_config = self._build_tsl_config(config, dropout_rate)
        if self.module_name == 'FEDformer':
            self.model = model_cls(
                tsl_config,
                version=_get_config_value(config, 'ppg_tsl_version', 'fourier'),
                mode_select=_get_config_value(config, 'ppg_tsl_mode_select', 'random'),
                modes=int(_get_config_value(config, 'ppg_tsl_modes', 16)))
        else:
            self.model = model_cls(tsl_config)
        self.output_projection = nn.Linear(input_channels, output_channels) if self.is_forecast_backbone else None

    def _build_tsl_config(self, config, dropout_rate):
        d_model = int(_get_config_value(config, 'ppg_tsl_d_model', min(self.output_channels, 64)))
        d_model = max(d_model, 16)
        n_heads = int(_get_config_value(config, 'ppg_tsl_n_heads', 4))
        if d_model % n_heads != 0:
            n_heads = 1
        e_layers = int(_get_config_value(config, 'ppg_tsl_e_layers', 1))
        d_ff = int(_get_config_value(config, 'ppg_tsl_d_ff', max(d_model * 2, 64)))
        patch_len = int(_get_config_value(config, 'ppg_patch_len', 16))
        patch_stride = int(_get_config_value(config, 'ppg_patch_stride', max(1, patch_len // 2)))
        task_name = 'long_term_forecast' if self.is_forecast_backbone else 'classification'
        pred_len = int(_get_config_value(config, 'ppg_tsl_pred_len', self.seq_len)) if self.is_forecast_backbone else 0

        return SimpleNamespace(
            task_name=task_name,
            seq_len=self.seq_len,
            label_len=max(1, self.seq_len // 2),
            pred_len=pred_len,
            enc_in=self.input_channels,
            dec_in=self.input_channels,
            c_out=self.input_channels,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            d_layers=int(_get_config_value(config, 'ppg_tsl_d_layers', 1)),
            n_heads=n_heads,
            dropout=dropout_rate,
            embed=_get_config_value(config, 'ppg_tsl_embed', 'fixed'),
            freq=_get_config_value(config, 'ppg_tsl_freq', 'h'),
            factor=int(_get_config_value(config, 'ppg_tsl_factor', 5)),
            activation=_get_config_value(config, 'ppg_tsl_activation', 'gelu'),
            num_class=self.output_channels,
            top_k=int(_get_config_value(config, 'ppg_tsl_top_k', 3)),
            num_kernels=int(_get_config_value(config, 'ppg_tsl_num_kernels', 3)),
            moving_avg=int(_get_config_value(config, 'ppg_tsl_moving_avg', 25)),
            down_sampling_window=int(_get_config_value(config, 'ppg_tsl_down_sampling_window', 2)),
            down_sampling_layers=int(_get_config_value(config, 'ppg_tsl_down_sampling_layers', 1)),
            down_sampling_method=_get_config_value(config, 'ppg_tsl_down_sampling_method', 'avg'),
            decomp_method=_get_config_value(config, 'ppg_tsl_decomp_method', 'moving_avg'),
            channel_independence=int(_get_config_value(config, 'ppg_tsl_channel_independence', 0)),
            use_norm=int(_get_config_value(config, 'ppg_tsl_use_norm', 1)),
            patch_len=patch_len,
            alpha=_get_config_value(config, 'ppg_tsl_alpha', None),
            top_p=_get_config_value(config, 'ppg_tsl_top_p', None),
            pos=bool(_get_config_value(config, 'ppg_tsl_pos', True)),
            batch_size=int(_get_config_value(config, 'batch_size', 1)),
            device=torch.device('cpu'),
            use_amp=bool(_get_config_value(config, 'ppg_tsl_use_amp', False)),
            stride=patch_stride,
        )

    def forward(self, x):
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        x_seq = x.transpose(1, 2).contiguous()

        if self.is_forecast_backbone:
            output = self.model(x_seq, None, None, None)
            if output.shape[1] != self.seq_len:
                output = F.interpolate(
                    output.transpose(1, 2),
                    size=self.seq_len,
                    mode='linear',
                    align_corners=False).transpose(1, 2)
            output = self.output_projection(output)
            return output.transpose(1, 2)

        padding_mask = torch.ones(x_seq.size(0), x_seq.size(1), device=x_seq.device)
        output = self.model(x_seq, padding_mask, None, None)
        if output.dim() == 3:
            output = output.mean(dim=1)
        return output.unsqueeze(-1)

