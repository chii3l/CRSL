from .common import _get_config_value, _parse_int_list
from .papagei import PaPaGeiBackbone1D
from .csfm import CardiacSensingFMBackbone1D
from .medformer import MedformerBackbone1D
from .tslanet import TSLANetBackbone1D
from .moment import MOMENTBackbone1D


RECENT_TOP_BACKBONE_NAMES = {
    'papagei',
    'ppg_papagei',
    'csfm',
    'cardiac_sensing_fm',
    'cardiac-sensing-fm',
    'cardiacfm',
    'medformer',
    'ppg_medformer',
    'tslanet',
    'ppg_tslanet',
    'moment',
    'ppg_moment',
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ['1', 'true', 'yes', 'y']
    return bool(value)


def create_recent_top_backbone(
        backbone_name,
        input_channels,
        output_channels,
        dropout_rate=0.10,
        config=None,
        seq_len=None):
    backbone_name = str(backbone_name).lower()
    hidden_channels = int(_get_config_value(config, 'ppg_recent_hidden', 128))
    num_layers = int(_get_config_value(config, 'ppg_recent_layers', 3))
    num_heads = int(_get_config_value(config, 'ppg_recent_heads', 4))
    patch_len = int(_get_config_value(config, 'ppg_recent_patch_len', 16))
    patch_stride = int(_get_config_value(config, 'ppg_recent_patch_stride', max(1, patch_len // 2)))
    kernel_size = int(_get_config_value(config, 'ppg_recent_kernel_size', 15))

    if backbone_name in ['papagei', 'ppg_papagei']:
        return PaPaGeiBackbone1D(
            input_channels,
            output_channels,
            base_filters=int(_get_config_value(config, 'ppg_papagei_base_filters', 32)),
            kernel_size=int(_get_config_value(config, 'ppg_papagei_kernel_size', 3)),
            stride=int(_get_config_value(config, 'ppg_papagei_stride', 2)),
            groups=int(_get_config_value(config, 'ppg_papagei_groups', 1)),
            n_block=int(_get_config_value(config, 'ppg_papagei_blocks', 18)),
            n_classes=int(_get_config_value(config, 'ppg_papagei_embedding_dim', 512)),
            n_experts=int(_get_config_value(config, 'ppg_papagei_experts', 3)),
            variant=_get_config_value(config, 'ppg_papagei_variant', 's'),
            channel_mode=_get_config_value(config, 'ppg_papagei_channel_mode', 'direct'),
            target_length=int(_get_config_value(config, 'ppg_papagei_target_len', 0)),
            dropout_rate=dropout_rate,
            pretrained_path=_get_config_value(config, 'ppg_papagei_weight_path', None),
            freeze_encoder=_as_bool(_get_config_value(config, 'ppg_papagei_freeze_encoder', False)))
    if backbone_name in ['csfm', 'cardiac_sensing_fm', 'cardiac-sensing-fm', 'cardiacfm']:
        return CardiacSensingFMBackbone1D(
            input_channels,
            output_channels,
            signal_size=int(_get_config_value(config, 'ppg_csfm_signal_size', seq_len or 300)),
            patch_size=int(_get_config_value(config, 'ppg_csfm_patch_size', 25)),
            variant=_get_config_value(config, 'ppg_csfm_variant', 'Tiny'),
            hidden_dim=_get_config_value(config, 'ppg_csfm_hidden_dim', None),
            depth=_get_config_value(config, 'ppg_csfm_depth', None),
            heads=_get_config_value(config, 'ppg_csfm_heads', None),
            mlp_dim=_get_config_value(config, 'ppg_csfm_mlp_dim', None),
            dim_head=int(_get_config_value(config, 'ppg_csfm_dim_head', 64)),
            channel_index=_get_config_value(config, 'ppg_csfm_channel_index', 12),
            dropout_rate=dropout_rate,
            pretrained_path=_get_config_value(config, 'ppg_csfm_weight_path', None),
            freeze_encoder=_as_bool(_get_config_value(config, 'ppg_csfm_freeze_encoder', False)))
    if backbone_name in ['medformer', 'ppg_medformer']:
        return MedformerBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=_parse_int_list(_get_config_value(config, 'ppg_recent_patch_sizes', None), (8, 16, 32)),
            dropout_rate=dropout_rate)
    if backbone_name in ['tslanet', 'ppg_tslanet']:
        return TSLANetBackbone1D(
            input_channels,
            output_channels,
            seq_len=int(seq_len or _get_config_value(config, 'ppg_seq_len', 300)),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mlp_ratio=float(_get_config_value(config, 'ppg_tslanet_mlp_ratio', 3.0)),
            use_icb=_as_bool(_get_config_value(config, 'ppg_tslanet_icb', True)),
            use_asb=_as_bool(_get_config_value(config, 'ppg_tslanet_asb', True)),
            adaptive_filter=_as_bool(_get_config_value(config, 'ppg_tslanet_adaptive_filter', True)),
            dropout_rate=dropout_rate)
    if backbone_name in ['moment', 'ppg_moment']:
        return MOMENTBackbone1D(
            input_channels,
            output_channels,
            seq_len=int(seq_len or _get_config_value(config, 'ppg_seq_len', 300)),
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_ff=int(_get_config_value(config, 'ppg_moment_d_ff', hidden_channels * 4)),
            dropout_rate=dropout_rate,
            patch_dropout=float(_get_config_value(config, 'ppg_moment_patch_dropout', dropout_rate)),
            revin_affine=_as_bool(_get_config_value(config, 'ppg_moment_revin_affine', False)),
            add_positional_embedding=_as_bool(_get_config_value(config, 'ppg_moment_add_positional_embedding', True)),
            value_embedding_bias=_as_bool(_get_config_value(config, 'ppg_moment_value_embedding_bias', False)),
            orth_gain=float(_get_config_value(config, 'ppg_moment_orth_gain', 1.41)),
            mask_ratio=float(_get_config_value(config, 'ppg_moment_mask_ratio', 0.0)),
            channel_reduction=_get_config_value(config, 'ppg_moment_channel_reduction', 'mean'),
            pad_end=_as_bool(_get_config_value(config, 'ppg_moment_pad_end', True)))
    raise ValueError(f"Unsupported recent top backbone: {backbone_name}")
