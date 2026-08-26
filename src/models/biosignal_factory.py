from .common import _get_config_value
from .eegnet import EEGNetBackbone1D
from .shallowconvnet import ShallowConvNetBackbone1D
from .deepconvnet import DeepConvNetBackbone1D
from .atcnet import ATCNetBackbone1D
from .eegconformer import EEGConformerBackbone1D
from .ecgresnet import ECGResNetBackbone1D


BIOSIGNAL_BACKBONE_NAMES = {
    'eegnet',
    'ppg_eegnet',
    'shallowconvnet',
    'shallow_convnet',
    'shallowfbcspnet',
    'deepconvnet',
    'deep_convnet',
    'deep4net',
    'atcnet',
    'eeg_tcnet',
    'eegconformer',
    'eeg_conformer',
    'ecgresnet',
    'ecg_resnet',
    'ecg_resnet1d',
    'ppg_ecgresnet',
}


def create_biosignal_backbone(backbone_name, input_channels, output_channels, dropout_rate=0.25, config=None):
    backbone_name = str(backbone_name).lower()
    hidden_channels = int(_get_config_value(config, 'ppg_bio_hidden', 96))
    num_layers = int(_get_config_value(config, 'ppg_bio_layers', 3))
    num_heads = int(_get_config_value(config, 'ppg_bio_heads', 4))
    temporal_kernel = int(_get_config_value(config, 'ppg_bio_kernel_size', 25))

    if backbone_name in ['eegnet', 'ppg_eegnet']:
        return EEGNetBackbone1D(
            input_channels,
            output_channels,
            temporal_filters=int(_get_config_value(config, 'ppg_bio_temporal_filters', 16)),
            depth_multiplier=int(_get_config_value(config, 'ppg_bio_depth_multiplier', 2)),
            separable_filters=int(_get_config_value(config, 'ppg_bio_separable_filters', max(hidden_channels, 64))),
            temporal_kernel=int(_get_config_value(config, 'ppg_bio_kernel_size', 64)),
            separable_kernel=int(_get_config_value(config, 'ppg_bio_separable_kernel', 16)),
            dropout_rate=dropout_rate)
    if backbone_name in ['shallowconvnet', 'shallow_convnet', 'shallowfbcspnet']:
        return ShallowConvNetBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=hidden_channels,
            temporal_kernel=temporal_kernel,
            pool_size=int(_get_config_value(config, 'ppg_bio_pool_size', 20)),
            pool_stride=int(_get_config_value(config, 'ppg_bio_pool_stride', 5)),
            dropout_rate=dropout_rate)
    if backbone_name in ['deepconvnet', 'deep_convnet', 'deep4net']:
        return DeepConvNetBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            temporal_kernel=int(_get_config_value(config, 'ppg_bio_kernel_size', 7)),
            dropout_rate=dropout_rate)
    if backbone_name in ['atcnet', 'eeg_tcnet']:
        return ATCNetBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            temporal_kernel=int(_get_config_value(config, 'ppg_bio_kernel_size', 15)),
            dropout_rate=dropout_rate)
    if backbone_name in ['eegconformer', 'eeg_conformer']:
        return EEGConformerBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            temporal_kernel=temporal_kernel,
            dropout_rate=dropout_rate)
    if backbone_name in ['ecgresnet', 'ecg_resnet', 'ecg_resnet1d', 'ppg_ecgresnet']:
        return ECGResNetBackbone1D(
            input_channels,
            output_channels,
            hidden_channels=int(_get_config_value(config, 'ppg_bio_hidden', 64)),
            num_layers=int(_get_config_value(config, 'ppg_bio_layers', 5)),
            dropout_rate=dropout_rate)
    raise ValueError(f"Unsupported biosignal backbone: {backbone_name}")


