# Copyright 2024 Nokia
# Licensed under the BSD 3-Clause Clear License.
# Adapted from:
# https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/models/resnet.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """Conv1d with TensorFlow-style SAME padding, matching the PaPaGei source."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=int(groups))

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad // 2
        pad_right = pad - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """MaxPool1d with SAME padding, matching the PaPaGei source."""
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad // 2
        pad_right = pad - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        return self.max_pool(x)


class BasicBlock(nn.Module):
    """PaPaGei/ResNet1D basic residual block."""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            downsample,
            use_bn,
            use_do,
            is_first_block=False,
            dropout_rate=0.5):
        super(BasicBlock, self).__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if downsample else 1
        self.groups = int(groups)
        self.downsample = bool(downsample)
        self.is_first_block = bool(is_first_block)
        self.use_bn = bool(use_bn)
        self.use_do = bool(use_do)

        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=float(dropout_rate))
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=float(dropout_rate))
        self.conv2 = MyConv1dPadSame(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=max(1, self.stride))

    def forward(self, x):
        identity = x
        out = x

        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            pad_left = (self.out_channels - self.in_channels) // 2
            pad_right = self.out_channels - self.in_channels - pad_left
            identity = F.pad(identity, (pad_left, pad_right), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class ResNet1D(nn.Module):
    """Official PaPaGei ResNet1D encoder with a dense embedding head."""
    def __init__(
            self,
            in_channels,
            base_filters,
            kernel_size,
            stride,
            groups,
            n_block,
            n_classes,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
            use_mt_regression=False,
            use_projection=False,
            dropout_rate=0.5):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = int(n_block)
        self.use_bn = bool(use_bn)
        self.use_do = bool(use_do)
        self.use_mt_regression = bool(use_mt_regression)
        self.use_projection = bool(use_projection)
        self.downsample_gap = int(downsample_gap)
        self.increasefilter_gap = int(increasefilter_gap)

        base_filters = int(base_filters)
        kernel_size = int(kernel_size)
        stride = int(stride)
        groups = int(groups)
        n_classes = int(n_classes)

        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=kernel_size,
            stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()

        self.basicblock_list = nn.ModuleList()
        out_channels = base_filters
        for i_block in range(self.n_block):
            is_first_block = i_block == 0
            downsample = i_block % self.downsample_gap == 1
            if is_first_block:
                in_block_channels = base_filters
                out_channels = in_block_channels
            else:
                in_block_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if i_block % self.increasefilter_gap == 0:
                    out_channels = in_block_channels * 2
                else:
                    out_channels = in_block_channels

            self.basicblock_list.append(BasicBlock(
                in_channels=in_block_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
                dropout_rate=dropout_rate))

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels, n_classes)

        if self.use_projection:
            self.projector = nn.Sequential(
                nn.Linear(out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128))

        if self.use_mt_regression:
            self.mt_regression = nn.Sequential(
                nn.Linear(n_classes, n_classes // 2),
                nn.BatchNorm1d(n_classes // 2),
                nn.Linear(n_classes // 2, n_classes // 4),
                nn.BatchNorm1d(n_classes // 4),
                nn.Linear(n_classes // 4, 1))

    def forward_features(self, x):
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        for block in self.basicblock_list:
            out = block(out)

        if self.use_bn:
            out = self.final_bn(out)
        return self.final_relu(out)

    def forward(self, x):
        features = self.forward_features(x)
        pooled = features.mean(-1)

        if self.use_projection:
            out = self.projector(pooled)
        else:
            out = self.dense(pooled)

        if self.use_mt_regression:
            out_regression = self.mt_regression(pooled)
            return out, out_regression, pooled
        return out, pooled


class ResNet1DMoE(nn.Module):
    """Official PaPaGei-S ResNet1D with two MoE morphology heads."""
    def __init__(
            self,
            in_channels,
            base_filters,
            kernel_size,
            stride,
            groups,
            n_block,
            n_classes,
            n_experts=3,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
            use_projection=False,
            dropout_rate=0.5):
        super(ResNet1DMoE, self).__init__()

        self.verbose = verbose
        self.n_experts = int(n_experts)
        self.use_projection = bool(use_projection)
        self.encoder = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            n_block=n_block,
            n_classes=n_classes,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_bn=use_bn,
            use_do=use_do,
            verbose=verbose,
            use_projection=use_projection,
            dropout_rate=dropout_rate)

        feature_dim = self.encoder.dense.in_features
        self.expert_layers_1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1))
            for _ in range(self.n_experts)
        ])
        self.gating_network_1 = nn.Sequential(
            nn.Linear(feature_dim, self.n_experts),
            nn.Softmax(dim=1))

        self.expert_layers_2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim // 2, 1))
            for _ in range(self.n_experts)
        ])
        self.gating_network_2 = nn.Sequential(
            nn.Linear(feature_dim, self.n_experts),
            nn.Softmax(dim=1))

    def forward(self, x):
        features = self.encoder.forward_features(x)
        pooled = features.mean(-1)

        if self.use_projection:
            out_class = self.encoder.projector(pooled)
        else:
            out_class = self.encoder.dense(pooled)

        expert_outputs_1 = torch.stack([expert(pooled) for expert in self.expert_layers_1], dim=1)
        gate_weights_1 = self.gating_network_1(pooled)
        out_moe1 = torch.sum(gate_weights_1.unsqueeze(2) * expert_outputs_1, dim=1)

        expert_outputs_2 = torch.stack([expert(pooled) for expert in self.expert_layers_2], dim=1)
        gate_weights_2 = self.gating_network_2(pooled)
        out_moe2 = torch.sum(gate_weights_2.unsqueeze(2) * expert_outputs_2, dim=1)

        return out_class, out_moe1, out_moe2, pooled


class ResNet1DBackBone(nn.Module):
    """Official PaPaGei temporal backbone without the final embedding head."""
    def __init__(
            self,
            in_channels,
            base_filters,
            kernel_size,
            stride,
            groups,
            n_block,
            n_classes=None,
            downsample_gap=2,
            increasefilter_gap=4,
            use_bn=True,
            use_do=True,
            verbose=False,
            dropout_rate=0.5):
        super(ResNet1DBackBone, self).__init__()
        self.encoder = ResNet1D(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            n_block=n_block,
            n_classes=n_classes or 512,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_bn=use_bn,
            use_do=use_do,
            verbose=verbose,
            dropout_rate=dropout_rate)

    def forward(self, x):
        return self.encoder.forward_features(x)


def _load_matching_state_dict(module, checkpoint_path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    current = module.state_dict()
    matched = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in current and current[clean_key].shape == value.shape:
            matched[clean_key] = value
            continue

        prefixed_key = f"encoder.{clean_key}"
        if prefixed_key in current and current[prefixed_key].shape == value.shape:
            matched[prefixed_key] = value
            continue

        if clean_key.startswith("encoder.") and clean_key[8:] in current:
            stripped_key = clean_key[8:]
            if current[stripped_key].shape == value.shape:
                matched[stripped_key] = value

    module.load_state_dict(matched, strict=False)
    return len(matched), len(current)


class PaPaGeiBackbone1D(nn.Module):
    """Official PaPaGei encoder adapted to this project's PPG feature interface.

    Input:  (batch, ppg_channels, seq_len), e.g. (B, 6, 300).
    Output: (batch, output_channels, 1), so the existing global pooling and H
    plug-in paths can consume it as a normal 1D backbone.
    """
    def __init__(
            self,
            input_channels,
            output_channels,
            base_filters=32,
            kernel_size=3,
            stride=2,
            groups=1,
            n_block=18,
            n_classes=512,
            n_experts=3,
            variant="s",
            channel_mode="direct",
            target_length=0,
            dropout_rate=0.10,
            pretrained_path=None,
            freeze_encoder=False):
        super(PaPaGeiBackbone1D, self).__init__()

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.n_classes = int(n_classes)
        self.variant = str(variant).lower()
        self.channel_mode = str(channel_mode).lower()
        self.target_length = int(target_length or 0)

        if self.variant not in ["s", "moe", "p", "resnet"]:
            raise ValueError("PaPaGei variant must be one of: s, moe, p, resnet")
        if self.channel_mode not in ["direct", "shared"]:
            raise ValueError("PaPaGei channel_mode must be one of: direct, shared")

        encoder_in_channels = 1 if self.channel_mode == "shared" else self.input_channels
        encoder_cls = ResNet1DMoE if self.variant in ["s", "moe"] else ResNet1D
        encoder_kwargs = dict(
            in_channels=encoder_in_channels,
            base_filters=int(base_filters),
            kernel_size=int(kernel_size),
            stride=int(stride),
            groups=int(groups),
            n_block=int(n_block),
            n_classes=self.n_classes,
            use_bn=True,
            use_do=True,
            dropout_rate=float(dropout_rate))
        if encoder_cls is ResNet1DMoE:
            encoder_kwargs["n_experts"] = int(n_experts)
        self.encoder = encoder_cls(**encoder_kwargs)

        self.channel_gate = None
        if self.channel_mode == "shared" and self.input_channels > 1:
            self.channel_gate = nn.Linear(self.n_classes, 1)

        self.projection = nn.Sequential(
            nn.Linear(self.n_classes, self.output_channels),
            nn.LayerNorm(self.output_channels),
            nn.GELU(),
            nn.Dropout(float(dropout_rate)))

        self.pretrained_load_report = None
        if pretrained_path:
            checkpoint_path = Path(pretrained_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"PaPaGei pretrained weight file not found: {checkpoint_path}")
            self.pretrained_load_report = _load_matching_state_dict(self.encoder, checkpoint_path)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _maybe_resize(self, x):
        if self.target_length > 0 and x.size(-1) != self.target_length:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        return x

    def _encode(self, x):
        outputs = self.encoder(x)
        return outputs[0]

    def forward(self, x):
        x = self._maybe_resize(x)

        if self.channel_mode == "shared" and self.input_channels > 1:
            batch_size, channels, seq_len = x.shape
            x = x.reshape(batch_size * channels, 1, seq_len)
            embeddings = self._encode(x).reshape(batch_size, channels, self.n_classes)
            gate = torch.softmax(self.channel_gate(embeddings).squeeze(-1), dim=1)
            embeddings = torch.sum(embeddings * gate.unsqueeze(-1), dim=1)
        else:
            embeddings = self._encode(x)

        return self.projection(embeddings).unsqueeze(-1)
