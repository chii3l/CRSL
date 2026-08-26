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


class MultiModalModel(nn.Module):
    def __init__(
            self,
            filter_structure,
            dense_structure,
            dropout_rate,
            fusion_name='concat',
            enable_deconv=False,
            ppg_backbone_name='resnet',
            ppg_backbone_config=None,
            ppg_seq_len=300,
            ppg_input_channels=6,
            active_modalities=None):
        super(MultiModalModel, self).__init__()
        
        self.fusion_name = fusion_name
        self.enable_deconv = enable_deconv
        if active_modalities is None:
            active_modalities = [1, 1, 1, 1, 1]
        if len(active_modalities) != 5:
            raise ValueError("active_modalities must contain 5 values: PPG, TH, Demo, DF, MI")
        self.register_buffer("modal_mask", torch.as_tensor(active_modalities, dtype=torch.float32), persistent=False)
        
        self.feature_dim = filter_structure[-1] * 2 + 6 + 9 + 1
        
        # 根据奇偶轮数动态设置 stride
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

        # OP 和 TH 使用相同的逻辑
        self.backbone_OP = create_backbone(
            int(ppg_input_channels),
            enable_deconv=enable_deconv,
            backbone_name=ppg_backbone_name)
        self.backbone_TH = create_backbone(6, enable_deconv=False)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        if self.fusion_name == 'concat':
            self.fc_layers.append(nn.Linear(self.feature_dim * 1, dense_structure[0]))
        elif self.fusion_name == 'attention':
            self.fc_layers.append(nn.Linear(self.feature_dim * 2, dense_structure[0]))
        for i in range(len(dense_structure) - 1):
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))
            self.fc_layers.append(nn.Linear(dense_structure[i], dense_structure[i + 1]))

        self.output_layer = nn.Linear(dense_structure[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Fusion and attention mechanism
        attention_input_dim = self.feature_dim
        if self.fusion_name == 'concat':
            self.fusion = lambda x: torch.cat(x, dim=1)  # 拼接
        elif self.fusion_name == 'attention':
            self.fusion = SelfAttention(input_dim=attention_input_dim)  # 自注意力
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_name}")

    def forward_logits(self, x_OP, x_TH, x_DF, x_Demo, x_MI):
        x_OP = self.backbone_OP(x_OP)
        x_TH = self.backbone_TH(x_TH)

        # Global pooling
        if self.enable_deconv:
            x_OP = F.adaptive_avg_pool2d(x_OP, 1).squeeze(-1).squeeze(-1)
        else:
            x_OP = F.adaptive_avg_pool1d(x_OP, 1).squeeze(-1)
            
        x_TH = F.adaptive_avg_pool1d(x_TH, 1).squeeze(-1)
        x_DF = F.adaptive_avg_pool1d(x_DF, 1).squeeze(-1)

        x_OP = x_OP * self.modal_mask[0]
        x_TH = x_TH * self.modal_mask[1]
        x_Demo = x_Demo * self.modal_mask[2]
        x_DF = x_DF * self.modal_mask[3]
        x_MI = x_MI * self.modal_mask[4]

        # Concatenate features
        combined_features = torch.cat([x_OP, x_TH, x_DF, x_Demo, x_MI], dim=1)
        
        if self.fusion_name == 'attention':
            # Apply attention if needed
            combined_features_att = self.fusion(combined_features.unsqueeze(1)).squeeze(1)

            combined_features = torch.cat([combined_features, combined_features_att], dim=1)
        elif self.fusion_name == 'concate':
            pass

        # Fully connected layers
        for layer in self.fc_layers:
            combined_features = layer(combined_features)

        return combined_features

    def forward(self, x_OP, x_TH, x_DF, x_Demo, x_MI):
        return self.sigmoid(self.forward_logits(x_OP, x_TH, x_DF, x_Demo, x_MI))
