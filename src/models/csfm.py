# Adapted from:
# https://github.com/guxiao0822/Cardiac-Sensing-FM/blob/main/network/model.py
# Cardiac-Sensing-FM: multimodal cardiac foundation model for ECG/PPG biosignals.

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = int(heads)
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        batch_size, num_tokens = x.shape[:2]
        q, k, v = [
            tensor.reshape(batch_size, num_tokens, self.heads, -1).transpose(1, 2)
            for tensor in qkv
        ]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots_mask = mask[:, None, :, None] * mask[:, None, None, :]
            dots = dots.masked_fill(dots_mask.bool(), -torch.finfo(dots.dtype).max)
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ])
            for _ in range(int(depth))
        ])

    def forward(self, x, mask=None, return_intermediate=False):
        intermediates = []
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            if return_intermediate:
                intermediates.append(x)
        x = self.norm(x)
        if return_intermediate:
            return x, intermediates
        return x


class CSFM(nn.Module):
    """Official Cardiac-Sensing-FM encoder core with optional text path omitted."""
    def __init__(
            self,
            signal_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool="cls",
            channels=13,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            text_len=64,
            vocab_size=30522):
        super(CSFM, self).__init__()
        signal_size = int(signal_size)
        patch_size = int(patch_size)
        if signal_size % patch_size != 0:
            raise ValueError("signal_size must be divisible by patch_size")
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be either 'cls' or 'mean'")

        self.signal_size = signal_size
        self.patch_size = patch_size
        self.num_patches = signal_size // patch_size
        self.num_channels = int(channels)
        self.encoder_dim = int(dim)
        self.pool = pool

        self.patch_norm = nn.LayerNorm(patch_size)
        self.patch_linear = nn.Linear(patch_size, dim)
        self.patch_out_norm = nn.LayerNorm(dim)

        self.text_to_embedding = nn.Sequential(
            nn.Embedding(vocab_size, dim),
            nn.LayerNorm(dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.ts_channel_type_embedding = nn.Parameter(torch.randn(1, channels, dim))
        self.text_type_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.ts_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, text_len, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def ts_to_patch_embedding(self, ts):
        batch_size, channels, seq_len = ts.shape
        if seq_len != self.signal_size:
            raise ValueError(f"CSFM expected signal length {self.signal_size}, got {seq_len}")
        ts = ts.reshape(batch_size, channels, self.num_patches, self.patch_size)
        ts = self.patch_norm(ts)
        ts = self.patch_linear(ts)
        return self.patch_out_norm(ts)

    def forward_tokens(self, ts, channel, text=None, mask=None):
        ts = self.ts_to_patch_embedding(ts)
        batch_size, channels, num_patches, _ = ts.shape
        channel = torch.as_tensor(channel, dtype=torch.long, device=ts.device)
        if channel.ndim == 0:
            channel = channel.repeat(channels)
        if channel.numel() != channels:
            raise ValueError(f"channel must contain {channels} indices, got {channel.numel()}")

        channel_emb = self.ts_channel_type_embedding[:, channel]
        channel_emb = channel_emb[:, :, None, :].expand(batch_size, channels, num_patches, -1)
        ts = ts + channel_emb

        pos_emb = self.ts_pos_embedding[:, None, :, :].expand(batch_size, channels, num_patches, -1)
        ts = ts + pos_emb
        ts = ts.reshape(batch_size, channels * num_patches, self.encoder_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        if text is not None:
            text = self.text_to_embedding(text)
            text = text + self.text_pos_embedding + self.text_type_embedding
            x = torch.cat((cls_tokens, ts, text), dim=1)
        else:
            x = torch.cat((cls_tokens, ts), dim=1)
        return self.transformer(self.dropout(x), mask=mask)

    def forward(self, ts, channel, text=None, mask=None, task="cls"):
        tokens = self.forward_tokens(ts, channel, text=text, mask=mask)
        if task == "tokens":
            return tokens
        x = tokens.mean(dim=1) if self.pool == "mean" else tokens[:, 0]
        return self.mlp_head(self.to_latent(x))


def _variant_config(variant):
    variant = str(variant).lower()
    if variant == "tiny":
        return {"dim": 768, "depth": 6, "heads": 8, "mlp_dim": 1024}
    if variant == "base":
        return {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072}
    if variant == "large":
        return {"dim": 1024, "depth": 16, "heads": 24, "mlp_dim": 4096}
    raise ValueError(f"Unknown CSFM variant '{variant}'. Use 'Tiny', 'Base', or 'Large'.")


def CSFM_model(variant="Base", **overrides):
    """Build a CSFM variant, matching the official constructor while allowing local overrides."""
    base_args = dict(
        signal_size=2500,
        patch_size=25,
        num_classes=1,
        channels=13,
        dropout=0.1,
        emb_dropout=0.1,
        text_len=64,
        pool="cls")
    base_args.update(_variant_config(variant))
    base_args.update(overrides)
    return CSFM(**base_args)


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
        clean_key = clean_key.replace("encoder.", "")
        if "mlp_head" in clean_key:
            continue
        if clean_key in current and current[clean_key].shape == value.shape:
            matched[clean_key] = value
    module.load_state_dict(matched, strict=False)
    return len(matched), len(current)


class CardiacSensingFMBackbone1D(nn.Module):
    """Official CSFM encoder adapted to this project's PPG backbone API.

    Local input:  (batch, ppg_channels, seq_len), e.g. (B, 6, 300).
    CSFM channel indices: ECG leads are 0-11 and PPG is 12. By default all
    local optical PPG channels are marked as CSFM channel type 12.
    """
    def __init__(
            self,
            input_channels,
            output_channels,
            signal_size=300,
            patch_size=25,
            variant="Tiny",
            hidden_dim=None,
            depth=None,
            heads=None,
            mlp_dim=None,
            dim_head=64,
            channel_index=12,
            dropout_rate=0.10,
            pretrained_path=None,
            freeze_encoder=False):
        super(CardiacSensingFMBackbone1D, self).__init__()
        arch = _variant_config(variant)
        if hidden_dim is not None:
            arch["dim"] = int(hidden_dim)
        if depth is not None:
            arch["depth"] = int(depth)
        if heads is not None:
            arch["heads"] = int(heads)
        if mlp_dim is not None:
            arch["mlp_dim"] = int(mlp_dim)

        self.input_channels = int(input_channels)
        self.signal_size = int(signal_size)
        self.patch_size = int(patch_size)
        if self.signal_size % self.patch_size != 0:
            raise ValueError("ppg_csfm_signal_size must be divisible by ppg_csfm_patch_size")

        self.model = CSFM_model(
            variant=variant,
            signal_size=self.signal_size,
            patch_size=self.patch_size,
            num_classes=arch["dim"],
            dropout=dropout_rate,
            emb_dropout=dropout_rate,
            dim_head=dim_head,
            **arch)
        self.model.mlp_head = nn.Identity()

        channel_index = torch.as_tensor(channel_index if isinstance(channel_index, (list, tuple)) else [channel_index])
        if channel_index.numel() == 1:
            channel_index = channel_index.repeat(self.input_channels)
        if channel_index.numel() != self.input_channels:
            raise ValueError(f"CSFM channel_index must contain 1 or {self.input_channels} values")
        self.register_buffer("channel_index", channel_index.long(), persistent=False)

        self.projection = nn.Sequential(
            nn.Conv1d(arch["dim"], output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.GELU())

        self.pretrained_load_report = None
        if pretrained_path:
            checkpoint_path = Path(pretrained_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"CSFM pretrained weight file not found: {checkpoint_path}")
            self.pretrained_load_report = _load_matching_state_dict(self.model, checkpoint_path)

        if freeze_encoder:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def _resize(self, x):
        if x.size(-1) != self.signal_size:
            x = F.interpolate(x, size=self.signal_size, mode="linear", align_corners=False)
        return x

    def forward(self, x):
        x = self._resize(x)
        tokens = self.model.forward_tokens(x, self.channel_index)
        ts_tokens = tokens[:, 1:, :]
        return self.projection(ts_tokens.transpose(1, 2))
