"""DINSNet model definition."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainInvariantFeatureNorm(nn.Module):
    """Batch-independent adaptive normalization with running stats for inference."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-5,
        reduction: int = 8,
        momentum: float = 0.1,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.affine = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )
        self.base_gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.base_beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _update_running_stats(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        with torch.no_grad():
            self.running_mean.mul_(1.0 - self.momentum).add_(mean * self.momentum)
            self.running_var.mul_(1.0 - self.momentum).add_(var * self.momentum)
            self.num_batches_tracked.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std_in = (var_in + self.eps).sqrt()
        x_in = (x - mean_in) / std_in

        if self.track_running_stats:
            if self.training:
                # Update running stats like BatchNorm, but keep instance stats too.
                mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
                var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
                self._update_running_stats(mean_bn, var_bn)
            else:
                mean_bn = self.running_mean
                var_bn = self.running_var
        else:
            mean_bn = mean_in
            var_bn = var_in

        x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()

        style_descriptor = torch.cat(
            [
                mean_in.view(batch, channels),
                std_in.view(batch, channels),
            ],
            dim=1,
        )
        mix_gate = self.gate(style_descriptor).view(batch, channels, 1, 1)
        mixed = mix_gate * x_in + (1.0 - mix_gate) * x_bn

        adaptive = self.affine(style_descriptor)
        gamma, beta = torch.chunk(adaptive, chunks=2, dim=1)
        gamma = gamma.view(batch, channels, 1, 1)
        beta = beta.view(batch, channels, 1, 1)
        return mixed * (self.base_gamma + gamma) + (self.base_beta + beta)


class StyleAgnosticAttention(nn.Module):
    """Channel-spatial hybrid attention driven by style-normalized activations."""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        padding = spatial_kernel_size // 2
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=padding, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance-normalize to reduce style bias before attention.
        normalized = F.instance_norm(x, eps=1e-5)
        pooled_avg = F.adaptive_avg_pool2d(normalized, output_size=1).flatten(1)
        pooled_max = F.adaptive_max_pool2d(normalized, output_size=1).flatten(1)
        channel_weights = torch.sigmoid(self.channel_mlp(pooled_avg) + self.channel_mlp(pooled_max))
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)

        spatial_avg = normalized.mean(dim=1, keepdim=True)
        spatial_max, _ = normalized.max(dim=1, keepdim=True)
        spatial_weights = torch.sigmoid(self.spatial_conv(torch.cat([spatial_avg, spatial_max], dim=1)))

        attended = x * channel_weights * spatial_weights
        return x + torch.tanh(self.alpha) * attended


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class NormFactory(nn.Module):
    """Wrapper to switch between DIFN and BatchNorm for ablations."""

    def __init__(
        self,
        channels: int,
        use_difn: bool,
        eps: float,
        reduction: int,
        momentum: float = 0.1,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.norm = (
            DomainInvariantFeatureNorm(
                channels=channels,
                eps=eps,
                reduction=reduction,
                momentum=momentum,
                track_running_stats=track_running_stats,
            )
            if use_difn
            else nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DINSResidualBlock(nn.Module):
    """Residual feature block with domain-invariant norm and style-agnostic attention."""

    def __init__(
        self,
        channels: int,
        difn_eps: float,
        difn_reduction: int,
        difn_momentum: float,
        difn_track: bool,
        attn_reduction: int,
        attn_kernel_size: int,
        disable_difn: bool,
        disable_attention: bool,
    ) -> None:
        super().__init__()
        use_difn = not disable_difn
        self.conv1 = SeparableConv2d(channels, channels, kernel_size=3, stride=1)
        self.norm1 = NormFactory(
            channels,
            use_difn=use_difn,
            eps=difn_eps,
            reduction=difn_reduction,
            momentum=difn_momentum,
            track_running_stats=difn_track,
        )
        self.conv2 = SeparableConv2d(channels, channels, kernel_size=3, stride=1)
        self.norm2 = NormFactory(
            channels,
            use_difn=use_difn,
            eps=difn_eps,
            reduction=difn_reduction,
            momentum=difn_momentum,
            track_running_stats=difn_track,
        )
        self.act = nn.GELU()
        self.attn = (
            nn.Identity()
            if disable_attention
            else StyleAgnosticAttention(
                channels=channels,
                reduction=attn_reduction,
                spatial_kernel_size=attn_kernel_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.attn(x)
        x = self.act(x + residual)
        return x


class EncoderStage(nn.Module):
    """Encoder stage with optional downsampling followed by residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        difn_eps: float,
        difn_reduction: int,
        difn_momentum: float,
        difn_track: bool,
        attn_reduction: int,
        attn_kernel_size: int,
        disable_difn: bool,
        disable_attention: bool,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        use_difn = not disable_difn
        if downsample:
            self.down = nn.Sequential(
                SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=2),
                NormFactory(
                    out_channels,
                    use_difn=use_difn,
                    eps=difn_eps,
                    reduction=difn_reduction,
                    momentum=difn_momentum,
                    track_running_stats=difn_track,
                ),
                nn.GELU(),
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                NormFactory(
                    out_channels,
                    use_difn=use_difn,
                    eps=difn_eps,
                    reduction=difn_reduction,
                    momentum=difn_momentum,
                    track_running_stats=difn_track,
                ),
                nn.GELU(),
            )

        self.blocks = nn.Sequential(
            *[
                DINSResidualBlock(
                    channels=out_channels,
                    difn_eps=difn_eps,
                    difn_reduction=difn_reduction,
                    difn_momentum=difn_momentum,
                    difn_track=difn_track,
                    attn_reduction=attn_reduction,
                    attn_kernel_size=attn_kernel_size,
                    disable_difn=disable_difn,
                    disable_attention=disable_attention,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.blocks(x)


class MultiScaleContext(nn.Module):
    """Bottleneck context fusion with multi-dilation separable convolutions."""

    def __init__(
        self,
        channels: int,
        difn_eps: float,
        difn_reduction: int,
        difn_momentum: float,
        difn_track: bool,
        disable_difn: bool,
    ) -> None:
        super().__init__()
        use_difn = not disable_difn
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    NormFactory(
                        channels,
                        use_difn=use_difn,
                        eps=difn_eps,
                        reduction=difn_reduction,
                        momentum=difn_momentum,
                        track_running_stats=difn_track,
                    ),
                    nn.GELU(),
                )
                for d in [1, 2, 4]
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            NormFactory(
                channels,
                use_difn=use_difn,
                eps=difn_eps,
                reduction=difn_reduction,
                momentum=difn_momentum,
                track_running_stats=difn_track,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(features, dim=1))


class SkipFusion(nn.Module):
    """Adaptive skip fusion for decoder using gate-conditioned concatenation."""

    def __init__(
        self,
        skip_channels: int,
        up_channels: int,
        out_channels: int,
        difn_eps: float,
        difn_reduction: int,
        difn_momentum: float,
        difn_track: bool,
        disable_difn: bool,
    ) -> None:
        super().__init__()
        use_difn = not disable_difn
        self.skip_gate = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels + up_channels, out_channels, kernel_size=3, padding=1, bias=False),
            NormFactory(
                out_channels,
                use_difn=use_difn,
                eps=difn_eps,
                reduction=difn_reduction,
                momentum=difn_momentum,
                track_running_stats=difn_track,
            ),
            nn.GELU(),
        )

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = self.skip_gate(skip)
        return self.fuse(torch.cat([upsampled, skip * gate], dim=1))


class DecoderStage(nn.Module):
    """Decoder stage with upsampling, skip fusion, and residual refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        difn_eps: float,
        difn_reduction: int,
        difn_momentum: float,
        difn_track: bool,
        attn_reduction: int,
        attn_kernel_size: int,
        disable_difn: bool,
        disable_attention: bool,
    ) -> None:
        super().__init__()
        self.pre_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip_fusion = SkipFusion(
            skip_channels=skip_channels,
            up_channels=out_channels,
            out_channels=out_channels,
            difn_eps=difn_eps,
            difn_reduction=difn_reduction,
            difn_momentum=difn_momentum,
            difn_track=difn_track,
            disable_difn=disable_difn,
        )
        self.refine = DINSResidualBlock(
            channels=out_channels,
            difn_eps=difn_eps,
            difn_reduction=difn_reduction,
            difn_momentum=difn_momentum,
            difn_track=difn_track,
            attn_reduction=attn_reduction,
            attn_kernel_size=attn_kernel_size,
            disable_difn=disable_difn,
            disable_attention=disable_attention,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.pre_up(x)
        x = self.skip_fusion(x, skip)
        return self.refine(x)


class DINSNet(nn.Module):
    """DINSNet: domain-invariant normalization with style-agnostic attention."""

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        in_channels = int(model_cfg["in_channels"])
        num_classes = int(model_cfg["num_classes"])
        base_channels = int(model_cfg["base_channels"])
        multipliers: List[int] = [int(x) for x in model_cfg["channel_multipliers"]]
        blocks_per_stage: List[int] = [int(x) for x in model_cfg["blocks_per_stage"]]

        difn_cfg = model_cfg["difn"]
        attn_cfg = model_cfg["attention"]
        ablation_cfg = model_cfg["ablation"]
        decoder_cfg = model_cfg["decoder"]

        difn_eps = float(difn_cfg["eps"])
        difn_reduction = int(difn_cfg["reduction"])
        difn_momentum = float(difn_cfg.get("momentum", 0.1))
        difn_track = bool(difn_cfg.get("track_running_stats", True))
        attn_reduction = int(attn_cfg["reduction"])
        attn_kernel = int(attn_cfg["spatial_kernel_size"])
        disable_difn = bool(ablation_cfg["disable_difn"])
        disable_attention = bool(ablation_cfg["disable_attention"])

        channels = [base_channels * multiplier for multiplier in multipliers]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            NormFactory(
                channels[0],
                use_difn=not disable_difn,
                eps=difn_eps,
                reduction=difn_reduction,
                momentum=difn_momentum,
                track_running_stats=difn_track,
            ),
            nn.GELU(),
        )

        self.encoder_stages = nn.ModuleList()
        for stage_idx, stage_channels in enumerate(channels):
            in_ch = channels[stage_idx - 1] if stage_idx > 0 else channels[0]
            downsample = stage_idx > 0
            self.encoder_stages.append(
                EncoderStage(
                    in_channels=in_ch,
                    out_channels=stage_channels,
                    num_blocks=blocks_per_stage[stage_idx],
                    difn_eps=difn_eps,
                    difn_reduction=difn_reduction,
                    difn_momentum=difn_momentum,
                    difn_track=difn_track,
                    attn_reduction=attn_reduction,
                    attn_kernel_size=attn_kernel,
                    disable_difn=disable_difn,
                    disable_attention=disable_attention,
                    downsample=downsample,
                )
            )

        self.context = MultiScaleContext(
            channels=channels[-1],
            difn_eps=difn_eps,
            difn_reduction=difn_reduction,
            difn_momentum=difn_momentum,
            difn_track=difn_track,
            disable_difn=disable_difn,
        )

        decoder_channels = list(reversed(channels[:-1]))
        self.decoder_stages = nn.ModuleList()
        in_ch = channels[-1]
        for idx, out_ch in enumerate(decoder_channels):
            skip_ch = decoder_channels[idx]
            self.decoder_stages.append(
                DecoderStage(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    difn_eps=difn_eps,
                    difn_reduction=difn_reduction,
                    difn_momentum=difn_momentum,
                    difn_track=difn_track,
                    attn_reduction=attn_reduction,
                    attn_kernel_size=attn_kernel,
                    disable_difn=disable_difn,
                    disable_attention=disable_attention,
                )
            )
            in_ch = out_ch

        self.dropout = nn.Dropout2d(p=float(decoder_cfg["dropout"]))
        self.seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.prob_head = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.encoder_stages) - 1:
                skips.append(x)

        x = self.context(x)
        for decoder_stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = decoder_stage(x, skip)

        x = self.dropout(x)
        logits = self.seg_head(x)
        return self.prob_head(logits) if apply_sigmoid else logits
