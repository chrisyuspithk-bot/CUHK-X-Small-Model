"""
Lightweight modality-specific encoders for CUHK-X Small Model Track.

All encoders produce a fixed-dimensional latent vector (D=256).
Designed to be parameter-efficient for sub-100MB total model size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """Pseudo-3D CNN for depth frames using MobileNetV3-inspired blocks.
    Input: (B, 1, T, H, W)  Output: (B, out_dim)
    """

    def __init__(self, in_channels=1, out_dim=512, T=60, H=112, W=112):
        super().__init__()
        self.spatial_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
        )
        self.temporal_conv2 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(192, out_dim)

    def forward(self, x):
        x = self.spatial_conv1(x)
        x = self.spatial_conv2(x)
        x = self.temporal_conv(x)
        x = self.temporal_conv2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SkeletonEncoder(nn.Module):
    """3D-CNN for skeleton heatmap volumes (PoseConv3D-inspired).
    Input: (B, J=17, T, H=56, W=56)  Output: (B, out_dim)
    """

    def __init__(self, in_channels=17, out_dim=512, T=60, H=56, W=56):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(192, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class IMUEncoder(nn.Module):
    """1D-CNN + BiGRU for IMU time-series data.
    Input: (B, 45, T) — 5 sensors × 9 features
    Output: (B, out_dim)
    """

    def __init__(self, in_channels=45, out_dim=512, T=60):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(192, 192, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(384, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc(x)


class RadarEncoder(nn.Module):
    """2D-CNN for radar pseudo-image projections.
    Input: (B, 3, T, H=32, W=32)  Output: (B, out_dim)
    """

    def __init__(self, in_channels=3, out_dim=512, T=60, H=32, W=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
        )
        self.temporal2 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(192, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.temporal(x)
        x = self.temporal2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SharedVisualEncoder(nn.Module):
    """Shared early layers for Depth, IR, Thermal with modality-specific adapters.
    Input: (B, 1, T, H=112, W=112)  Output: (B, out_dim)
    """

    def __init__(self, out_dim=512, T=60, H=112, W=112):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 48, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 192, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(192, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
