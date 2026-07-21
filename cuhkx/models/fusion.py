"""
Gated Multimodal Fusion (GMF) module with Stochastic Modality Dropout.
"""

import random

import torch
import torch.nn as nn

from .encoders import IMUEncoder, RadarEncoder, SharedVisualEncoder, SkeletonEncoder


class GatedMultimodalFusion(nn.Module):
    """Dynamic gating fusion: z_m = σ(W·[h_1;...;h_M] + b), h_fused = Σ(z_m ⊙ h_m)"""

    def __init__(self, num_modalities, embed_dim=512):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, embeddings, masks=None):
        """
        Args:
            embeddings: List of (B, D) tensors, one per modality.
            masks: (B, M) float tensor, 1.0 = present, 0.0 = missing.
        Returns:
            fused: (B, D) fused representation
        """
        B, D = embeddings[0].size(0), embeddings[0].size(1)
        device = embeddings[0].device

        if masks is None:
            masks = torch.ones(B, self.num_modalities, device=device)

        stacked = torch.stack(embeddings, dim=1)  # (B, M, D)

        # Compute gates from concatenated embeddings
        concat = stacked.reshape(B, -1)  # (B, M*D)
        gates = self.gate_mlp(concat)  # (B, M)
        gates = gates * masks  # Zero out missing modalities

        # Gated fusion
        gated = gates.unsqueeze(-1) * stacked  # (B, M, D)
        fused = gated.sum(dim=1)  # (B, D)

        # Normalize by number of active modalities
        active_count = masks.sum(dim=1, keepdim=True).clamp(min=1)
        fused = fused / active_count * self.num_modalities

        return fused


class CUHKXModel(nn.Module):
    """Full multimodal HAR model for CUHK-X Small Model Track.

    Architecture:
        Skeleton  → SkeletonEncoder     → (B, 256)
        Depth     → SharedVisualEncoder → (B, 256)
        IR        → SharedVisualEncoder → (B, 256)
        Thermal   → SharedVisualEncoder → (B, 256)
        IMU       → IMUEncoder          → (B, 256)
        Radar     → RadarEncoder        → (B, 256)
        All       → GatedMultimodalFusion → Classifier → (B, 40)

    Modality keys: 'Skeleton', 'Depth_Color', 'IR', 'Thermal', 'IMU', 'Radar'
    """

    def __init__(self, num_classes=40, embed_dim=512, T=60, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.T = T

        # Visual encoders (shared backbone, separate instances for modality-specific adaptation)
        self.depth_encoder = SharedVisualEncoder(out_dim=embed_dim, T=T, H=112, W=112)
        self.ir_encoder = SharedVisualEncoder(out_dim=embed_dim, T=T, H=112, W=112)
        self.thermal_encoder = SharedVisualEncoder(out_dim=embed_dim, T=T, H=112, W=112)

        # Specialized encoders
        self.skeleton_encoder = SkeletonEncoder(in_channels=17, out_dim=embed_dim, T=T, H=56, W=56)
        self.imu_encoder = IMUEncoder(in_channels=45, out_dim=embed_dim, T=T)
        self.radar_encoder = RadarEncoder(in_channels=3, out_dim=embed_dim, T=T)

        # Fusion
        self.fusion = GatedMultimodalFusion(num_modalities=6, embed_dim=embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._modality_dropout_prob = 0.2
        self._single_modality_prob = 0.05

    def set_dropout(self, modality_dropout_prob=0.2, single_modality_prob=0.05):
        self._modality_dropout_prob = modality_dropout_prob
        self._single_modality_prob = single_modality_prob

    def forward(self, skeleton=None, depth=None, thermal=None, ir=None, imu=None, radar=None,
                modality_mask=None, training=True):
        """
        Args:
            Each modality is a tensor:
                skeleton: (B, 17, T, 56, 56)
                depth:    (B, 1, T, 112, 112)
                thermal:  (B, 1, T, 112, 112)
                ir:       (B, 1, T, 112, 112)
                imu:      (B, 45, T)
                radar:    (B, 3, T, 32, 32)
            modality_mask: (B, 6) float tensor, 1.0 = present, 0.0 = missing.
        Returns:
            logits: (B, 40)
        """
        modalities = {
            "Skeleton": skeleton,
            "Depth_Color": depth,
            "IR": ir,
            "Thermal": thermal,
            "IMU": imu,
            "Radar": radar,
        }

        # Stochastic modality dropout during training
        if training:
            modalities, modality_mask = self._apply_modality_dropout(modalities, modality_mask)

        # Encode each modality
        embeddings = []
        for name in ["Skeleton", "Depth_Color", "IR", "Thermal", "IMU", "Radar"]:
            data = modalities[name]

            if name == "Skeleton":
                emb = self.skeleton_encoder(data)
            elif name == "IMU":
                emb = self.imu_encoder(data)
            elif name == "Radar":
                emb = self.radar_encoder(data)
            elif name == "Depth_Color":
                emb = self.depth_encoder(data)
            elif name == "IR":
                emb = self.ir_encoder(data)
            elif name == "Thermal":
                emb = self.thermal_encoder(data)
            else:
                emb = torch.zeros(data.size(0), self.embed_dim, device=data.device)

            embeddings.append(emb)

        # Fusion
        fused = self.fusion(embeddings, masks=modality_mask)
        return self.classifier(fused)

    def _apply_modality_dropout(self, modalities, modality_mask=None):
        """Randomly zero out modalities during training."""
        B = modalities["Skeleton"].size(0)
        device = modalities["Skeleton"].device

        if modality_mask is None:
            modality_mask = torch.ones(B, 6, device=device)

        if not self.training:
            return modalities, modality_mask

        new_mask = modality_mask.clone()
        result = dict(modalities)

        # Single modality mode
        if random.random() < self._single_modality_prob:
            active = torch.where(new_mask.sum(dim=0) > 0)[0]
            if len(active) > 0:
                keep_idx = random.choice(active.tolist())
                new_mask.zero_()
                new_mask[:, keep_idx] = modality_mask[:, keep_idx]
                keys = list(modalities.keys())
                for i, k in enumerate(keys):
                    if i != keep_idx:
                        result[k] = torch.zeros_like(modalities[k])
            return result, new_mask

        # Independent dropout
        for i, name in enumerate(modalities.keys()):
            if random.random() < self._modality_dropout_prob:
                new_mask[:, i] = 0.0
                result[name] = torch.zeros_like(modalities[name])

        # Ensure at least one modality remains
        if new_mask.sum() == 0:
            active = torch.where(modality_mask.sum(dim=0) > 0)[0]
            if len(active) > 0:
                keep_idx = random.choice(active.tolist())
                new_mask[:, keep_idx] = 1.0
                keys = list(modalities.keys())
                result[keys[keep_idx]] = modalities[keys[keep_idx]]

        return result, new_mask

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def estimate_size_mb(self):
        """Estimate model size in MB (FP32)."""
        total, _ = self.count_parameters()
        return total * 4 / (1024 * 1024)  # 4 bytes per FP32 parameter
