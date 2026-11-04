"""Multi-level convolutional decoder for DiT feature maps."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderV2(nn.Module):
    def __init__(self, in_dim=768, embed_dim=256, num_classes=6):
        super().__init__()
        self.num_classes = num_classes

        self.projs = nn.ModuleList([nn.Conv2d(in_dim, embed_dim, kernel_size=1) for _ in range(4)])

        # GroupNorm supports small training batches.
        self.fusion_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, embed_dim),
                    nn.ReLU(),
                )
                for _ in range(4)
            ]
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        self.seg_branch = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, features):
        feats = [proj(f) for proj, f in zip(self.projs, features)]

        target_size = (feats[0].shape[2] * 2, feats[0].shape[3] * 2)

        up_feats = []
        for i, f in enumerate(feats):
            # Convolve at token resolution to reduce memory use.
            f_conv = self.fusion_convs[i](f)
            f_up = F.interpolate(f_conv, size=target_size, mode="bilinear", align_corners=False)
            up_feats.append(f_up)

        combined = self.fusion_layer(torch.cat(up_feats, dim=1))

        return self.seg_branch(combined)
