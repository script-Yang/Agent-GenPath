"""Multiscale projection, resize/refinement, and feature fusion decoder.

Features from blocks [2, 5, 8, 11] are projected to image strides [32, 16, 8, 4],
resized to stride 4, refined, concatenated, and decoded to semantic logits.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ScaleProjection(nn.Module):
    def __init__(self, in_dim, channels, patch_size, image_stride):
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, channels, 1)
        if image_stride == patch_size:
            self.resample = nn.Identity()
        elif image_stride > patch_size and image_stride % patch_size == 0:
            self.resample = nn.Conv2d(
                channels, channels, 3, stride=image_stride // patch_size, padding=1
            )
        elif image_stride < patch_size and patch_size % image_stride == 0:
            scale = patch_size // image_stride
            self.resample = nn.ConvTranspose2d(channels, channels, scale, stride=scale)
        else:
            raise ValueError("Projection strides must be integer multiples of patch size")

    def forward(self, feature):
        return self.resample(self.reduce(feature))


class DecoderV1(nn.Module):
    def __init__(self, in_dim=768, embed_dim=256, num_classes=6, patch_size=8,
                 image_strides=(32, 16, 8, 4)):
        super().__init__()
        if tuple(image_strides) != (32, 16, 8, 4):
            raise ValueError("image_strides must be [32, 16, 8, 4]")
        if embed_dim % 32:
            raise ValueError("embed_dim must be divisible by 32 for inherited GroupNorm")
        self.patch_size = patch_size
        self.image_strides = tuple(image_strides)
        self.common_stride = min(image_strides)
        self.projections = nn.ModuleList([
            ScaleProjection(in_dim, embed_dim, patch_size, stride)
            for stride in image_strides
        ])
        self.refinements = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, embed_dim), nn.ReLU(),
            ) for _ in range(4)
        ])
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, embed_dim), nn.ReLU(), nn.Dropout2d(0.1),
        )
        self.seg_branch = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=self.common_stride, mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim // 2, num_classes, 3, padding=1),
        )

    def forward(self, features):
        if len(features) != 4 or any(f.shape != features[0].shape for f in features):
            raise ValueError("Expected four equally shaped feature maps in block order [2,5,8,11]")
        image_size = tuple(size * self.patch_size for size in features[0].shape[-2:])
        if any(size % max(self.image_strides) for size in image_size):
            raise ValueError("Image dimensions must be divisible by the largest stride")
        common_size = tuple(size // self.common_stride for size in image_size)
        aligned = []
        for feature, projection, refinement in zip(features, self.projections, self.refinements):
            feature = projection(feature)
            feature = F.interpolate(feature, size=common_size, mode="bilinear", align_corners=False)
            aligned.append(refinement(feature))
        return self.seg_branch(self.fusion_layer(torch.cat(aligned, dim=1)))
