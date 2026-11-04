"""Shared DiT backbone with decoder_v1 or decoder_v2 (Mini, default)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.checkpoint import load_weights
from models.dit import MFDiT
from .decoders.decoder_v2 import DecoderV2
from .decoders.decoder_v1 import DecoderV1


def tokens_to_2d(tokens, h_w_size):
    B, T, D = tokens.shape
    H_p, W_p = h_w_size
    num_spatial_tokens = H_p * W_p

    # Discard leading non-spatial tokens.
    if T > num_spatial_tokens:
        tokens = tokens[:, -num_spatial_tokens:, :]
    elif T < num_spatial_tokens:
        raise ValueError(f"Token count {T} is less than spatial size {H_p}x{W_p}")

    return tokens.transpose(1, 2).reshape(B, D, H_p, W_p)


class DiTForSegmentation(nn.Module):
    def __init__(
        self, dit_model, patch_size=8, img_size=256, num_classes=6, embed_dim=256,
        out_indices=None, head="decoder_v2", decoder_options=None
    ):
        super().__init__()
        self.backbone = dit_model
        self.patch_size = patch_size
        self.img_size = img_size
        self.h_w_size = (img_size // patch_size, img_size // patch_size)
        self.out_indices = out_indices or [2, 5, 8, 11]
        decoder_args = dict(
            in_dim=dit_model.pos_embed.shape[-1], embed_dim=embed_dim, num_classes=num_classes
        )
        if head == "decoder_v2":
            self.decoder = DecoderV2(**decoder_args)
        elif head == "decoder_v1":
            self.decoder = DecoderV1(
                **decoder_args, patch_size=patch_size, **(decoder_options or {})
            )
        else:
            raise ValueError(f"Unknown segmentation head: {head!r}")

    def forward(self, img, t, r, y=None):
        token_features = self.backbone.forward_features(
            img, t, r, y=y, out_indices=self.out_indices
        )

        spatial_features = [tokens_to_2d(f, self.h_w_size) for f in token_features]

        seg_logits = self.decoder(spatial_features)

        if seg_logits.shape[-2:] != img.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=img.shape[-2:], mode="bilinear", align_corners=False
            )

        return seg_logits


def build_model(cfg, initialize=False):
    model_cfg = cfg["model"]
    if model_cfg.get("head", "decoder_v2") not in {"decoder_v2", "decoder_v1"}:
        raise ValueError("model.head must be decoder_v1 or decoder_v2")
    net = MFDiT(**model_cfg["backbone"])
    if initialize and cfg.get("pretrained"):
        load_weights(net, cfg["pretrained"])
    return DiTForSegmentation(
        net,
        patch_size=model_cfg["backbone"]["patch_size"],
        img_size=model_cfg["backbone"]["input_size"],
        num_classes=model_cfg["num_classes"],
        embed_dim=model_cfg["decoder_dim"],
        out_indices=model_cfg["out_indices"],
        head=model_cfg.get("head", "decoder_v2"),
        decoder_options=model_cfg.get("decoder_options"),
    )


def predict(model, images, cfg):
    t = torch.full((len(images),), float(cfg["training"]["rate"]), device=images.device)
    return model(images, t, torch.zeros_like(t))
