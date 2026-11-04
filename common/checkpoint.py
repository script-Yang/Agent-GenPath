"""Strict state-dict loading for shared and task models."""

from .config import read_path


def load_weights(model, path):
    import torch

    state = torch.load(read_path(path), map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
