# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model Zoo — PyTorch port of TCC models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Default config values (mirrors the MODEL section of tf2:config.py).
# These are used when no explicit cfg dict is supplied.
# ---------------------------------------------------------------------------
_DEFAULT_CFG: Dict[str, Any] = {
    # Base model ---------------------------------------------------------
    "base_model_network": "Resnet50_pretrained",
    # Which ResNet layer to extract features from.
    # "layer3" corresponds to conv4_block3_out in Keras ResNet50V2
    # and produces 14x14x1024 feature maps for 224x224 input.
    "base_model_layer": "layer3",
    # frozen | only_bn | train_all
    "train_base": "only_bn",

    # Embedder type: "conv" | "convgru" -----------------------------------
    "embedder_type": "conv",

    # ConvEmbedder --------------------------------------------------------
    # Each tuple: (channels, kernel_size, activate?)
    "conv_embedder_conv_layers": [(256, 3, True), (256, 3, True)],
    "conv_embedder_flatten_method": "max_pool",
    # Each tuple: (channels, activate?)
    "conv_embedder_fc_layers": [(256, True), (256, True)],
    "conv_embedder_capacity_scalar": 2,
    "conv_embedder_embedding_size": 128,
    "conv_embedder_l2_normalize": False,
    "conv_embedder_base_dropout_rate": 0.0,
    "conv_embedder_base_dropout_spatial": False,
    "conv_embedder_fc_dropout_rate": 0.1,
    "conv_embedder_use_bn": True,

    # ConvGRU Embedder ----------------------------------------------------
    "convgru_conv_layers": [(512, 3, True), (512, 3, True)],
    "convgru_gru_layers": [128],
    "convgru_dropout_rate": 0.0,
    "convgru_use_bn": True,

    # Data ----------------------------------------------------------------
    "num_steps": 2,  # context frames
    "num_frames": 20,  # training frames

    # FC layers for embedder head
    "fc_hidden": 512,
}


def _cfg_get(cfg: Optional[Dict[str, Any]], key: str) -> Any:
    """Look up *key* in *cfg*, falling back to ``_DEFAULT_CFG``."""
    if cfg is not None and key in cfg:
        return cfg[key]
    return _DEFAULT_CFG[key]


# =========================================================================
# BaseModel — CNN feature extractor
# =========================================================================

class BaseModel(nn.Module):
    """ResNet-50 backbone that extracts spatial feature maps per frame.

    Input shape:  ``[B, T, 3, H, W]``  (channels-first, PyTorch convention)
    Output shape: ``[B, T, C, H', W']`` where ``C=1024, H'=W'=14`` for
    ``layer3`` with 224x224 input.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._train_base = _cfg_get(cfg, "train_base")

        # Build ResNet-50 with ImageNet weights
        resnet = tv_models.resnet50(weights="IMAGENET1K_V1")

        # Map layer name -> children slice
        layer_map = {
            "layer1": 5,
            "layer2": 6,
            "layer3": 7,
            "layer4": 8,
        }
        layer_name = _cfg_get(cfg, "base_model_layer")
        if layer_name not in layer_map:
            raise ValueError(
                f"Unsupported base_model_layer '{layer_name}'. "
                f"Choose from {list(layer_map.keys())}."
            )
        end = layer_map[layer_name]
        children = list(resnet.children())[:end]
        self.backbone = nn.Sequential(*children)

        # Apply frozen / only_bn / train_all
        self._apply_training_mode()

    # ------------------------------------------------------------------
    def _apply_training_mode(self) -> None:
        mode = self._train_base
        if mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif mode == "only_bn":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    for p in m.parameters():
                        p.requires_grad = True
        elif mode == "train_all":
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(
                f"Unsupported train_base mode '{mode}'. "
                "Choose from: frozen, only_bn, train_all."
            )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "BaseModel":
        """Override to keep BN in eval mode when backbone is frozen."""
        super().train(mode)
        if self._train_base == "frozen":
            self.backbone.eval()
        elif self._train_base == "only_bn":
            # Non-BN layers stay in eval; BN layers follow the global mode.
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.train(mode)
        return self

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, 3, H, W]``
        Returns:
            ``[B, T, C, H', W']``
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.backbone(x)
        _, Cout, Hout, Wout = x.shape
        x = x.reshape(B, T, Cout, Hout, Wout)
        return x


# =========================================================================
# ConvEmbedder — Conv3D + FC embedding head
# =========================================================================

class ConvEmbedder(nn.Module):
    """Temporal-convolution embedder network.

    Takes CNN feature maps with context frames and produces a fixed-size
    embedding vector per frame.

    Input:  ``[B, T_total, C, H, W]`` where ``T_total = num_frames * num_context``
    Output: ``[B * num_frames, embedding_size]``
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        conv_params = _cfg_get(cfg, "conv_embedder_conv_layers")
        fc_params = _cfg_get(cfg, "conv_embedder_fc_layers")
        use_bn = _cfg_get(cfg, "conv_embedder_use_bn")
        embedding_size = _cfg_get(cfg, "conv_embedder_embedding_size")
        cap = _cfg_get(cfg, "conv_embedder_capacity_scalar")
        self._flatten_method = _cfg_get(cfg, "conv_embedder_flatten_method")
        self._l2_normalize = _cfg_get(cfg, "conv_embedder_l2_normalize")
        self._base_dropout_rate = _cfg_get(cfg, "conv_embedder_base_dropout_rate")
        self._base_dropout_spatial = _cfg_get(cfg, "conv_embedder_base_dropout_spatial")
        self._fc_dropout_rate = _cfg_get(cfg, "conv_embedder_fc_dropout_rate")
        self._num_steps = _cfg_get(cfg, "num_steps")

        conv_params = [(cap * ch, ks, act) for ch, ks, act in conv_params]
        fc_params = [(cap * ch, act) for ch, act in fc_params]

        # --- Conv3D layers ------------------------------------------------
        # PyTorch Conv3d: input is [B, C, D, H, W]
        # We infer in_channels from the first forward pass (lazy) or
        # require the caller to set it.  Here we use 1024 as default for
        # layer3 of ResNet-50.
        in_ch: int = _cfg_get(cfg, "conv_in_channels") if (
            cfg and "conv_in_channels" in cfg
        ) else 1024

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self._activations: List[bool] = []
        for ch, ks, act in conv_params:
            self.conv_layers.append(
                nn.Conv3d(in_ch, ch, kernel_size=ks, padding=ks // 2)
            )
            if use_bn:
                self.bn_layers.append(nn.BatchNorm3d(ch))
            self._activations.append(act)
            in_ch = ch

        self._use_bn = use_bn

        # --- FC layers ----------------------------------------------------
        flat_ch = in_ch  # after pooling, we have in_ch features
        self.fc_layers = nn.ModuleList()
        for ch, act in fc_params:
            layers: list[nn.Module] = [nn.Linear(flat_ch, ch)]
            if act:
                layers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Sequential(*layers))
            flat_ch = ch

        self.embedding_layer = nn.Linear(flat_ch, embedding_size)

        # Kaiming init for conv & linear
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, T_total, C, H, W]`` from BaseModel.
            num_frames: number of target frames (T_total // num_frames gives
                        the context window size).
        Returns:
            ``[B * num_frames, embedding_size]``
        """
        B, T_total, C, H, W = x.shape
        num_context = T_total // num_frames
        # Reshape to [B*num_frames, C, num_context, H, W] — Conv3d format
        x = x.reshape(B * num_frames, num_context, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [BF, C, D, H, W]

        # Base dropout
        if self.training and self._base_dropout_rate > 0:
            if self._base_dropout_spatial:
                # Drop entire channels (spatial dropout for 3D)
                x = F.dropout3d(x, p=self._base_dropout_rate, training=True)
            else:
                x = F.dropout(x, p=self._base_dropout_rate, training=True)

        # Conv layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self._use_bn:
                x = self.bn_layers[i](x)
            if self._activations[i]:
                x = F.relu(x, inplace=True)

        # Spatial pooling
        if self._flatten_method == "max_pool":
            x = F.adaptive_max_pool3d(x, 1).flatten(1)
        elif self._flatten_method == "avg_pool":
            x = F.adaptive_avg_pool3d(x, 1).flatten(1)
        elif self._flatten_method == "flatten":
            x = x.flatten(1)
        else:
            raise ValueError(
                f"Unsupported flatten method '{self._flatten_method}'. "
                "Choose from: max_pool, avg_pool, flatten."
            )

        # FC layers
        for fc in self.fc_layers:
            x = F.dropout(x, p=self._fc_dropout_rate, training=self.training)
            x = fc(x)

        x = self.embedding_layer(x)

        if self._l2_normalize:
            x = F.normalize(x, dim=-1)

        return x


# =========================================================================
# ConvGRUEmbedder — Conv2D + GRU alternative
# =========================================================================

class ConvGRUEmbedder(nn.Module):
    """Conv2D per frame followed by GRU over the temporal dimension.

    Input:  ``[B, T, C, H, W]``
    Output: ``[B * T, gru_hidden]``
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        num_steps = _cfg_get(cfg, "num_steps")
        if num_steps != 1:
            raise ValueError("Cannot use ConvGRUEmbedder with context frames (num_steps must be 1).")

        conv_params = _cfg_get(cfg, "convgru_conv_layers")
        use_bn = _cfg_get(cfg, "convgru_use_bn")
        gru_units_list = _cfg_get(cfg, "convgru_gru_layers")
        dropout_rate = _cfg_get(cfg, "convgru_dropout_rate")

        # --- Conv2D layers -----------------------------------------------
        in_ch: int = _cfg_get(cfg, "conv_in_channels") if (
            cfg and "conv_in_channels" in cfg
        ) else 1024

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self._activations: List[bool] = []
        self._use_bn = use_bn

        for ch, ks, act in conv_params:
            self.conv_layers.append(
                nn.Conv2d(in_ch, ch, kernel_size=ks, padding=ks // 2)
            )
            if use_bn:
                self.bn_layers.append(nn.BatchNorm2d(ch))
            self._activations.append(act)
            in_ch = ch

        self.dropout = nn.Dropout(dropout_rate)

        # --- GRU layers ---------------------------------------------------
        self.gru_layers = nn.ModuleList()
        gru_in = in_ch
        for units in gru_units_list:
            self.gru_layers.append(
                nn.GRU(input_size=gru_in, hidden_size=units, batch_first=True)
            )
            gru_in = units

        self._gru_out_dim = gru_in
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, C, H, W]``
            num_frames: ignored (kept for API compatibility).
        Returns:
            ``[B * T, gru_hidden]``
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        # Conv2d per frame
        for i, conv in enumerate(self.conv_layers):
            x = self.dropout(x)
            x = conv(x)
            if self._use_bn:
                x = self.bn_layers[i](x)
            if self._activations[i]:
                x = F.relu(x, inplace=True)

        # Global max pool -> [B*T, C']
        x = F.adaptive_max_pool2d(x, 1).flatten(1)

        c = x.shape[-1]
        x = x.reshape(B, T, c)

        for gru in self.gru_layers:
            x, _ = gru(x)

        # Reshape back to [B*T, hidden]
        x = x.reshape(B * T, -1)
        return x


# =========================================================================
# Classifier — FC head for SAL / classification tasks
# =========================================================================

class Classifier(nn.Module):
    """Configurable fully-connected classifier head.

    Args:
        fc_layers: list of ``(units, activate)`` tuples.
        dropout_rate: dropout probability applied before each FC layer.
    """

    def __init__(
        self,
        fc_layers: List[Tuple[int, bool]],
        dropout_rate: float = 0.0,
        in_features: int = 128,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        self.fc_layers = nn.ModuleList()
        in_dim = in_features
        for units, activate in fc_layers:
            sublayers: list[nn.Module] = [nn.Linear(in_dim, units)]
            if activate:
                sublayers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Sequential(*sublayers))
            in_dim = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fc_layers:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = fc(x)
        return x


# =========================================================================
# Factory
# =========================================================================

def get_model(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, nn.Module]:
    """Build and return the model dictionary.

    Returns:
        ``{"cnn": BaseModel, "emb": ConvEmbedder | ConvGRUEmbedder}``
    """
    cnn = BaseModel(cfg=cfg)

    embedder_type = _cfg_get(cfg, "embedder_type")
    if embedder_type == "conv":
        emb: nn.Module = ConvEmbedder(cfg=cfg)
    elif embedder_type == "convgru":
        emb = ConvGRUEmbedder(cfg=cfg)
    else:
        raise ValueError(f"Unsupported embedder_type '{embedder_type}'.")

    return {"cnn": cnn, "emb": emb}
