"""Baseline va CBM.

CBM: anh -> backbone (2 tang dac trung: stride 16 va stride 32) -> chieu 1x1 ve d_model
     -> + vi tri hoc duoc + nhung theo tang -> 1..n khoi Transformer (self-attention)
     -> 14 truy van concept, multi-head cross-attention (moi concept 1 heatmap)
     -> 14 sigmoid concept -> MLP -> 5 benh.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
}


def torchvision_model(name, pretrained=True):
    fn, w = SPECS[name]
    return fn(weights=w if pretrained else None)


class Baseline(nn.Module):
    """Giong baseline cu: chi thay lop cuoi bang Linear(.,5)."""

    def __init__(self, name, n_classes, pretrained=True):
        super().__init__()
        m = torchvision_model(name, pretrained)
        if name == "resnet18":
            m.fc = nn.Linear(m.fc.in_features, n_classes)
        elif name == "densenet121":
            m.classifier = nn.Linear(m.classifier.in_features, n_classes)
        else:
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        self.net = m

    def forward(self, x, return_attention=False):
        return {"class_logits": self.net(x)}


def _resize_pos(pos, grid, target):
    if tuple(grid) == tuple(target):
        return pos
    h, w = grid
    p = pos.reshape(1, h, w, -1).permute(0, 3, 1, 2)
    p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
    return p.permute(0, 2, 3, 1).reshape(1, target[0] * target[1], -1)


class ConceptAttentionCBM(nn.Module):
    def __init__(self, name, feature_nodes, n_concepts, n_classes, image_size, d_model=256, n_heads=8,
                 n_transformer_blocks=1, ff_dim=512, dropout=0.1, mlp_hidden=64, pretrained=True, **_):
        super().__init__()
        base = torchvision_model(name, pretrained)
        self.body = create_feature_extractor(base, return_nodes={feature_nodes["low"]: "low", feature_nodes["high"]: "high"})
        with torch.no_grad():
            self.body.eval()
            f = self.body(torch.zeros(1, 3, image_size, image_size))
            self.body.train()
        c_lo, h_lo, w_lo = f["low"].shape[1:]
        c_hi, h_hi, w_hi = f["high"].shape[1:]
        self.grid_low, self.grid_high = (h_lo, w_lo), (h_hi, w_hi)
        self.channels = (c_lo, c_hi)
        d = d_model
        self.proj_low = nn.Conv2d(c_lo, d, 1)
        self.proj_high = nn.Conv2d(c_hi, d, 1)
        self.pos_low = nn.Parameter(torch.zeros(1, h_lo * w_lo, d))
        self.pos_high = nn.Parameter(torch.zeros(1, h_hi * w_hi, d))
        self.scale_emb = nn.Parameter(torch.zeros(2, d))
        for p in (self.pos_low, self.pos_high, self.scale_emb):
            nn.init.trunc_normal_(p, std=0.02)
        if n_transformer_blocks > 0:
            layer = nn.TransformerEncoderLayer(d, n_heads, ff_dim, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_transformer_blocks, enable_nested_tensor=False)
        else:
            self.encoder = nn.Identity()
        self.norm_tokens = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(n_concepts, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.concept_w = nn.Parameter(torch.randn(n_concepts, d) * 0.02)
        self.concept_b = nn.Parameter(torch.zeros(n_concepts))
        self.task_head = nn.Sequential(nn.Linear(n_concepts, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(mlp_hidden, n_classes))

    def forward(self, x, return_attention=False):
        f = self.body(x)
        lo, hi = self.proj_low(f["low"]), self.proj_high(f["high"])
        b, _, h_lo, w_lo = lo.shape
        h_hi, w_hi = hi.shape[2:]
        t_lo = lo.flatten(2).transpose(1, 2) + _resize_pos(self.pos_low, self.grid_low, (h_lo, w_lo)) + self.scale_emb[0]
        t_hi = hi.flatten(2).transpose(1, 2) + _resize_pos(self.pos_high, self.grid_high, (h_hi, w_hi)) + self.scale_emb[1]
        tokens = self.norm_tokens(self.encoder(torch.cat([t_lo, t_hi], 1)))
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        z, attn = self.cross(q, tokens, tokens, need_weights=return_attention, average_attn_weights=True)
        z = self.norm_q(z + q)
        concept_logits = (z * self.concept_w).sum(-1) + self.concept_b
        class_logits = self.task_head(torch.sigmoid(concept_logits))
        out = {"class_logits": class_logits, "concept_logits": concept_logits}
        if return_attention:
            n_lo = h_lo * w_lo
            a_lo = attn[:, :, :n_lo].reshape(b, -1, h_lo, w_lo)
            a_hi = attn[:, :, n_lo:].reshape(b, -1, h_hi, w_hi)
            a_hi = F.interpolate(a_hi.float(), size=(h_lo, w_lo), mode="bilinear", align_corners=False)
            out["heatmap"] = a_lo.float() + a_hi  # (B, 14, h, w)
        return out


def build_model(cfg, mode, pretrained=True):
    n_cls = len(cfg["classes"])
    name = cfg["backbone"]["name"]
    if mode == "baseline":
        return Baseline(name, n_cls, pretrained)
    if mode == "cbm":
        return ConceptAttentionCBM(name, cfg["backbone"]["feature_nodes"], len(cfg["concepts"]), n_cls,
                                   cfg["image_size"], pretrained=pretrained, **cfg["cbm"])
    raise ValueError(mode)
