import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Arquitetura IDÊNTICA à da Fase 2 (nomes de módulos precisam casar com o checkpoint).
# ---------- Componentes da Fase 1 (NÃO alterar: precisam casar com os pesos salvos) ----------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.dropout(attn)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.dropout(self.fc2(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], dim=1)
        x = self.pos_dropout(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class FFTBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    @staticmethod
    def to_log_magnitude_spectrum(x):
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        return torch.log1p(torch.abs(fft))

    def forward(self, x):
        feat = self.conv(self.to_log_magnitude_spectrum(x))
        return self.fc(self.pool(feat).flatten(1))


# ---------- Novo na Fase 2 ----------
class CNNBranch(nn.Module):
    # EfficientNet-B0 (ImageNet) -> vetor de 1280 dims. Converte a normalização [-1,1] do projeto
    # para a normalização esperada pelos pesos do ImageNet.
    def __init__(self, pretrained=True):
        super().__init__()
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.pool(self.features(x)).flatten(1)


class Phase2Detector(nn.Module):
    DIMS = {"vit": 192, "fft": 128, "cnn": 1280}

    def __init__(self, branches=("vit", "fft", "cnn"), fusion="gate", proj_dim=256,
                 pretrained_cnn=True, num_classes=2, dropout=0.2):
        super().__init__()
        self.names = [n for n in ("vit", "fft", "cnn") if n in branches]
        self.fusion = fusion
        mods = {}
        if "vit" in self.names: mods["vit"] = ViTTiny(embed_dim=192)
        if "fft" in self.names: mods["fft"] = FFTBranch(out_dim=128)
        if "cnn" in self.names: mods["cnn"] = CNNBranch(pretrained_cnn)
        self.branches = nn.ModuleDict(mods)
        self.proj = nn.ModuleDict({n: nn.Sequential(nn.LayerNorm(self.DIMS[n]), nn.Linear(self.DIMS[n], proj_dim))
                                   for n in self.names})
        k = len(self.names)
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(k * proj_dim, 64), nn.GELU(), nn.Linear(64, k))
            cls_in = proj_dim
        else:
            self.gate = None
            cls_in = k * proj_dim
        self.classifier = nn.Sequential(nn.LayerNorm(cls_in), nn.Linear(cls_in, 128), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x, return_gate=False):
        feats = [self.proj[n](self.branches[n](x)) for n in self.names]
        gate_w = None
        if self.fusion == "gate":
            gate_w = torch.softmax(self.gate(torch.cat(feats, dim=1)), dim=1)          # (B, k)
            fused = (torch.stack(feats, dim=1) * gate_w.unsqueeze(-1)).sum(dim=1)       # (B, D)
        else:
            fused = torch.cat(feats, dim=1)
        logits = self.classifier(fused)
        return (logits, gate_w) if return_gate else logits
