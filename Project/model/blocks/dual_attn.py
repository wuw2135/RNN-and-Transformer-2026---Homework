import torch
import torch.nn as nn
import torch.nn.functional as F

# 復用你倉庫裡的積木：LayerNorm / FeedForward / 通道自注意力(ChA) / 空間注意力(OCDA, CDA)
from .diffatts import LayerNorm, FeedForward, ChA, OCDA, CDA  # ChA/OCDA/CDA/FFN 都在 diffatts.py

class CrossGate(nn.Module):
    """雙向交叉門控：用對側特徵生成 gate 調制本側"""
    def __init__(self, c):
        super().__init__()
        self.gx = nn.Sequential(nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())
        self.gy = nn.Sequential(nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())
    def forward(self, x, y):
        return x * self.gx(y), y * self.gy(x)

class DualAttentionInteractiveBlockCore(nn.Module):
    """
    DAIB 核心：左路=自注意力(通道)，右路=空間注意力(CDA/OCDA) → 交叉門控 → 各自 GDFN → 殘差縮放
    注意：這個 Core 只做雙流互動，不改變特徵解析度與通道數。
    """
    def __init__(
        self,
        dim: int,
        spatial_attn_type: str = "OCDA",   # "CDA"=全域, "OCDA"=重疊窗
        num_spatial_heads: int = 4,
        num_channel_heads: int = 8,        # ChA 的 head 數
        depth_for_attn: int = 1,           # 傳給 CDA/OCDA 的 depth
        ffn_expansion: float = 2.0,
        layernorm_type: str = "WithBias",
        window_size: int = 8,
        overlap_ratio: float = 0.5,
    ):
        super().__init__()
        self.norm = LayerNorm(dim, layernorm_type)

        # 左：通道自注意力（ChA）
        self.self_attn_left = ChA(dim, num_heads=num_channel_heads, bias=False)  # ChA【:contentReference[oaicite:3]{index=3}】

        # 右：空間注意力（CDA or OCDA）
        if spatial_attn_type == "CDA":
            self.spatial_attn_right = CDA(dim, num_heads=num_spatial_heads, depth=depth_for_attn, bias=False)  # CDA【:contentReference[oaicite:4]{index=4}】【:contentReference[oaicite:5]{index=5}】
        elif spatial_attn_type == "OCDA":
            self.spatial_attn_right = OCDA(
                dim, num_heads=num_spatial_heads, depth=depth_for_attn,
                window_size=window_size, overlap_ratio=overlap_ratio, bias=False
            )  # OCDA【:contentReference[oaicite:6]{index=6}】
        else:
            raise ValueError("spatial_attn_type must be 'CDA' or 'OCDA'.")

        self.cross = CrossGate(dim)
        self.ffn_left  = FeedForward(dim, ffn_expansion, bias=False)   # GDFN【:contentReference[oaicite:7]{index=7}】
        self.ffn_right = FeedForward(dim, ffn_expansion, bias=False)   # GDFN【:contentReference[oaicite:8]{index=8}】

        # 可學殘差縮放（與 DSIT 模塊習慣一致）
        self.a = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x_left, x_right):
        xl = self.norm(x_left)
        xr = self.norm(x_right)

        xl_attn = self.self_attn_left(xl)      # 通道向自注意力（左）
        xr_attn = self.spatial_attn_right(xr)  # 空間向注意力（右）

        x_left  = x_left  + self.a * xl_attn
        x_right = x_right + self.a * xr_attn

        xl_g, xr_g = self.cross(x_left, x_right)
        x_left  = x_left  + self.b * self.ffn_left(xl_g)
        x_right = x_right + self.b * self.ffn_right(xr_g)
        return x_left, x_right


class DualAttentionInteractiveFuse(nn.Module):
    """
    可直接替換 FuseGated 的「DAIB 融合」版本：
    forward(x_topdown, x_lateral) -> fused(single tensor)

    步驟：
    1) 上採樣 x_topdown 到 x_lateral 尺寸（對齊 FuseGated 的行為）
    2) DAIB Core：左=自注意力(通道)、右=空間注意力
    3) 融合：concat 後 1×1 降回 dim，並接一個 3×3 混合卷積（與原 FuseGated 的 mix 對齊）
    """
    def __init__(
        self,
        dim: int,
        spatial_attn_type: str = "OCDA",
        num_spatial_heads: int = 4,
        num_channel_heads: int = 8,
        depth_for_attn: int = 1,
        ffn_expansion: float = 2.0,
        layernorm_type: str = "WithBias",
        window_size: int = 8,
        overlap_ratio: float = 0.5,
    ):
        super().__init__()
        self.core = DualAttentionInteractiveBlockCore(
            dim=dim,
            spatial_attn_type=spatial_attn_type,
            num_spatial_heads=num_spatial_heads,
            num_channel_heads=num_channel_heads,
            depth_for_attn=depth_for_attn,
            ffn_expansion=ffn_expansion,
            layernorm_type=layernorm_type,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
        )
        self.fuse1x1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.mix = nn.Sequential(                  # 模仿原 FuseGated 的「mix」段
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_topdown, x_lateral):
        # 1) 尺寸對齊（與 FuseGated 完全一致）
        x_td = F.interpolate(x_topdown, size=x_lateral.shape[-2:], mode="bilinear", align_corners=False)

        # 2) DAIB 互動
        xl, xr = self.core(x_td, x_lateral)

        # 3) 融合成單一路：concat→1×1→3×3混合
        fused = self.fuse1x1(torch.cat([xl, xr], dim=1))
        return self.mix(fused)
