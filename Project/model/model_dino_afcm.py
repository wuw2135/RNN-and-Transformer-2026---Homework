import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbone.mobilenetv2 import mobilenet_v2, ConvBNReLU
from .backbone.lwganet import LWGANet, Stem
from .blocks.heads import GatedResidualUpHead
from .blocks.fpn import FPN
from .blocks.diffatts import TransformerBlock
from .blocks.afcm import AFCM

class DINOv3_ViT_L16(nn.Module):
    def __init__(self, in_chans=3, checkpoint_path=None):
        super().__init__()

        self.vit = timm.create_model(
            'vit_large_patch16_dinov3.sat493m', pretrained=(checkpoint_path is None),
            features_only=False, in_chans=3
        )

        self.embed_dim = self.vit.embed_dim  # 1024
        self.channels = [self.embed_dim] * 5  # 為了與 FPN 介面對齊

    def _tokens_to_map(self, x, H, W):
        """
        x: [B, N, C]（可能包含 CLS、register/distilled 等特殊 tokens）
        只取最後的 h*w 個 patch tokens 來還原空間特徵圖。
        """
        B, N, C = x.shape
        h, w = H // 16, W // 16
        num_patch = h * w
        if N < num_patch:
            raise RuntimeError(f"Token數量不足：N={N}, 需要至少 {num_patch} 個 patch tokens")
        # 取最後的 patch tokens（通常 patch tokens 在序列結尾）
        patch_tokens = x[:, -num_patch:, :]                 # [B, h*w, C]
        patch_tokens = patch_tokens.transpose(1, 2).contiguous().view(B, C, h, w)
        return patch_tokens

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.vit.forward_features(x)

        # 兼容不同 timm 版本的回傳型態
        if isinstance(out, dict) and "x" in out:
            out = out["x"]  # 有些版本會回 dict

        if out.dim() == 3:  # [B, N, C]
            p4 = self._tokens_to_map(out, H, W)   # 1/16, 128→8×8
        elif out.dim() == 4:  # [B, C, h, w]
            p4 = out
        else:
            raise RuntimeError(f"不支援的 forward_features 輸出型態: {out.shape}")

        p3 = F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)  # 1/8
        p2 = F.interpolate(p4, scale_factor=4, mode='bilinear', align_corners=False)  # 1/4
        p5 = F.avg_pool2d(p4, 2, 2)                                                   # 1/32
        dummy = p2
        return [p2, p3, p4, p5]


def get_backbone(backbone_name):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    elif backbone_name == "lwganet":
        backbone = LWGANet(
            in_chans=3,
            stem_dim=64,
            depths=(1, 2, 4, 2),
            att_kernel=(9, 9, 9, 9),
            fork_feat=True,
        )
        backbone.channels = [64 * 2**i for i in range(4)]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch=12,
        backbone="mobilenetv2",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone_name=backbone)
        if backbone == "mobilenetv2":
            first_block = self.backbone.features[0]
            out_ch = first_block[0].out_channels
            stride = first_block[0].stride[0]
            self.backbone.features[0] = ConvBNReLU(in_ch, out_ch, stride=stride)
        elif backbone == "lwganet":
            stem_dim = self.backbone.Stem.proj.out_channels
            self.backbone.Stem = Stem(in_chans=in_ch, stem_dim=stem_dim)

        self.fpn = FPN(
            in_channels=(self.backbone.channels[-4:]),
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )

    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4

        return fea


class FuseGated(nn.Module):
    def __init__(self, dim, gate_scale=1.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.gate_scale = nn.Parameter(torch.tensor(gate_scale, dtype=torch.float32))
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1)) * self.gate_scale
        fused = x2 + g * x1
        return self.mix(fused)


class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        num_classes=14,
        n_layers=[1, 2, 2, 1],
        dropout_rate=0.1,
        size=128,
        afcm_down=4,
        **kwargs,
    ):
        super().__init__()
        self.size = (size, size)

        self.p5_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p4_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p3_head = nn.Conv2d(fpn_channels, num_classes, 1)
        self.p2_head = nn.Conv2d(fpn_channels, num_classes, 1)

        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="WithBias",
                )
                for _ in range(n_layers[0])
            ]
        )
        self.tb4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="WithBias",
                )
                for _ in range(n_layers[1])
            ]
        )
        self.tb3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=1,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="WithBias",
                )
                for _ in range(n_layers[2])
            ]
        )
        self.tb2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=0,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="WithBias",
                )
                for _ in range(n_layers[3])
            ]
        )

        self.head = GatedResidualUpHead(
            fpn_channels, num_classes, dropout_rate=dropout_rate
        )

        self.a5 = nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False)
        self.a4 = nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False)
        self.a3 = nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False)
        self.a2 = nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False)

        self.afcm5 = AFCM(fpn_channels, kernel_size=7, alpha=0.5, gamma=0.5, spatial_downsample=afcm_down)
        self.afcm4 = AFCM(fpn_channels, kernel_size=7, alpha=0.5, gamma=0.5, spatial_downsample=afcm_down)
        self.afcm3 = AFCM(fpn_channels, kernel_size=7, alpha=0.5, gamma=0.5, spatial_downsample=afcm_down//2 if afcm_down>1 else 1)
        self.afcm2 = AFCM(fpn_channels, kernel_size=7, alpha=0.5, gamma=0.5, spatial_downsample=max(1, afcm_down//2))

    def forward(self, x1s, x2s):
        ### Extract backbone features
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        p5 = self.a5(torch.cat([t1_p5, t2_p5], 1))
        p4 = self.a4(torch.cat([t1_p4, t2_p4], 1))
        p3 = self.a3(torch.cat([t1_p3, t2_p3], 1))
        p2 = self.a2(torch.cat([t1_p2, t2_p2], 1))

        p5 = self.afcm5(p5)   # 1/32 尺度，張量小、代價低
        p4 = self.afcm4(p4)   # 1/16
        p3 = self.afcm3(p3)   # 1/8
        p2 = self.afcm2(p2) 

        fea_p5 = self.tb5(p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, p2)
        fea_p2 = self.tb2(fea_p2)
        pred_p2 = self.p2_head(fea_p2)
        pred = self.head(fea_p2)
        # pred = F.interpolate(pred, size=self.size, mode="bilinear", align_corners=False) 

        pred_p2 = F.interpolate(
            pred_p2, size=self.size, mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=self.size, mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=self.size, mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=self.size, mode="bilinear", align_corners=False
        )

        return pred, pred_p2, pred_p3, pred_p4, pred_p5




class SegModel(nn.Module):
    def __init__(self, backbone="mobilenetv2", fpn_channels=96, **kwargs):
        super().__init__()
        self.encoder1 = Encoder(in_ch=12, backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.encoder2 = Encoder(in_ch=2,  backbone=backbone, fpn_channels=fpn_channels, **kwargs)

        self.encoder_dino = DINOv3_ViT_L16(in_chans=3, checkpoint_path=None)  # 只餵 RGB
        # 取 DINO 的通道數（ViT-L/16 是 1024）
        dino_c = getattr(self.encoder_dino, "embed_dim", None)
        if dino_c is None:
            dino_c = getattr(self.encoder_dino, "channels", [fpn_channels])[-1]

        # 將 DINO 的每一層特徵投影到 fpn_channels，確保和 CNN 對齊
        self.dino_proj_p2 = nn.Conv2d(dino_c, fpn_channels, kernel_size=1, bias=False)
        self.dino_proj_p3 = nn.Conv2d(dino_c, fpn_channels, kernel_size=1, bias=False)
        self.dino_proj_p4 = nn.Conv2d(dino_c, fpn_channels, kernel_size=1, bias=False)
        self.dino_proj_p5 = nn.Conv2d(dino_c, fpn_channels, kernel_size=1, bias=False)

        # 門控融合（維持原本設計）
        self.fuse1_p2 = FuseGated(fpn_channels)
        self.fuse1_p3 = FuseGated(fpn_channels)
        self.fuse1_p4 = FuseGated(fpn_channels)
        self.fuse1_p5 = FuseGated(fpn_channels)

        self.fuse2_p2 = FuseGated(fpn_channels)
        self.fuse2_p3 = FuseGated(fpn_channels)
        self.fuse2_p4 = FuseGated(fpn_channels)
        self.fuse2_p5 = FuseGated(fpn_channels)

        self.detector = Detector(fpn_channels=fpn_channels, **kwargs)


    @torch.inference_mode()
    def _forward(self, x1, x2):
        # for inference
        # x1 = self.upsample1(x1)
        # x2 = self.upsample2(x2)
        fea1 = self.encoder1(x1)
        fea2 = self.encoder2(x2)

        x_rgb = x1[:, [1, 2, 3], :, :]
        with torch.no_grad():
            feaD = self.encoder_dino(x_rgb)
        d_p2, d_p3, d_p4, d_p5 = feaD[-4:]

        d_p2 = self.dino_proj_p2(d_p2)
        d_p3 = self.dino_proj_p3(d_p3)
        d_p4 = self.dino_proj_p4(d_p4)
        d_p5 = self.dino_proj_p5(d_p5)

        t1_p2 = self.fuse1_p2(fea1[0], d_p2)
        t1_p3 = self.fuse1_p3(fea1[1], d_p3)
        t1_p4 = self.fuse1_p4(fea1[2], d_p4)
        t1_p5 = self.fuse1_p5(fea1[3], d_p5)

        t2_p2 = self.fuse2_p2(fea2[0], d_p2)
        t2_p3 = self.fuse2_p3(fea2[1], d_p3)
        t2_p4 = self.fuse2_p4(fea2[2], d_p4)
        t2_p5 = self.fuse2_p5(fea2[3], d_p5)

        pred, _, _, _, _ = self.detector([t1_p2, t1_p3, t1_p4, t1_p5],
                              [t2_p2, t2_p3, t2_p4, t2_p5])
        return pred

    def forward(self, x1, x2):
        # x1 = self.upsample1(x1)
        # x2 = self.upsample2(x2)
        fea1 = self.encoder1(x1)
        fea2 = self.encoder2(x2)

        x_rgb = x1[:, [1, 2, 3], :, :]   # band2=blue, band3=green, band4=red
        with torch.no_grad():
            feaD = self.encoder_dino(x_rgb)
        d_p2, d_p3, d_p4, d_p5 = feaD[-4:]

        d_p2 = self.dino_proj_p2(d_p2)
        d_p3 = self.dino_proj_p3(d_p3)
        d_p4 = self.dino_proj_p4(d_p4)
        d_p5 = self.dino_proj_p5(d_p5)

        t1_p2 = self.fuse1_p2(fea1[0], d_p2)
        t1_p3 = self.fuse1_p3(fea1[1], d_p3)
        t1_p4 = self.fuse1_p4(fea1[2], d_p4)
        t1_p5 = self.fuse1_p5(fea1[3], d_p5)

        t2_p2 = self.fuse2_p2(fea2[0], d_p2)
        t2_p3 = self.fuse2_p3(fea2[1], d_p3)
        t2_p4 = self.fuse2_p4(fea2[2], d_p4)
        t2_p5 = self.fuse2_p5(fea2[3], d_p5)

        preds = self.detector([t1_p2, t1_p3, t1_p4, t1_p5],
                            [t2_p2, t2_p3, t2_p4, t2_p5])
        return preds, (t1_p2, t1_p3, t1_p4, t1_p5), (t2_p2, t2_p3, t2_p4, t2_p5)


