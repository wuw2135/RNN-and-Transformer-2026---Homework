import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbone.mobilenetv2 import mobilenet_v2, ConvBNReLU
from .backbone.lwganet import LWGANet, Stem
from .blocks.heads import GatedResidualUpHead
from .blocks.fpn import FPN
from .blocks.diffatts import TransformerBlock
from mmcv.ops.carafe import CARAFEPack


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

    def forward(self, x1s, x2s):
        ### Extract backbone features
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        p5 = self.a5(torch.cat([t1_p5, t2_p5], 1))
        p4 = self.a4(torch.cat([t1_p4, t2_p4], 1))
        p3 = self.a3(torch.cat([t1_p3, t2_p3], 1))
        p2 = self.a2(torch.cat([t1_p2, t2_p2], 1))

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
        self.encoder1 = Encoder(
            in_ch=12, backbone=backbone, fpn_channels=fpn_channels, **kwargs
        )
        self.encoder2 = Encoder(
            in_ch=2, backbone=backbone, fpn_channels=fpn_channels, **kwargs
        )
        self.detector = Detector(fpn_channels=fpn_channels, **kwargs)

    @torch.inference_mode()
    def _forward(self, x1, x2):
        # for inference
        fea1 = self.encoder1(x1)
        fea2 = self.encoder2(x2)
        pred, _, _, _, _ = self.detector(fea1, fea2)
        return pred

    def forward(self, x1, x2):
        # for training
        fea1 = self.encoder1(x1)
        fea2 = self.encoder2(x2)
        preds = self.detector(fea1, fea2)
        return preds, fea1, fea2  # pred, pred_p2, pred_p3, pred_p4, pred_p5
