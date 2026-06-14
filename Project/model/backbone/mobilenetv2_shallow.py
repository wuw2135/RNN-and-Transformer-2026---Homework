import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1
    ):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    dilation=dilation,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Shallow(nn.Module):
    def __init__(
        self,
        pretrained=True,
        width_mult=1.0,
        replace_stride_with_dilation=False,
    ):
        super(MobileNetV2Shallow, self).__init__()
        block = InvertedResidual

        # Deep path (standard MobileNetV2 feature extractor)
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 1, 2] if replace_stride_with_dilation else [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 1, 2] if replace_stride_with_dilation else [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]

        features = [ConvBNReLU(3, input_channel, stride=2)]
        in_ch = input_channel
        for t, c, n, s, d in inverted_residual_setting:
            out_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(
                    block(
                        in_ch,
                        out_ch,
                        stride,
                        expand_ratio=t,
                        dilation=dilation,
                    )
                )
                in_ch = out_ch
        features.append(ConvBNReLU(in_ch, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # Shallow 1x path: mirror first two layers but keep stride=1 at input
        shallow_layers = []
        shallow_in = int(32 * width_mult)
        shallow_layers.append(ConvBNReLU(3, shallow_in, stride=1))  # keep 1x
        # First inverted residual block t=1 -> c=16 at 1x
        shallow_out = int(16 * width_mult)
        shallow_layers.append(block(shallow_in, shallow_out, stride=1, expand_ratio=1))
        self.shallow = nn.Sequential(*shallow_layers)

        self.channels = [int(16 * width_mult), int(320 * width_mult)]

        # weight initialization (same as original)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["mobilenet_v2"], progress=True
            )
            # Load what matches into the deep path; ignore unmatched shallow keys
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # 1x feature from shallow path
        feat_1x = self.shallow(x)

        feats = [feat_1x]
        y = x
        for idx, m in enumerate(self.features):
            y = m(y)
            if idx in (1, 3, 6, 13, 17):
                feats.append(y)

        return feats


def mobilenet_v2_shallow(
    pretrained=True, progress=True, replace_stride_with_dilation=False, **kwargs
):
    # keep signature similar; progress currently unused because we call in __init__
    model = MobileNetV2Shallow(
        pretrained=pretrained,
        replace_stride_with_dilation=replace_stride_with_dilation,
        **kwargs,
    )
    return model
