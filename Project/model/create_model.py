import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from .loss.focal import FocalLoss
from .loss.dice import DICELoss
import importlib
from lejepa.univariate import EppsPulley
from lejepa.multivariate import SlicingUnivariateTest




def _import_segmodel(segmodel_variant: str):
    """
    segmodel_variant:
      - "base"  -> 從 .model 讀 SegModel
      - "dino"  -> 從 .model_dino 讀 SegModel
      - 其他字串 -> 視為模組名（相對於目前 package），例如 ".model_dinov3_fused"
    """
    if segmodel_variant in (None, "", "base"):
        module_name = __package__ + ".model"
    elif segmodel_variant == "dino":
        module_name = __package__ + ".model_dino"
    elif segmodel_variant == "dino_iaff":
        module_name = __package__ + ".model_dino_iAFF"
    elif segmodel_variant == "dino_afcm":
        module_name = __package__ + ".model_dino_afcm"
    elif segmodel_variant == "dino_freqfusion_lejepa":
        module_name = __package__ + ".model_dino_van"
    else:
        module_name = __package__ + (segmodel_variant if segmodel_variant.startswith(".") else "." + segmodel_variant)

    module = importlib.import_module(module_name)
    return module.SegModel


def get_model(backbone_name="mobilenetv2", fpn_channels=128, segmodel_variant="base", **kwargs):
    SegModel = _import_segmodel(segmodel_variant)
    model = SegModel(backbone_name, fpn_channels, **kwargs)
    return model


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device(
            "cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu"
        )
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        seg_variant = getattr(opt, "segmodel", os.getenv("SEGMODEL_VARIANT", "base"))

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            num_heads=opt.num_heads,
            num_points=opt.num_points,
            kernel_layers=opt.kernel_layers,
            dropout_rate=opt.dropout_rate,
            init_type=opt.init_type,
            segmodel_variant=seg_variant, 
        )


        self.focal = FocalLoss(alpha=[1] * 14, gamma=opt.gamma)
        self.dice = DICELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=5e-7
        )
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.name, opt.backbone)
        self.model.cuda()

        self.sig_uni = EppsPulley(n_points=17)
        self.sigreg = SlicingUnivariateTest(
            univariate_test=self.sig_uni,
            num_slices=256,
        )
        self.lambda_sig = 0.1   # 你可以 tune


        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label):
        preds, fea1, fea2 = self.model(x1, x2)
        focal = self.focal(preds[0], label)
        dice = self.dice(preds[0], label)
        for i in range(1, len(preds)):
            focal += self.focal(preds[i], label)
            dice += 0.5 * self.dice(preds[i], label)

        # ----------------------------
        #   ★ 加入 LeJEPA / SIGReg ★
        # ----------------------------
        # fea1 = (t1_p2, t1_p3, t1_p4, t1_p5)
        t1_p5 = fea1[3]   # shape [B, C, H, W]
        t2_p5 = fea2[3]

        # Global Average Pooling → [B, C]
        z1 = F.adaptive_avg_pool2d(t1_p5, 1).squeeze(-1).squeeze(-1)
        z2 = F.adaptive_avg_pool2d(t2_p5, 1).squeeze(-1).squeeze(-1)

        # 分別對兩條支路做 isotropic Gaussian regularization
        sig1 = self.sigreg(z1)
        sig2 = self.sigreg(z2)

        sig_loss = (sig1 + sig2) * 0.5 * self.lambda_sig

        # ----------------------------
        #   原本就有的 placeholder
        # ----------------------------
        infonce = torch.tensor(0.0, dtype=focal.dtype, device=focal.device)

        return preds[0], focal, dice, sig_loss

    # def forward(self, x1, x2, label):
    #     preds, fea1, fea2 = self.model(x1, x2)
    #     focal = self.focal(preds[0], label)
    #     dice = self.dice(preds[0], label)
    #     for i in range(1, len(preds)):
    #         focal += self.focal(preds[i], label)
    #         dice += 0.5 * self.dice(preds[i], label)

    #     infonce = torch.tensor(0.0, dtype=focal.dtype, device=focal.device)

    #     return preds[0], focal, dice, infonce

    @torch.inference_mode()
    def inference(self, x1, x2):
        pred = self.model._forward(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(
                save_path, map_location=self.device, weights_only=True
            )
            network.load_state_dict(checkpoint["network"], strict=False)
            print("load pre-trained")

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = "%s_%s_best.pth" % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()
