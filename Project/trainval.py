import torch
from option import Options
from data.dataset import DataLoader
from model import create_model
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter, get_confuse_matrix
import os
import json
import numpy as np
import random
from datetime import datetime
from util.util import (
    make_numpy_grid,
    de_norm,
    logits_batch_to_rgb,
    DEFAULT_PALETTE,
    ema_alpha_update,
)
import matplotlib.pyplot as plt


def palette_label(label):
    palette = torch.as_tensor(DEFAULT_PALETTE, dtype=torch.uint8, device=label.device)
    return palette[label]


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  #!
    torch.backends.cudnn.benchmark = True  #!
    torch.backends.cudnn.enabled = True  #! for accelerating training


def exp_decreasing_map(
    x: int, n: int, min_val: float, max_val: float, k: float = 5.0
) -> float:
    if not (0 <= x <= n):
        raise ValueError("x must be in the range [0, n]")
    ratio = x / n
    return min_val + (max_val - min_val) * math.exp(-k * ratio)


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = "val"
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = "train"

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.l1_weight = getattr(opt, "l1_weight", 0.0)

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=14)
        self.alpha = 0.5

        self.log_path = os.path.join(self.model.save_dir, "record.txt")
        self.vis_path = os.path.join(self.model.save_dir, opt.vis_path)
        os.makedirs(self.vis_path, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("# Record of training/validation metrics\n")
                f.write(
                    "# name: %s | backbone: %s\n"
                    % (opt.name, getattr(opt, "backbone", "NA"))
                )
                f.write("# time,epoch,train_loss,train_focal,train_dice,lr,")
                f.write("val_metrics(json)\n")

    def _append_log_line(self, epoch: int, train_stats: dict, val_scores: dict):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
            f"{ts},{epoch},"
            f"{train_stats.get('loss', float('nan')):.6f},"
            f"{train_stats.get('focal', float('nan')):.6f},"
            f"{train_stats.get('dice', float('nan')):.6f},"
            f"{train_stats.get('infonce', float('nan')):.6f},"
            f"{train_stats.get('l1', float('nan')):.6f},"
            f"{train_stats.get('lr', float('nan')):.8f},"
            + json.dumps(val_scores, ensure_ascii=False)
            + "\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _plot_result(self, x1, x2, pred, target, epoch, stage):
        x1 = x1[:, [3, 2, 1], :, :]
        x2 = x2[:, 0, :, :]
        if pred.ndim == 4:
            pred = torch.argmax(pred, dim=1)
        if stage == "train":
            vis_input = make_numpy_grid(
                de_norm(
                    x1[0:8],
                    self.train_data.dataset.mean_msi[[3, 2, 1]].unsqueeze(0),
                    self.train_data.dataset.std_msi[[3, 2, 1]].unsqueeze(0),
                )
            )
            vis_input2 = make_numpy_grid(
                de_norm(
                    x2[0:8],
                    self.train_data.dataset.mean_sar[0].unsqueeze(0),
                    self.train_data.dataset.std_sar[0].unsqueeze(0),
                )
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
            )
        else:
            vis_input = make_numpy_grid(
                de_norm(
                    x1[0:8],
                    self.val_data.dataset.mean_msi[[3, 2, 1]].unsqueeze(0),
                    self.val_data.dataset.std_msi[[3, 2, 1]].unsqueeze(0),
                )
            )
            vis_input2 = make_numpy_grid(
                de_norm(
                    x2[0:8],
                    self.val_data.dataset.mean_sar[0].unsqueeze(0),
                    self.val_data.dataset.std_sar[0].unsqueeze(0),
                )
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
            )
        vis_pred = make_numpy_grid(logits_batch_to_rgb(pred[0:8]))
        vis_gt = make_numpy_grid(palette_label(target[0:8]).permute(0, 3, 1, 2))
        vis = np.concatenate(
            [vis_input, vis_input2, vis_pred / 255, vis_gt / 255], axis=0
        )
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_path, f"{stage}_" + str(epoch) + ".jpg")
        plt.imsave(file_name, vis)

    def _compute_l1_on_classifier(self):
        """
        對 segmentation detector 裡的各個分類 head (p2_head~p5_head) 做 L1 regularization。
        """
        l1 = 0.0
        # self.model 是 Model wrapper；真正的 segmentation network 在 self.model.model
        for name, p in self.model.model.named_parameters():
            if not p.requires_grad:
                continue
            # 只挑 detector 裡 p2/p3/p4/p5 的 head
            if ("detector.p2_head" in name or
                "detector.p3_head" in name or
                "detector.p4_head" in name or
                "detector.p5_head" in name):
                l1 = l1 + p.abs().mean()
        return l1

    def train(self, epoch):
        tbar = tqdm(self.train_data, ncols=150)
        opt.phase = "train"
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0
        _infonce_loss = 0.0
        _l1_loss = 0.0          # 新增：記錄 L1
        last_lr = self.optimizer.param_groups[0]["lr"]


        for i, data in enumerate(tbar):
            self.model.model.train()
            pred, focal, dice, infonce = self.model(
                data["img1"].cuda(), data["img2"].cuda(), data["label"].cuda()
            )

            lam = getattr(self.opt, "contrastive_weight", 0.1)

            if self.l1_weight > 0.0:
                l1_reg = self._compute_l1_on_classifier()
            else:
                l1_reg = 0.0

            loss = focal * self.alpha + dice + lam * infonce + self.l1_weight * l1_reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            _infonce_loss += infonce.item()
            _l1_loss += 0.0   # 新增：記錄 L1 值
            last_lr = self.optimizer.param_groups[0]["lr"]
            del loss

            tbar.set_description(
                "Loss: %.3f, Focal: %.3f, Dice: %.3f, InfoNCE: %.3f, L1: %.6f, LR: %.6f"
                % (
                    _loss / (i + 1),
                    _focal_loss / (i + 1),
                    _dice_loss / (i + 1),
                    _infonce_loss / (i + 1),
                    _l1_loss / (i + 1),
                    last_lr,
                )
            )


            if i == len(tbar) - 1:
                self._plot_result(
                    data["img1"], data["img2"], pred, data["label"], epoch, "train"
                )
        self.schedular.step()

        n = max(1, i + 1)

        return {
            "loss": _loss / n,
            "focal": _focal_loss / n,
            "dice": _dice_loss / n,
            "infonce": _infonce_loss / n,
            "l1": _l1_loss / n,
            "lr": last_lr,
        }


    def val(self, epoch):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        opt.phase = "val"
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(
                    _data["img1"].cuda(), _data["img2"].cuda()
                )
                val_target = _data["label"].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(
                    pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy()
                )
                if i == len(tbar) - 1:
                    self._plot_result(
                        _data["img1"],
                        _data["img2"],
                        val_pred,
                        _data["label"],
                        epoch,
                        "val",
                    )
            val_scores = self.running_metric.get_scores()
            message = "(phase: %s) " % (self.opt.phase)
            for k, v in val_scores.items():
                message += "%s: %.3f " % (k, v * 100)
            print(message)

        if val_scores.get("miou", 0.0) >= self.previous_best:
            self.model.save(self.opt.name, self.opt.backbone)
            self.previous_best = val_scores["miou"]

        return val_scores


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=getattr(opt, "seed", 77777))

    alpha_np = np.array([1.0] * 14)
    for epoch in range(1, opt.num_epochs + 1):
        print(
            "\n==> Name %s, Epoch %i, previous best = %.3f"
            % (opt.name, epoch, trainval.previous_best * 100)
        )
        print(alpha_np)
        train_stats = trainval.train(epoch)
        val_scores = trainval.val(epoch)

        # update focal alpha
        alpha_np = ema_alpha_update(
            val_scores, alpha_np, momentum=0.99, clamp=(0.8, 1.5)
        )
        trainval.model.focal.set_alpha(alpha_np)

        trainval._append_log_line(epoch, train_stats, val_scores)

    print("Done!")
