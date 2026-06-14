from .transform import Transforms
import numpy as np
import os
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from scipy.signal import convolve2d


@torch.no_grad()
def lee_filter(img: torch.Tensor, win: int = 5, eps: float = 1e-6) -> torch.Tensor:
    C, H, W = img.shape
    x = img.unsqueeze(0)  # -> (1,C,H,W)
    pad = win // 2

    # depthwise conv kernel: (C,1,win,win)
    kernel = torch.ones((C, 1, win, win), dtype=torch.float) / (win * win)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    mean = F.conv2d(x_pad, kernel, stride=1, padding=0, groups=C)  # (1,C,H,W)
    mean_sq = F.conv2d(
        x_pad * x_pad, kernel, stride=1, padding=0, groups=C
    )  # (1,C,H,W)
    var = torch.clamp_min(mean_sq - mean * mean, 0.0)

    var_flat = var.view(C, -1)
    noise_var = var_flat.median(dim=1, keepdim=True).values.view(C, 1, 1)
    noise_var = noise_var.unsqueeze(0)  # (1,C,1,1)

    w = var / (var + noise_var + eps)
    out = mean + w * (x - mean)  # (1,C,H,W)
    return out.squeeze(0)


def to_db(x, clip_min=1e-8):
    return 10.0 * torch.log10(x.clamp_min(clip_min))


def make_dataset(dir, phase):
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    assert phase in ["train", "val", "test"], "phase must be 'train' or 'val'"
    img_paths = []
    names = []

    names = os.listdir(dir)
    names.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if phase == "train":
        names = names[: int(len(names) * 0.8)]
    elif phase == "val":
        names = names[int(len(names) * 0.8) :]

    for name in names:
        img_path = os.path.join(dir, name)
        img_paths.append(img_path)

    return img_paths, names


def build_train_transform(opt):
    aug_mode = getattr(opt, "aug_mode", "full")

    if aug_mode == "none":
        return None
    if aug_mode == "noise_erase":
        return Transforms(
            p_hflip=0.0,
            p_vflip=0.0,
            p_rotate=0.0,
            p_crop=0.0,
            p_noise=getattr(opt, "aug_p_noise", 0.5),
            p_erase=getattr(opt, "aug_p_erase", 0.5),
        )
    if aug_mode == "full":
        return Transforms(
            p_noise=getattr(opt, "aug_p_noise", 0.5),
            p_erase=getattr(opt, "aug_p_erase", 0.5),
        )

    raise ValueError(f"Unsupported aug_mode: {aug_mode}")


class Load_Dataset(Dataset):
    def __init__(self, opt, transform=None):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        self.dir1 = os.path.join(opt.dataroot, opt.dataset, "msi_images")
        self.msi_paths, self.fnames = make_dataset(self.dir1, opt.phase)

        self.dir2 = os.path.join(opt.dataroot, opt.dataset, "sar_images")
        self.sar_paths, _ = make_dataset(self.dir2, opt.phase)

        self.label_paths = None
        if opt.phase in ["train", "val"]:
            self.dir_label = os.path.join(opt.dataroot, opt.dataset, "masks")
            self.label_paths, _ = make_dataset(self.dir_label, opt.phase)

        self.dataset_size = len(self.msi_paths)

        self.mean_msi = (
            torch.from_numpy(
                np.load(os.path.join(opt.stat_dir, f"means_msi_train.npy"))
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.mean_sar = (
            torch.from_numpy(
                np.load(os.path.join(opt.stat_dir, f"means_sar_train_db.npy"))
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.std_msi = (
            torch.from_numpy(np.load(os.path.join(opt.stat_dir, f"stds_msi_train.npy")))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        self.std_sar = (
            torch.from_numpy(
                np.load(os.path.join(opt.stat_dir, f"stds_sar_train_db.npy"))
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        if transform is None:
            transform = build_train_transform(opt)

        self.transform = None
        if transform is not None:
            self.transform = transforms.Compose([transform])

        self.n_class = getattr(opt, "n_class", 14)     # 預設 14（與訓練程式一致）  :contentReference[oaicite:2]{index=2}
        self.cb_power = getattr(opt, "cb_power", 1.0)  # 反比權重 γ（可調）
        self.eps = 1e-8

        if self.label_paths is not None and self.opt.phase == "train":
            # 統計整體類別頻率（像素級）
            self.class_hist = self._compute_class_hist()
            # 為每張影像計算取樣權重
            self.sample_weights = self._compute_image_weights()
        else:
            self.class_hist = None
            self.sample_weights = None
    
    # --- 新增：統計整體類別直方圖 ---
    def _compute_class_hist(self):
        hist = np.zeros(self.n_class, dtype=np.float64)
        for p in self.label_paths:
            lab = tifffile.imread(p).astype(np.int64)
            # 安全裁切非法值
            lab = np.clip(lab, 0, self.n_class - 1)
            h, _ = np.histogram(lab, bins=self.n_class, range=(0, self.n_class))
            hist += h
        # 避免全零
        hist = np.maximum(hist, 1.0)
        return hist  # 各類別像素數

    # --- 新增：把整體頻率轉成每張影像的抽樣權重 ---
    def _compute_image_weights(self):
        # 反比類別權重 (1/f)^gamma
        inv_cls = (1.0 / (self.class_hist + self.eps)) ** self.cb_power
        inv_cls = inv_cls / inv_cls.sum()  # 正規化僅為尺度穩定

        weights = []
        for p in self.label_paths:
            lab = tifffile.imread(p).astype(np.int64)
            lab = np.clip(lab, 0, self.n_class - 1)
            h, _ = np.histogram(lab, bins=self.n_class, range=(0, self.n_class))
            if h.sum() == 0:
                w = 0.0
            else:
                p_i = h / (h.sum() + self.eps)          # 影像內類別比例
                w = float((p_i * inv_cls).sum())        # 加權平均
            weights.append(w)

        w = np.asarray(weights, dtype=np.float64)
        # 若出現 NaN/Inf 或全零，回退為均勻
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(weights), dtype=np.float64)

        # 正規化到筆數
        w = w / w.sum() * len(w)
        return torch.as_tensor(w, dtype=torch.double)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        msi_path = self.msi_paths[index]
        fname = self.fnames[index]
        img1 = tifffile.imread(msi_path).astype(np.float32)

        sar_path = self.sar_paths[index]
        img2 = tifffile.imread(sar_path).astype(np.float32)

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        img2 = to_db(lee_filter(img2))

        if self.label_paths is not None:
            label_path = self.label_paths[index]
            label = tifffile.imread(label_path).astype(np.float32)
            label = torch.from_numpy(label).unsqueeze(0)

        if self.opt.phase == "train" and self.transform is not None:
            _data = self.transform({"img1": img1, "img2": img2, "label": label})
            img1, img2, label = _data["img1"], _data["img2"], _data["label"]

        img1 = self.normalize(img1, self.mean_msi, self.std_msi)
        img2 = self.normalize(img2, self.mean_sar, self.std_sar)

        if self.label_paths is not None:
            input_dict = {
                "img1": img1.float(),
                "img2": img2.float(),
                "label": label.squeeze(0).long(),
                "fname": fname,
            }
        else:
            input_dict = {
                "img1": img1.float(),
                "img2": img2.float(),
                "fname": fname,
            }

        return input_dict

    def normalize(self, img, mean, std):
        img = (img - mean) / std
        return img


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)

        use_class_balance = getattr(opt, "class_balance", True) and opt.phase == "train"
        sampler = None
        shuffle = opt.phase == "train"

        # if use_class_balance and getattr(self.dataset, "sample_weights", None) is not None:
        #     sampler = WeightedRandomSampler(
        #         weights=self.dataset.sample_weights,
        #         num_samples=len(self.dataset),     # 每個 epoch 看到與原先相同數量的樣本
        #         replacement=True
        #     )
        #     shuffle = False  # sampler 與 shuffle 互斥

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            pin_memory=True,
            drop_last=opt.phase == "train",
            num_workers=int(opt.num_workers),
        )


    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
