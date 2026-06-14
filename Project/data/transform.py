# transform.py
import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math


class Transforms(object):
    """
    幾何增強 + 噪聲擾動 + 隨機遮擋
    - 幾何：hflip / vflip / 90-180-270 旋轉 / RandomResizedCrop
    - 噪聲：高斯加性雜訊 (MSI, SAR)；乘性 speckle (特別適合 SAR)
    - 遮擋：Random Erase（矩形區塊，label 不改）
    會同時且一致地作用在 img1(MSI)、img2(SAR)、label（幾何變換），
    以維持多模態對位；噪聲與遮擋只作用在影像，不改 label。
    """

    def __init__(
        self,
        size: int = 128,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate: float = 0.5,
        p_crop: float = 0.5,
        p_noise: float = 0.5,
        p_erase: float = 0.5,
        # 高斯雜訊強度（加性）：影像值通常為 float32，這裡取相對保守的範圍
        gaussian_sigma_range=(0.01, 0.05),
        # speckle（乘性）：特別適合 SAR
        speckle_sigma_range=(0.02, 0.08),
        # 隨機遮擋參數：面積比例與長寬比
        erase_scale_range=(0.02, 0.20),
        erase_ratio_range=(0.3, 3.3),
        erase_fill_value=0.0,  # 用 0.0 填充（也可改為 per-channel 均值）
    ):
        self.size = size
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate = p_rotate
        self.p_crop = p_crop
        self.p_noise = p_noise
        self.p_erase = p_erase

        self.gaussian_sigma_range = gaussian_sigma_range
        self.speckle_sigma_range = speckle_sigma_range

        self.erase_scale_range = erase_scale_range
        self.erase_ratio_range = erase_ratio_range
        self.erase_fill_value = erase_fill_value

        self._angles = [90, 180, 270]

    def __call__(self, _data):
        img1, img2, label = _data["img1"], _data["img2"], _data["label"]

        # ---- 幾何：水平翻轉 ----
        if random.random() < self.p_hflip:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)

        # ---- 幾何：垂直翻轉 ----
        if random.random() < self.p_vflip:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)

        # ---- 幾何：離散旋轉（90/180/270）----
        if random.random() < self.p_rotate:
            angle = random.choice(self._angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)

        # ---- 幾何：隨機裁切 + 縮放 ----
        if random.random() < self.p_crop:
            i, j, h, w = transforms.RandomResizedCrop(size=(self.size, self.size)).get_params(
                img=img1, scale=[0.333, 1.0], ratio=[0.75, 1.333]
            )
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.size, self.size),
                interpolation=InterpolationMode.BILINEAR,
            )
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.size, self.size),
                interpolation=InterpolationMode.BILINEAR,
            )
            label = TF.resized_crop(
                label, i, j, h, w, size=(self.size, self.size),
                interpolation=InterpolationMode.NEAREST,  # 避免 label 混色
            )

        # ---- 噪聲擾動（加在影像，不改 label）----
        if random.random() < self.p_noise:
            # 1) 對 MSI 與 SAR 都加一點高斯雜訊（加性）
            img1 = self._add_gaussian_noise(img1, self.gaussian_sigma_range)
            img2 = self._add_gaussian_noise(img2, self.gaussian_sigma_range)
            # 2) 對 SAR 再額外加 speckle 雜訊（乘性），更貼近雷達影像特性
            img2 = self._add_speckle_noise(img2, self.speckle_sigma_range)

        # ---- Random Erase（遮擋，只作用在影像，不改 label）----
        if random.random() < self.p_erase:
            img1, img2 = self._random_erase_pair(img1, img2)

        return {"img1": img1, "img2": img2, "label": label}

    # ----------------- helpers -----------------

    @staticmethod
    def _rand_uniform(a, b):
        return a + (b - a) * random.random()

    def _add_gaussian_noise(self, img, sigma_range):
        sigma = self._rand_uniform(*sigma_range)
        noise = torch.randn_like(img) * sigma
        return img + noise

    def _add_speckle_noise(self, img, sigma_range):
        """乘性雜訊：img * (1 + N(0, sigma))，適合 SAR。"""
        sigma = self._rand_uniform(*sigma_range)
        noise = 1.0 + torch.randn_like(img) * sigma
        return img * noise

    def _random_erase_pair(self, img1, img2):
        """
        產生同一個矩形遮擋區，套用在 MSI 與 SAR 上（label 不改）。
        參考 torchvision 的參數設計，但用 functional.erase 以保證同步。
        """
        _, H, W = img1.shape
        area = H * W

        for _ in range(10):  # 嘗試多次尋找合理框
            target_area = area * self._rand_uniform(*self.erase_scale_range)

            # ❗ 改用 math.log / math.exp，避免把 float 丟進 torch.exp 造成型別錯誤
            log_ratio_min = math.log(self.erase_ratio_range[0])
            log_ratio_max = math.log(self.erase_ratio_range[1])
            log_aspect = self._rand_uniform(log_ratio_min, log_ratio_max)
            aspect = math.exp(log_aspect)

            h = int(round((target_area * aspect) ** 0.5))
            w = int(round((target_area / aspect) ** 0.5))

            if 0 < h <= H and 0 < w <= W:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                # 使用同一個框，對兩個模態一致遮擋
                img1 = TF.erase(img1, i, j, h, w, v=self.erase_fill_value, inplace=False)
                img2 = TF.erase(img2, i, j, h, w, v=self.erase_fill_value, inplace=False)
                return img1, img2

        # 如果找不到合適遮擋區，就不做遮擋
        return img1, img2
