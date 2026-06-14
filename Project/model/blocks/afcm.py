import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _dct_mat(n, device, dtype):
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)   # [n,1]
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)   # [1,n]
    M = torch.cos(math.pi * (2*i + 1) * k / (2*n))                 # [n,n]
    M[0, :] *= 1 / math.sqrt(n)
    M[1:, :] *= math.sqrt(2/n)
    return M

def dct2(x):
    B, C, H, W = x.shape
    DH = _dct_mat(H, x.device, x.dtype)
    DW = _dct_mat(W, x.device, x.dtype)
    x = torch.einsum('bchw,hh->bchw', x, DH)
    x = torch.einsum('bchw,ww->bchw', x, DW)
    return x

def idct2(X):
    B, C, H, W = X.shape
    DH = _dct_mat(H, X.device, X.dtype).t()
    DW = _dct_mat(W, X.device, X.dtype).t()
    x = torch.einsum('bchw,hh->bchw', X, DH)
    x = torch.einsum('bchw,ww->bchw', x, DW)
    return x

class SSEM(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, Ffreq):  # [B,C,H,W]
        B, C, H, W = Ffreq.shape
        # sparsity: 以 |x|>eps 的非零比例近似
        nz = (Ffreq.abs() > self.eps).float().sum(dim=(0,2,3))       # [C]
        mspa = nz / (nz.sum() + self.eps)

        # covariance diag: 對空間位置做變異數
        X = Ffreq.permute(0,2,3,1).reshape(-1, C)
        Xc = X - X.mean(dim=0, keepdim=True)
        var = (Xc.pow(2).sum(dim=0) / max(Xc.shape[0]-1, 1))        # [C]
        mcca = torch.softmax(var, dim=0)

        mssa = self.alpha * mspa + (1 - self.alpha) * mcca
        w = mssa.view(1, C, 1, 1)
        return Ffreq * w

class FrequencyAttention(nn.Module):
    def __init__(self, gamma=0.5, spatial_downsample=4):
        super().__init__()
        self.gamma = gamma
        self.spatial_downsample = spatial_downsample

    def forward(self, Ff):  # [B,C,H,W]
        B, C, H, W = Ff.shape
        if self.spatial_downsample > 1:
            Fd = F.avg_pool2d(Ff, self.spatial_downsample, self.spatial_downsample)
        else:
            Fd = Ff
        _, _, h, w = Fd.shape
        N = h*w

        # Energy spectrum attention
        E = (Fd.abs() ** 2)
        Esum = E.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-12)
        Fes = E / Esum                      # [B,C,h,w]
        Q = Fes.view(B, C, N).transpose(1, 2)  # [B,N,C]
        K = Fes.view(B, C, N)                  # [B,C,N]
        A_es = torch.softmax(torch.bmm(Q, K), dim=-1)  # [B,N,N]

        # Positional self-attention（輕量版）
        Qp = Fd.view(B, C, N).transpose(1, 2)          # [B,N,C]
        Kp = Fd.view(B, C, N)                          # [B,C,N]
        A_fp = torch.softmax(torch.bmm(Qp, Kp) / math.sqrt(C), dim=-1)

        A = self.gamma * A_es + (1 - self.gamma) * A_fp
        V = Fd.view(B, C, N).transpose(1, 2)           # [B,N,C]
        out = torch.bmm(A, V).transpose(1, 2).view(B, C, h, w)
        out = out + Fd

        if self.spatial_downsample > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

class LearnableGaussianConv(nn.Module):
    def __init__(self, channels, kernel_size=7, min_sigma=0.1, max_sigma=5.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.log_sigma = nn.Parameter(torch.zeros(channels))

    def forward(self, x):  # [B,C,H,W]
        B, C, H, W = x.shape
        k = self.kernel_size
        r = k // 2
        sigmas = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(self.log_sigma)
        xs = torch.arange(-r, r+1, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(xs, xs, indexing='ij')
        kernels = []
        for c in range(C):
            s = sigmas[c].clamp_min(1e-6)
            g = torch.exp(-(xx**2 + yy**2) / (2 * s**2))
            g = g / (g.sum() + 1e-12)
            kernels.append(g)
        weight = torch.stack(kernels, dim=0).unsqueeze(1)  # [C,1,k,k]
        return F.conv2d(x, weight, stride=1, padding=r, groups=C)

class FDAM(nn.Module):
    def __init__(self, channels, kernel_size=7, gamma=0.5, spatial_downsample=4):
        super().__init__()
        self.low_sep = LearnableGaussianConv(channels, kernel_size=kernel_size)
        self.fa_high = FrequencyAttention(gamma=gamma, spatial_downsample=spatial_downsample)
        self.fa_low  = FrequencyAttention(gamma=gamma, spatial_downsample=spatial_downsample)

    def forward(self, Fin):  # [B,C,H,W] (frequency domain)
        Flow  = self.low_sep(Fin)
        Fhigh = Fin - Flow
        FlowS = self.low_sep(Flow)
        return self.fa_high(Fhigh) + self.fa_low(FlowS)

class AFCM(nn.Module):
    def __init__(self, channels, kernel_size=7, alpha=0.5, gamma=0.5, spatial_downsample=4):
        super().__init__()
        self.ssem = SSEM(alpha=alpha)
        self.fdam = FDAM(channels, kernel_size=kernel_size, gamma=gamma, spatial_downsample=spatial_downsample)

    def forward(self, x):     # [B,C,H,W] (spatial feature)
        Xf = dct2(x)
        Xf = self.ssem(Xf)
        Xf = self.fdam(Xf)
        y  = idct2(Xf)
        return x + y   