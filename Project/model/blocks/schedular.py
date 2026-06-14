"""
https://github.com/huggingface/transformers
"""

import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, SequentialLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _multi_phase_lr_lambda(
    current_step: int,
    *,
    warmup1_steps: int,
    warmup2_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
):
    # 第一階段 warm-up
    if current_step < warmup1_steps:
        return float(current_step) / float(max(1, warmup1_steps))
    # 第二階段再 warm-up
    elif current_step < warmup1_steps + warmup2_steps:
        return float(current_step - warmup1_steps) / float(max(1, warmup2_steps))
    # 第三階段 cosine annealing
    else:
        progress = float(current_step - warmup1_steps - warmup2_steps) / float(
            max(1, total_steps - warmup1_steps - warmup2_steps)
        )
        # 半週 cosine（0.5 cycles）或多週看 num_cycles 參數
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))


def get_two_warmup_cosine_scheduler(
    optimizer,
    warmup1_steps: int,
    warmup2_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    三段式 LR 調度：
      1) Step 0..warmup1_steps-1: linear 0→1
      2) Step warmup1_steps..warmup1_steps+warmup2_steps-1: linear 0→1
      3) 剩餘 Step: cosine annealing 1→0
    Args:
      optimizer: your optimizer
      warmup1_steps:   第一次 warmup 的 step 數 (e.g. 10)
      warmup2_steps:   第二次 warmup 的 step 數 (e.g. 10)
      total_steps:     全部訓練 step (e.g. epochs)
      num_cycles:      cosine cycles，0.5 表示 half-cosine
      last_epoch:      繼續訓練時接續的 epoch 編號
    """
    lr_lambda = partial(
        _multi_phase_lr_lambda,
        warmup1_steps=warmup1_steps,
        warmup2_steps=warmup2_steps,
        total_steps=total_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
