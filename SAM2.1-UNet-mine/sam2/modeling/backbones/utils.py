# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    将输入特征划分成不重叠的小窗口，如果尺寸不整除，则自动padding。
    Args:
        x (tensor): input tokens with [B, H, W, C].
        x (tensor): 输入特征张量，形状为 [B, H, W, C]
        window_size (int): window size.
        window_size (int): 要划分的窗口大小，比如 8、16等
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        windows (tensor): 所有窗口，形状为 [B * num_windows, window_size, window_size, C]
        (Hp, Wp): padded height and width before partition
        (Hp, Wp): padding 后的高度和宽度（用于 reverse 时还原）
    """
    # 获取输入张量的形状信息：批次B，高H，宽W，通道数C
    B, H, W, C = x.shape

    # 计算padding数量：确保H和W能被window_size整除
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # 如果需要padding（说明H或W不是window_size的整数倍）
    if pad_h > 0 or pad_w > 0:
        # 使用F.pad进行padding
        # 参数格式: (最后一个维度的pad_left, pad_right, 倒数第二个维度的pad_left, pad_right, ...)
        # 对W维（dim=2）pad_w个像素，对H维（dim=1）pad_h个像素，通道维不动
        # 素图像维度参数的倒序..
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    # 更新padding后的H和W（用于之后还原）
    Hp, Wp = H + pad_h, W + pad_w

    # 重新reshape，将特征图变为：
    # [B, Hp//W, window, Wp//W, window, C]
    # 相当于将图像分割成网格，每个小网格大小为 window_size × window_size
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    # 调整维度顺序，将小窗口移到一起，变成：[B, num_H, num_W, window, window, C]
    # 然后 reshape 成 [B * num_windows, window, window, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)

    # 返回窗口序列和padding后的高度宽度
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        windows：形状 [B * num_windows, window_size, window_size, C] 的窗口序列（即之前被划分成小块的特征图）
        window_size (int): window size.
        window_size：每个窗口的高和宽
        pad_hw (Tuple): padded height and width (Hp, Wp).
        pad_hw：padding 后的高和宽（Hp, Wp）
        hw (Tuple): original height and width (H, W) before padding.
        hw：原始的高和宽（H, W）
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    # 将 [B * num_windows, window_size, window_size, C] 重新变形为
    # [B, Hp//W, window, Wp//W, window, C]
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    # 将所有小窗口合并还原成一个大图 [B, Hp, Wp, C]
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    # 如果有 padding，多出来的区域就剪掉，只保留原始的大小 [B, H, W, C]
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
