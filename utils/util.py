import torch
import torch.nn as nn
import numpy as np
from torch.nn import MSELoss


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


def draw(eeg_data, channel_names):
    import numpy as np
    import matplotlib.pyplot as plt

    # 假设有22个通道，每个通道1000个时间点
    channels = 22
    time_points = 2000
    sampling_rate = 200  # 假设采样率为250 Hz
    time = np.arange(0, time_points) / sampling_rate

    # 创建波形图
    plt.figure(figsize=(10, 6))

    # 为了清晰，每个通道的信号沿Y轴叠加一个偏移量
    offset = 50  # 每个通道之间的间隔
    for i in range(channels):
        plt.plot(time, eeg_data[i] + i * offset, label=f"Channel {i + 1}")

    # 设置标签
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + Offset")
    plt.title("22-Channel EEG Waveform")
    plt.yticks([i * offset for i in range(channels)], [channel_name for channel_name in channel_names])
    # plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=8)  # 调整图例位置
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


class VLBLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss()

    def forward(self, recon_x, x, mu, logvar):
        recon_term = self.mse_loss(recon_x, x)
        kl_term = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)
        return recon_term + 0.00001 * kl_term