import torch.nn.functional as F
from torch import nn
from torch.nn import GroupNorm


class SPADE(nn.Module):
    def __init__(self, input_channel, n_embeddings):
        super().__init__()
        self.norm = GroupNorm(num_groups=1, num_channels=input_channel, affine=False)
        self.conv = nn.Conv2d(n_embeddings, 128, kernel_size=3, padding=1)
        self.conv_gamma = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, input_channel, kernel_size=3, padding=1)

    def forward(self, x, heatmaps):
        normalized_x = self.norm(x)
        heatmaps_features = F.leaky_relu(self.conv(heatmaps), 0.2)
        heatmaps_gamma = self.conv_gamma(heatmaps_features)
        heatmaps_beta = self.conv_beta(heatmaps_features)
        return (1 + heatmaps_gamma) * normalized_x + heatmaps_beta


class SPADEResBlk(nn.Module):
    def __init__(self, in_channel, out_channel, n_embeddings):
        super().__init__()
        mid_channel = min(in_channel, out_channel)
        self.learn_shortcut = in_channel != out_channel
        self.spade1 = SPADE(in_channel, n_embeddings)
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        self.spade2 = SPADE(mid_channel, n_embeddings)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)

        if self.learn_shortcut:
            self.spade_shortcut = SPADE(in_channel, n_embeddings)
            self.conv_shortcut = nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=1
            )

    def forward(self, x, heatmaps):
        shortcut = x
        x = self.conv1(F.leaky_relu(self.spade1(x, heatmaps), 0.2))
        x = self.conv2(F.leaky_relu(self.spade2(x, heatmaps), 0.2))

        if self.learn_shortcut:
            shortcut = self.conv_shortcut(self.spade_shortcut(shortcut, heatmaps))

        return x + shortcut
