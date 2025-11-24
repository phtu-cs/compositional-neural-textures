import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str) -> None:
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.BatchNorm2d(out_channels),
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv_res(x)
        x = self.net(x)
        return self.relu(x + res)


class TransposedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class TextonEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_size = 32

        self.conv = nn.Sequential(
            ResBlock(3, 64, padding_mode="reflect"),
            ResBlock(64, 128, padding_mode="reflect"),
            ResBlock(128, 256, padding_mode="reflect"),
            ResBlock(256, 512, padding_mode="reflect"),
            TransposedBlock(512, 256, padding_mode="reflect"),
            TransposedBlock(256, 128, padding_mode="reflect"),
        )

    def forward(self, img: torch.Tensor) -> dict:
        feature_map = self.conv(img)
        return feature_map


class DisentangledTextonEncoder(nn.Module):
    def __init__(self, blob_feature_dim) -> None:
        super().__init__()
        # self.n_parts = hyper_paras.n_parts
        self.output_size = 32

        self.conv = nn.Sequential(
            ResBlock(3, 64, padding_mode="reflect"),
            ResBlock(64, 128, padding_mode="reflect"),
            ResBlock(128, 256, padding_mode="reflect"),
            ResBlock(256, 512, padding_mode="reflect"),
        )

        self.app_output = nn.Sequential(
            TransposedBlock(512, 256, padding_mode="reflect"),
            TransposedBlock(256, blob_feature_dim - 2, padding_mode="reflect"),
        )

        self.rot_output = nn.Sequential(
            TransposedBlock(512, 256, padding_mode="reflect"),
            TransposedBlock(256, 2, padding_mode="reflect"),
        )

    def forward(self, img: torch.Tensor) -> dict:
        mid_feature_map = self.conv(img)

        app_feature_map = self.app_output(mid_feature_map)
        rot_feature_map = self.rot_output(mid_feature_map)

        feature_map = app_feature_map.clone()

        return feature_map, rot_feature_map
