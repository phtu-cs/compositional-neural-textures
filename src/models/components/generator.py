import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.base_model import BaseModel
from src.models.components.spade import SPADEResBlk


def pyramid_resize(img, cutoff):
    out = [img]
    while img.shape[-1] > cutoff:
        img = F.interpolate(
            img, img.shape[-1] // 2, mode="bilinear", align_corners=False
        )
        out.append(img)
    return {i.size(-1): i for i in out}


class FullyConvolutionalSPADEGeneratorModule(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.n_embedding = hparams.n_embedding
        self.feature_map_size_ratios = hparams.feature_map_size_ratios

        self.spade_blocks = nn.ModuleList(
            [
                SPADEResBlk(self.n_embedding, self.n_embedding, self.n_embedding),  # 32
                SPADEResBlk(
                    self.n_embedding, self.n_embedding // 2, self.n_embedding
                ),  # 64
                SPADEResBlk(
                    self.n_embedding // 2, self.n_embedding // 4, self.n_embedding
                ),  # 128
                SPADEResBlk(self.n_embedding // 4, 3, self.n_embedding),  # 128
            ]
        )

    def forward_func(self, layout):
        score_img = layout["layer_level_layouts"][-1][0]["score_img"]
        img_size = score_img.shape[-1]
        assert score_img.shape[-1] == score_img.shape[-2]

        scores_pyramid = pyramid_resize(
            layout["layer_level_layouts"][-1][0]["score_img"],
            self.feature_map_size_ratios[0] * img_size,
        )
        blob_features = layout["layer_level_layouts"][-1][0]["blob_features"]

        feature_map_sizes = [
            round(img_size * ratio) for ratio in self.feature_map_size_ratios
        ]

        style_maps = {}
        for map_size in set(feature_map_sizes):
            scores = scores_pyramid[map_size]
            style_map = torch.einsum("bnc,bnhw->bchw", blob_features, scores)
            style_maps[map_size] = style_map

        # feature_grid = layout['feature_grid']
        # const_in = #self.constant_input(blob_features)
        feature_in = (
            scores_pyramid[feature_map_sizes[0]][:, 1:]
            .amax(dim=1)[:, None]
            .repeat(1, self.n_embedding, 1, 1)
        )
        x = feature_in.clone()
        for i in range(len(feature_map_sizes)):
            scores = scores_pyramid[feature_map_sizes[i]]
            style_map = style_maps[feature_map_sizes[i]]
            x = self.spade_blocks[i](x, style_map)
            if i == len(feature_map_sizes) - 1:
                break
            elif feature_map_sizes[i] != feature_map_sizes[i + 1]:
                x = F.interpolate(
                    x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False
                )

        self.output["recon"] = {-1: torch.nn.functional.tanh(x.unsqueeze(1))}
        self.visual_vars = {
            "feature_in": feature_in,
            "style_map": style_maps[feature_map_sizes[-1]],
        }

    def update_visuals(self):
        if isinstance(self.output["recon"], torch.Tensor):
            h = self.output["recon"].shape[-1]
            vis_feature_grid = torch.einsum(
                "bnc,bnhw->bchw",
                self.visual_vars["latent"][-1]["spatial_style"],
                self.visual_vars["latent"][-1]["scores_pyramid"][h],
            )
            self.visuals = {
                "recon": self.output["recon"],
                "feature_img": vis_feature_grid[:, :3],
            }
        elif isinstance(self.output["recon"], dict):
            for layer_id, recon in self.output["recon"].items():
                for level in range(recon.shape[1]):
                    self.visuals.update(
                        {
                            "recon_layer" + str(layer_id) + "level" + str(level): recon[
                                :, level
                            ]
                        }
                    )
        else:
            assert False

        self.visuals["feature_in"] = self.visual_vars["feature_in"][:, :3, :, :]
        self.visuals["style_map"] = self.visual_vars["style_map"][:, :3, :, :]
