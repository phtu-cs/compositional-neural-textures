import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.base_model import BaseModel


class splat_blob_layer_tree_module(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aux_layers = hparams.aux_layers
        self.tree_splat = hparams.splat_func

    def forward_func(self, blob):
        self.visuals["img"] = blob["img"]
        bs = blob["img"].shape[0]
        self.blob = blob

        layer_level_layouts = {}
        layer_level_feature_grids = {}
        for layer_id in self.aux_layers:
            if (
                blob["global_features"] is None
                and blob["layer_global_features"] is not None
            ):
                assert False
            elif (
                blob["global_features"] is not None
                and blob["layer_global_features"] is None
            ):
                assert False
            elif (
                blob["global_features"] is None
                and blob["layer_global_features"] is None
            ):
                assert "per_texel_feature" in blob["layer_tree_levels"][layer_id][0]
                blob_tmp = {
                    "tree_levels": blob["layer_tree_levels"][layer_id],
                    "img": blob["img"],
                    "global_features": None,
                    "bg_features": blob["bg_features"],
                }
                if "segments" in blob:
                    blob_tmp["segments"] = blob["segments"]
            else:
                assert False

            layout, layout_visuals = self.tree_splat(blob_tmp)
            layer_level_layouts[layer_id] = layout["level_layouts"]
            layer_level_feature_grids[layer_id] = layout["level_feature_grids"]

        self.output["layer_level_feature_grids"] = layer_level_feature_grids
        self.output["layer_level_layouts"] = layer_level_layouts


def pyramid_resize(img, cutoff):
    out = [img]
    while img.shape[-1] > cutoff:
        img = F.interpolate(
            img, img.shape[-1] // 2, mode="bilinear", align_corners=False
        )
        out.append(img)
    return {i.size(-1): i for i in out}


class splat_blob_tree_module_occluded_flexible(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.use_covs_feature = hparams.use_covs_feature
        self.cutoff = hparams.cutoff
        self.img_size = hparams.img_size

        self.down_channel = nn.Conv2d(
            in_channels=hparams.num_levels * 128,
            out_channels=128,
            kernel_size=1,
            stride=1,
        )

        if self.use_covs_feature:
            self.linear_covs_feature = nn.Linear(4, hparams.hidden_dim)

    def splat_func(self, nodes, global_features, bg_features, name="leaves"):
        if "img_size" in nodes:
            used_img_size = nodes["img_size"]
        else:
            used_img_size = self.img_size

        xs = nodes["xy"][:, :, 0]
        ys = nodes["xy"][:, :, 1]

        covs = nodes["covs"]
        device = xs.device

        feature_coords = (torch.stack((xs, ys), -1) + 3).div(6).mul(used_img_size)
        grid_coords = torch.stack(
            (
                torch.arange(used_img_size, device=device).repeat(used_img_size),
                torch.arange(used_img_size, device=device).repeat_interleave(
                    used_img_size
                ),
            )
        )
        delta = (grid_coords[None, None] - feature_coords[..., None]).div(
            used_img_size
        ) * 3

        sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
        sq_mahalanobis = einops.rearrange(
            sq_mahalanobis, "n m (s1 s2) -> n s1 s2 m", s1=used_img_size
        )
        mahalanobis = sq_mahalanobis.sqrt()
        blob_influence_mask = mahalanobis > self.cutoff
        sq_mahalanobis[blob_influence_mask] = 1e10

        tmp = -1 * sq_mahalanobis  # -0.5
        gaussian_weights = nodes["weights"][:, None, None, :, 0] * tmp.exp()
        gaussian_weight_map = gaussian_weights.sum(dim=-1)

        bg_weights = torch.ones_like(gaussian_weights[..., :1])
        comb_weights = torch.cat((bg_weights, gaussian_weights), dim=-1)

        rev = list(range(comb_weights.size(-1) - 1, -1, -1))
        alpha_cumprod = (1 - comb_weights[..., rev]).cumprod(-1)[..., rev].roll(
            -1, -1
        ) * comb_weights
        alpha_cumprod[..., -1] = comb_weights[..., -1]
        gaussian_contri = alpha_cumprod.unsqueeze(-1)

        score_img = einops.rearrange(alpha_cumprod, "n h w m -> n m h w")

        if "per_texel_feature" in nodes and nodes["per_texel_feature"] is not None:
            if global_features != None:
                assert False
            else:
                if self.use_covs_feature:
                    assert False
                else:
                    blob_features = nodes["per_texel_feature"]
        else:
            assert False

        if "texton_encoder" in self.hparams:
            if self.hparams.texton_encoder == "disentangled_local_texton":
                blob_features = torch.cat((blob_features, nodes["rot_feature"]), dim=-1)
        blob_features = torch.cat((bg_features.unsqueeze(1), blob_features), dim=1)
        normalized_feature_grid = (
            torch.einsum("bnc,bhwnd->bhwcd", blob_features, gaussian_contri)
            .squeeze(-1)
            .permute(0, 3, 1, 2)
        )
        random_colors = torch.rand(
            1,
            blob_features.shape[1],
            3,
            device=blob_features.device,
            generator=torch.Generator(device=blob_features.device).manual_seed(42),
        ).repeat(xs.shape[0], 1, 1)
        vis_gaussian_contri = (
            torch.einsum("bnc,bhwnd->bhwcd", random_colors, gaussian_contri)
            .squeeze(-1)
            .permute(0, 3, 1, 2)
        )

        layout = {
            "xs": xs,
            "ys": ys,
            "covs": covs,
            "weights": nodes["weights"],
            "gaussian_weights": gaussian_weights,
            "mahalanobis": mahalanobis,
            "blob_weights": nodes["weights"][:, None, None, :, 0],
            "gaussian_contri": gaussian_contri,
            "feature_grid": normalized_feature_grid,
            "vis_gaussian_contri": vis_gaussian_contri,
            "blob_features": blob_features,
            "score_img": score_img,
        }

        output = {name: layout}

        return output

    def forward_func(self, blob):
        self.visuals["img"] = blob["img"]
        self.blob = blob

        level_feature_grids = []
        level_layouts = []
        for l in range(len(blob["tree_levels"])):
            name = "level" + str(l)
            if isinstance(blob["global_features"], list):
                assert False
            elif isinstance(blob["global_features"], torch.Tensor):
                assert False
            elif blob["global_features"] is None:
                output_level = self.splat_func(
                    blob["tree_levels"][l],
                    global_features=None,
                    bg_features=blob["bg_features"],
                    name=name,
                )
            else:
                assert False

            level_layouts.append(output_level[name])
            level_feature_grids.append(output_level[name]["feature_grid"])

        if self.hparams.merged_feature_grid:
            assert False
        else:
            self.output["feature_grid"] = level_feature_grids[-1]
        assert len(level_layouts) == len(level_feature_grids)

        self.output["level_feature_grids"] = level_feature_grids
        self.output["level_layouts"] = level_layouts
