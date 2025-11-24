# Modified from MaskFormer code

import math
from typing import Tuple

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup
from detectron2.modeling import build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork

from src.models.components.base_model import BaseModel
from src.models.components.encoders import (
    DisentangledTextonEncoder,
    TextonEncoder,
)
from src.models.components.Mask2Former.mask2former import add_maskformer2_config
from src.models.components.Mask2Former.mask2former.modeling.criterion import (
    SetCriterion,
)
from src.models.components.Mask2Former.mask2former.modeling.matcher import (
    HungarianMatcher,
)


class MaskFormer(BaseModel):
    # @configurable
    def initialize(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    def setup(self, args):
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    def __init__(self, hparams, latent2blob):
        super().__init__(hparams)
        cfg = self.setup(hparams)

        cfg.defrost()
        assert hparams.num_levels == 1
        if self.hparams.use_bg:
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = hparams.num_queries + 1
        else:
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = hparams.num_queries
        cfg.freeze()

        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        self.initialize(
            **{
                "backbone": backbone,
                "sem_seg_head": sem_seg_head,
                "criterion": criterion,
                "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
                "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
                "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                ),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
                "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
                "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
                "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            }
        )

        self.hparams = hparams

        if hparams.use_point_sampled_features:
            self.linear_feature = nn.Linear(256, self.hparams.blob_feature_dim)
        else:
            self.id_embed = nn.Embedding(
                cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.hparams.blob_feature_dim
            )

        if self.hparams.encoder == "local_texton":
            self.texton_encoder = TextonEncoder()
            self.conv_down_feature = nn.Conv2d(
                128, self.hparams.blob_feature_dim, kernel_size=1
            )
        elif self.hparams.encoder == "disentangled_local_texton":
            self.texton_encoder = DisentangledTextonEncoder(
                self.hparams.blob_feature_dim
            )
        elif self.hparams.encoder == "seg_feature":
            pass
        elif self.hparams.encoder == "shared_encoder":
            self.local_image_encoder = FeaturePyramidNetwork(
                [256, 512, 1024, 2048], self.hparams.blob_feature_dim
            )

        if hparams.original_res_mask:
            coord_y, coord_x = torch.meshgrid(
                torch.linspace(-3, 3, hparams.img_size),
                torch.linspace(-3, 3, hparams.img_size),
            )
        else:
            coord_y, coord_x = torch.meshgrid(
                torch.linspace(-3, 3, hparams.img_size // 4),
                torch.linspace(-3, 3, hparams.img_size // 4),
            )
            assert False

        coords = torch.stack((coord_x, coord_y), 0)
        self.register_buffer("coords", coords)
        self.cfg = cfg

        if self.hparams.use_bg:
            assert self.hparams.bg_feature_dim != -1
            self.mlp_bg_feature = nn.Sequential(
                nn.Linear(self.hparams.blob_feature_dim, self.hparams.bg_feature_dim),
                nn.ReLU(),
                nn.Linear(self.hparams.bg_feature_dim, self.hparams.bg_feature_dim),
                nn.ReLU(),
                nn.Linear(self.hparams.bg_feature_dim, self.hparams.blob_feature_dim),
            )

        else:
            pass

        if self.hparams.use_shifted_blob_sp_params:
            self.mlp_shift_head = nn.Sequential(
                nn.Linear(256, self.hparams.blob_feature_dim),
                nn.ReLU(),
                nn.Linear(self.hparams.blob_feature_dim, self.hparams.blob_feature_dim),
            )

            self.mlp_shift_output = nn.Sequential(
                nn.Linear(
                    2 * self.hparams.blob_feature_dim, self.hparams.blob_feature_dim
                ),
                nn.ReLU(),
                nn.Linear(self.hparams.blob_feature_dim, 2),
                nn.Tanh(),
            )

            self.mlp_shift_weight = nn.Sequential(
                nn.Linear(
                    2 * self.hparams.blob_feature_dim, self.hparams.blob_feature_dim
                ),
                nn.ReLU(),
                nn.Linear(self.hparams.blob_feature_dim, 1),
                nn.Sigmoid(),
            )

            self.mlp_covs_shift = nn.Sequential(
                nn.Linear(2 * hparams.blob_feature_dim, hparams.blob_feature_dim),
                nn.LeakyReLU(),
                nn.Linear(hparams.blob_feature_dim, 3),
            )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def exp_tau(self, step):
        # exp base
        base = (self.hparams.min_tau / self.hparams.max_tau) ** (
            1 / self.hparams.max_step_tau
        )
        if step < self.hparams.max_step_tau:
            tau = self.hparams.max_tau * (base**step)
        else:
            tau = self.hparams.min_tau

        self.output.update({"tau": tau})
        return tau

    def compute_img_mask(self, img):
        mask = (~((img == 0).all(dim=1, keepdim=True))).float()
        return mask

    def generate_texton_feature_map(self, images, features, segment_feature_map):
        if self.hparams.encoder == "local_texton":
            texton_feature_map = self.texton_encoder(images)
            texton_feature_map = self.conv_down_feature(texton_feature_map)
            rot_feature_map = None
        elif self.hparams.encoder == "disentangled_local_texton":
            texton_feature_map, rot_feature_map = self.texton_encoder(images)
        elif self.hparams.encoder == "shared_encoder":
            assert False
        elif self.hparams.encoder == "seg_feature":
            assert False
        else:
            assert False

        if self.hparams.original_res_mask:
            texton_feature_map = F.interpolate(
                texton_feature_map,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if rot_feature_map != None:
                rot_feature_map = F.interpolate(
                    rot_feature_map,
                    size=(images.shape[-2], images.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            assert False

        if self.hparams.encoder == "disentangled_local_texton":
            return texton_feature_map, rot_feature_map
        else:
            return texton_feature_map

    def generate_background_feature(self, input_dict):
        B = input_dict["images"].shape[0]
        if self.hparams.use_bg:
            bg_mask = input_dict["mask_exist_prob"][:, -1]
            bg_features_tmp = (
                input_dict["texton_feature_map"] * bg_mask.unsqueeze(1)
            ).sum(dim=(-1, -2)) / (bg_mask.unsqueeze(1).sum(dim=(-1, -2)) + 1e-16)
            bg_features = self.mlp_bg_feature(bg_features_tmp)
        else:
            if self.hparams.bg_feature_method == "zero":
                bg_features = torch.zeros(
                    B, self.hparams.blob_feature_dim, device=self.device
                )
            elif self.hparams.bg_feature_method == "mean":
                bg_features = input_dict["texton_feature_map"].mean(dim=(-1, -2))
            else:
                assert False

        return bg_features

    def step_bd_thred(self):
        max_bd_thred = self.hparams.bd_thred.max
        delayed_step = self.hparams.bd_thred.delayed_step
        max_step = self.hparams.bd_thred.max_step

        if self.global_step < delayed_step:
            curr_bd_thred = 0
        elif self.global_step >= delayed_step and self.global_step < max_step:
            curr_bd_thred = (
                (self.global_step - delayed_step)
                / (max_step - delayed_step)
                * max_bd_thred
            )
        else:
            curr_bd_thred = max_bd_thred

        self.visual_vars["curr_bd_thred"] = curr_bd_thred
        return curr_bd_thred

    def is_boundary(self, blob_xy):
        curr_bd_thred = self.step_bd_thred()

        is_boundary = torch.logical_or(
            (blob_xy < -3 + curr_bd_thred).any(dim=-1),
            (blob_xy > 3 - curr_bd_thred).any(dim=-1),
        )

        return is_boundary.float()

    def generate_blob_sp_feat_params(
        self,
        final_mask_exist_prob,
        texton_feature_map,
        shift_features,
        rot_feature_map=None,
    ):
        B, num_blobs = final_mask_exist_prob.shape[:2]
        H, W = final_mask_exist_prob.shape[-2:]
        assert H == W
        if self.coords.shape[-2:] == final_mask_exist_prob.shape[-2:]:
            coords = self.coords[None].repeat(B, 1, 1, 1)
        else:
            if not hasattr(self, "new_coords"):
                if self.hparams.original_res_mask:
                    coord_y, coord_x = torch.meshgrid(
                        torch.linspace(-3, 3, H), torch.linspace(-3, 3, H)
                    )
                else:
                    coord_y, coord_x = torch.meshgrid(
                        torch.linspace(-3, 3, H // 4), torch.linspace(-3, 3, H // 4)
                    )
                    assert False
                self.new_coords = torch.stack((coord_x, coord_y), 0).to(self.device)

            if self.training:
                coords = self.new_coords[None].repeat(B, 1, 1, 1)
            else:
                coords = self.new_coords[None].repeat(B, 1, 1, 1).to(self.device)

        device = final_mask_exist_prob.device

        blob_xy = torch.einsum("bchw,bnhw->bcn", coords, final_mask_exist_prob) / (
            final_mask_exist_prob.sum(dim=(-1, -2))[:, None] + 1e-16
        )
        xy_diff = coords[:, :, None] - blob_xy[..., None, None]
        blob_covs = (
            torch.einsum(
                "bijnhw,bjknhw->biknhw", xy_diff.unsqueeze(2), xy_diff.unsqueeze(1)
            )
            * final_mask_exist_prob[:, None, None]
        ).sum(dim=(-1, -2)) / (
            final_mask_exist_prob[:, None, None].sum(dim=(-1, -2)).clamp_min(1e-6)
        )

        blob_covs = blob_covs + 1e-7 * torch.eye(2, device=device)[
            None, :, :, None
        ].repeat(B, 1, 1, num_blobs)

        blob_xy = blob_xy.permute(0, 2, 1)
        blob_covs = blob_covs.permute(0, -1, 1, 2)

        if self.hparams.shrink_covs:
            L = torch.linalg.cholesky(blob_covs)
            covs_scale = math.sqrt(0.5) * torch.eye(2, device=device)
            scaled_L = L @ covs_scale[None, None, ...]
            blob_covs = scaled_L @ scaled_L.transpose(-1, -2)

        try:
            assert blob_xy.min() >= -3 and blob_xy.max() <= 3
        except:
            print("min:", blob_xy.min(), "max:", blob_xy.max())

        if self.hparams.use_point_sampled_features:
            assert False
        else:
            if texton_feature_map.shape[-2:] != final_mask_exist_prob.shape[-2:]:
                used_final_mask_exist_prob = F.interpolate(
                    final_mask_exist_prob,
                    size=texton_feature_map.shape[-2:],
                    mode="bilinear",
                )
            else:
                used_final_mask_exist_prob = final_mask_exist_prob

            blob_features_tmp = torch.einsum(
                "bchw,bnhw->bcn", texton_feature_map, used_final_mask_exist_prob
            ) / used_final_mask_exist_prob.sum(dim=(-1, -2)).clamp_min(1e-6).unsqueeze(
                1
            )
            blob_features = blob_features_tmp.permute(0, 2, 1)

            if rot_feature_map != None:
                rot_features_tmp = torch.einsum(
                    "bchw,bnhw->bcn", rot_feature_map, used_final_mask_exist_prob
                ) / used_final_mask_exist_prob.sum(dim=(-1, -2)).clamp_min(
                    1e-6
                ).unsqueeze(1)
                rot_features = rot_features_tmp.permute(0, 2, 1)
            else:
                rot_features = None

        blob_features = F.normalize(blob_features, p=2.0, dim=-1)

        if self.hparams.encoder == "local_texton":
            assert rot_features == None
            rot_features = None
        elif self.hparams.encoder == "disentangled_local_texton":
            rot_features = F.normalize(rot_features, p=2.0, dim=-1)
        else:
            assert False

        if self.hparams.use_shifted_blob_sp_params:
            shifted_blob_xy, shifted_blob_covs, shifts = self.shift_blob_sp_feat_params(
                blob_xy,
                blob_covs,
                blob_features,
                rot_features,
                shift_features,
                final_mask_exist_prob,
            )
            return (
                blob_xy,
                blob_covs,
                blob_features,
                shifted_blob_xy,
                shifted_blob_covs,
                shifts,
                coords,
                rot_features,
            )
        else:
            return (
                blob_xy,
                blob_covs,
                blob_features,
                None,
                None,
                None,
                coords,
                rot_features,
            )

    def covs_shift_no_bd_thred(self, blob_covs, blob_covs_raw):
        covs_sigma = torch.pow(
            self.hparams.max_shift_covs_scale, blob_covs_raw[..., :2].tanh()
        )
        raw_angle = torch.pi * blob_covs_raw[..., 2].tanh()
        covs_q = torch.stack(
            (
                torch.stack((torch.cos(raw_angle), torch.sin(raw_angle)), dim=-1),
                torch.stack((-torch.sin(raw_angle), torch.cos(raw_angle)), dim=-1),
            ),
            dim=-1,
        )
        blob_covs_scale_sq, blob_covs_q = torch.linalg.eigh(blob_covs)

        discard = (blob_covs_scale_sq[..., 0] - blob_covs_scale_sq[..., 1]).abs() < 1e-6
        if discard.any() != False:
            _, discard_blob_covs_q = torch.linalg.eigh(blob_covs[discard])
            _, valid_blob_covs_q = torch.linalg.eigh(blob_covs[~discard])

            temp_covs_s = torch.diag_embed(blob_covs_scale_sq.clamp(1e-8).sqrt())

            final_corr_covs_q = torch.zeros_like(blob_covs_q)
            final_corr_covs_q[discard] = torch.eye(2, device=self.device)[None].repeat(
                discard_blob_covs_q.shape[0], 1, 1
            )
            final_corr_covs_q[~discard] = valid_blob_covs_q

            m_covs_q = covs_q @ final_corr_covs_q
            m_covs_s = temp_covs_s @ torch.diag_embed(covs_sigma)
            m_covs_lambda = m_covs_s.pow(2)
            print("Discard!")

        else:
            temp_covs_s = torch.diag_embed(blob_covs_scale_sq.clamp(1e-8).sqrt())
            m_covs_q = covs_q @ blob_covs_q
            m_covs_s = temp_covs_s @ torch.diag_embed(covs_sigma)
            m_covs_lambda = m_covs_s.pow(2)

        shifted_blob_covs = m_covs_q @ m_covs_lambda @ m_covs_q.transpose(-1, -2)
        return shifted_blob_covs, {"scale_shift": covs_sigma, "rot_shift": raw_angle}

    def covs_shift_bd_thred(self, blob_covs, blob_covs_raw, is_boundary):
        blob_covs_scale_sq, blob_covs_q = torch.linalg.eigh(blob_covs)

        covs_sigma = blob_covs_scale_sq.clamp(1e-8).sqrt() * torch.pow(
            self.hparams.max_shift_covs_scale, blob_covs_raw[..., :2].tanh()
        )

        raw_angle = torch.pi * blob_covs_raw[..., 2].tanh()
        covs_lambda = torch.diag_embed(covs_sigma.pow(2))
        covs_q = torch.stack(
            (
                torch.stack((torch.cos(raw_angle), torch.sin(raw_angle)), dim=-1),
                torch.stack((-torch.sin(raw_angle), torch.cos(raw_angle)), dim=-1),
            ),
            dim=-1,
        )

        direct_blob_covs = covs_q @ covs_lambda @ covs_q.transpose(-1, -2)
        if torch.rand(1) < 0.5:
            shifted_blob_covs = (1 - is_boundary)[
                ..., None, None
            ] * blob_covs + is_boundary[..., None, None] * direct_blob_covs
        else:
            shifted_blob_covs = blob_covs.clone()

        return shifted_blob_covs, direct_blob_covs

    def shift_blob_sp_feat_params(
        self,
        blob_xy,
        blob_covs,
        blob_features,
        rot_features,
        shift_features,
        final_mask_exist_prob,
    ):
        if rot_features == None:
            blob_shift_features = torch.cat((blob_features, shift_features), dim=-1)
        else:
            blob_shift_features = torch.cat(
                (blob_features, rot_features, shift_features), dim=-1
            )

        blob_xy_shift = self.hparams.max_blob_shift * self.mlp_shift_output(
            blob_shift_features
        )

        blob_covs_raw = self.mlp_covs_shift(blob_shift_features)

        shifts = {}
        if self.hparams.use_bd_thred:
            is_boundary = self.is_boundary(blob_xy)

            if self.hparams.use_xy_shift:
                shifted_blob_xy = blob_xy + is_boundary[..., None] * blob_xy_shift
            else:
                shifted_blob_xy = blob_xy.clone()

            if self.hparams.use_covs_shift:
                shifted_blob_covs, direct_blob_covs = self.covs_shift_bd_thred(
                    blob_covs, blob_covs_raw, is_boundary
                )
                shifts.update(
                    {"direct_blob_covs": direct_blob_covs, "is_boundary": is_boundary}
                )
            else:
                shifted_blob_covs = blob_covs.clone()

            shifts.update({"xy_shift": blob_xy_shift})
            self.visual_vars["is_boundary"] = is_boundary
        else:
            shifted_blob_xy = blob_xy + blob_xy_shift
            shifted_blob_covs, covs_shifts = self.covs_shift_no_bd_thred(
                blob_covs, blob_covs_raw
            )
            shifts.update({"xy_shift": blob_xy_shift})
            shifts.update(covs_shifts)

        return shifted_blob_xy, shifted_blob_covs, shifts

    def generate_segment_and_blob(self, input_dict):
        images = input_dict["images"]
        B = images.shape[0]

        seg_outputs = input_dict["seg_outputs"]
        global_step = input_dict["global_step"]
        rot_feature_map = input_dict["rot_feature_map"]
        texton_feature_map = input_dict["texton_feature_map"]

        pred_logits = seg_outputs["pred_logits"]
        assert pred_logits.shape[-1] == 2
        mask_logits = seg_outputs["pred_masks"]

        if self.hparams.original_res_mask:
            mask_logits = F.interpolate(
                mask_logits,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            assert False

        tau = self.exp_tau(global_step)
        if self.training:
            if self.hparams.compute_blob_weight == "pred_logit_modulation":
                final_mask_logits = mask_logits - pred_logits[:, :, 0][..., None, None]
                mask_exist_prob = F.softmax(final_mask_logits, dim=1)

                blob_exist_prob = mask_exist_prob.pow(2).sum(dim=(-1, -2)) / (
                    mask_exist_prob.sum(dim=(-1, -2)).clamp_min(1e-6)
                )
                blob_weight = blob_exist_prob.clone()
                blob_exist_prob = blob_exist_prob.unsqueeze(-1)
                blob_weight = blob_weight.unsqueeze(-1)

                self.visual_vars["mod_pred_logits"] = pred_logits[:, :, 0]

            elif self.hparams.compute_blob_weight == "direct_blob_weight":
                final_mask_logits = mask_logits

                mask_exist_prob = F.softmax(final_mask_logits, dim=1)

                blob_weight = F.gumbel_softmax(pred_logits, dim=-1, tau=tau)[..., 0:1]
                blob_exist_prob = F.softmax(pred_logits, dim=-1)[..., 0:1]

            elif self.hparams.compute_blob_weight == "pred_logit_modulation_gumbel":
                final_mask_logits = mask_logits - pred_logits[:, :, 0][..., None, None]
                mask_exist_prob = F.softmax(final_mask_logits, dim=1)
                blob_exist_prob = mask_exist_prob.pow(2).sum(dim=(-1, -2)) / (
                    mask_exist_prob.sum(dim=(-1, -2)).clamp_min(1e-6)
                )
                blob_exist_prob = blob_exist_prob.unsqueeze(-1)
                clamped_blob_exist_prob = blob_exist_prob.clamp(1e-6, 1 - 1e-6)
                logits_tmp = torch.cat(
                    (
                        clamped_blob_exist_prob.log(),
                        (1 - clamped_blob_exist_prob).log(),
                    ),
                    dim=-1,
                )
                blob_weight = F.gumbel_softmax(logits_tmp, dim=-1, tau=tau)[..., 0:1]
                self.visual_vars["mod_pred_logits"] = pred_logits[:, :, 0]
            elif (
                self.hparams.compute_blob_weight == "pred_logit_modulation_gumbel_area"
            ):
                assert False
            else:
                assert False

        else:
            # inference
            final_mask_logits = mask_logits - pred_logits[:, :, 0][..., None, None]
            mask_exist_prob = F.softmax(final_mask_logits, dim=1)

            blob_exist_prob = mask_exist_prob.pow(2).sum(dim=(-1, -2)) / (
                mask_exist_prob.sum(dim=(-1, -2)).clamp_min(1e-6)
            )
            blob_exist_prob = blob_exist_prob.round().unsqueeze(-1)
            blob_weight = blob_exist_prob
            self.visual_vars["mod_pred_logits"] = pred_logits[:, :, 0]

        bg_features = self.generate_background_feature(
            {
                "images": images,
                "mask_exist_prob": mask_exist_prob,
                "texton_feature_map": texton_feature_map,
            }
        )

        if self.hparams.use_bg:
            assert False
        else:
            bg_mask_exist_prob = torch.zeros_like(mask_exist_prob[:, 0][:, None])
            fg_mask_exist_prob = mask_exist_prob  # all is forground
            fg_blob_exist_prob = blob_exist_prob
            fg_blob_weight = blob_weight

        if self.hparams.use_shifted_blob_sp_params:
            if self.hparams.use_bg:
                shift_features = self.mlp_shift_head(
                    input_dict["seg_outputs"]["decoder_outputs"][:, :-1]
                )
            else:
                shift_features = self.mlp_shift_head(
                    input_dict["seg_outputs"]["decoder_outputs"]
                )
        else:
            shift_features = None

        (
            blob_xy,
            blob_covs,
            blob_features,
            shifted_blob_xy,
            shifted_blob_covs,
            shifts,
            coords,
            rot_features,
        ) = self.generate_blob_sp_feat_params(
            fg_mask_exist_prob,
            texton_feature_map,
            shift_features,
            rot_feature_map=rot_feature_map,
        )

        assert (
            fg_mask_exist_prob.shape[1] == self.hparams.num_queries
            and blob_xy.shape[1] == blob_covs.shape[1]
            and blob_xy.shape[1] == self.hparams.num_queries
            and fg_blob_exist_prob.shape[1] == self.hparams.num_queries
            and fg_blob_weight.shape[1] == self.hparams.num_queries
            and blob_features.shape[1] == self.hparams.num_queries
        )

        segment = {
            "mask_exist_prob": mask_exist_prob,
            "fg_mask_exist_prob": fg_mask_exist_prob,
            "bg_mask_exist_prob": bg_mask_exist_prob,
            "img_coords": coords,
        }

        if self.hparams.use_shifted_blob_sp_params:
            if rot_features != None:
                one_level_blob = {
                    "xy": shifted_blob_xy,
                    "covs": shifted_blob_covs,
                    "per_texel_feature": blob_features,
                    "rot_feature": rot_features,
                    "exist_prob": fg_blob_exist_prob,
                    "weights": fg_blob_weight,
                }
            else:
                one_level_blob = {
                    "xy": shifted_blob_xy,
                    "covs": shifted_blob_covs,
                    "per_texel_feature": blob_features,
                    "exist_prob": fg_blob_exist_prob,
                    "weights": fg_blob_weight,
                }

            blob = {
                "layer_tree_levels": {-1: [one_level_blob]},
                "img": images,
                "bg_features": bg_features,
                "global_features": None,
                "layer_global_features": None,
                "tau": tau,
                "img_mask": self.compute_img_mask(images),
                "unshifted": {"xy": blob_xy, "covs": blob_covs},
                "shifts": shifts,
            }

        else:
            if self.training:
                assert False

            if rot_features != None:
                one_level_blob = {
                    "xy": blob_xy,
                    "covs": blob_covs,
                    "per_texel_feature": blob_features,
                    "rot_feature": rot_features,
                    "exist_prob": fg_blob_exist_prob,
                    "weights": fg_blob_weight,
                }

            else:
                one_level_blob = {
                    "xy": blob_xy,
                    "covs": blob_covs,
                    "per_texel_feature": blob_features,
                    "exist_prob": fg_blob_exist_prob,
                    "weights": fg_blob_weight,
                }

            blob = {
                "layer_tree_levels": {-1: [one_level_blob]},
                "img": images,
                "bg_features": bg_features,
                "global_features": None,
                "layer_global_features": None,
                "tau": tau,
                "img_mask": self.compute_img_mask(images),
            }

        return segment, blob

    def forward_func(self, batched_inputs):
        self.visual_vars = {}

        self.global_step = batched_inputs["global_step"]

        orig_img = batched_inputs["img_input"]["img"]

        H, W = orig_img.shape[-2:]
        assert H == W and (H == 128 or H == 256)
        if H == 256:
            img = F.interpolate(orig_img, scale_factor=0.5, mode="bilinear")
        elif H == 128:
            img = orig_img
        else:
            assert False

        images_for_seg = (img + 1) / 2 * 255
        images_for_seg = [
            (x - self.pixel_mean) / self.pixel_std for x in images_for_seg
        ]
        images_for_seg = ImageList.from_tensors(images_for_seg, self.size_divisibility)

        if "freeze_seg" in self.hparams:
            if self.hparams.freeze_seg:
                for params in self.backbone.parameters():
                    params.requires_grad = False
                for params in self.sem_seg_head.parameters():
                    params.requires_grad = False
            else:
                assert False

        features = self.backbone(images_for_seg.tensor)
        outputs, segment_feature_map = self.sem_seg_head(features)

        if self.hparams.encoder == "disentangled_local_texton":
            texton_feature_map, rot_feature_map = self.generate_texton_feature_map(
                orig_img, features, segment_feature_map
            )
        else:
            texton_feature_map = self.generate_texton_feature_map(
                orig_img, features, segment_feature_map
            )
            rot_feature_map = None

        assert (
            outputs["pred_logits"].shape[1]
            == self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        )
        if self.hparams.use_bg:
            assert outputs["pred_logits"].shape[1] == self.hparams.num_queries + 1
        else:
            assert outputs["pred_logits"].shape[1] == self.hparams.num_queries

        input_dict = {
            "seg_outputs": outputs,
            "texton_feature_map": texton_feature_map,
            "rot_feature_map": rot_feature_map,
            # "bg_features":bg_features,
            "images": orig_img,
            "global_step": batched_inputs["global_step"],
        }

        segments, blobs = self.generate_segment_and_blob(input_dict)
        self.output = blobs
        self.output["segments"] = segments

        with torch.no_grad():
            if self.hparams.original_res_mask:
                segmentation = segments["mask_exist_prob"].argmax(dim=1)
                sp_max_mask_exist_prob = segments["mask_exist_prob"].amax(dim=1)

                self.visual_vars.update(
                    {
                        "img": orig_img,
                        "bg_mask_exist_prob": segments["bg_mask_exist_prob"],
                        "texton_feature_map": texton_feature_map,
                        "segment_feature_map": segment_feature_map,
                        "sp_max_mask_exist_prob": sp_max_mask_exist_prob,
                        "mask_exist_prob": segments["mask_exist_prob"],
                        "segmentation": segmentation,
                        "blob_weights": blobs["layer_tree_levels"][-1][0]["weights"],
                        "blob_exist_prob": blobs["layer_tree_levels"][-1][0][
                            "exist_prob"
                        ],
                    }
                )
            else:
                assert False

        return
