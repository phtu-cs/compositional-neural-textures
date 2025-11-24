import copy
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from PIL import Image
from torchvision import transforms


class TexAutoEncoderInference:
    def __init__(
        self,
        ckpt_path: Path,
        device=None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with initialize(config_path="../../configs/inference"):
            cfg = compose(config_name="generator")
            self.generator = hydra.utils.instantiate(cfg)

            cfg = compose(config_name="layout_net")
            self.layout_net = hydra.utils.instantiate(cfg)

            cfg = compose(config_name="layout_splat")
            self.layout_splat = hydra.utils.instantiate(cfg)

        # the checkpoint was saved from pytorch lighthing LightningModule, so it also contains information about losses, etc
        # but inference only need the modules below

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        layout_net_weights = {
            k.removeprefix("layout_net."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("layout_net.")
        }
        layout_splat_weights = {
            k.removeprefix("layout_splat."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("layout_splat.")
        }
        generator_weights = {
            k.removeprefix("generator."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("generator.")
        }

        # Due to this issue https://github.com/facebookresearch/Mask2Former/issues/95
        # below is a workaround, otherwise we should do "self.layout_net.load_state_dict(layout_net_weights)""
        layout_net_state_dict = self.layout_net.state_dict()
        layout_net_state_dict.update(layout_net_weights)
        self.layout_net.load_state_dict(layout_net_state_dict)
        self.layout_net.to(self.device).eval()

        self.layout_splat.load_state_dict(layout_splat_weights)
        self.layout_splat.to(self.device).eval()

        self.generator.load_state_dict(generator_weights)
        self.generator.to(self.device).eval()

        # self.layout_net = self.model.layout_net
        # self.layout_splat = self.model.layout_splat
        # self.generator = self.model.generator

        # not really needed in inference but used in training, this serves as a placeholder only
        self.global_step = 0

    def encode(self, image_torch):
        input_dict = {"img": image_torch}
        layout_net_input = {
            "img_input": input_dict,
            "global_step": self.global_step,
        }
        blob, _ = self.layout_net(layout_net_input)
        return blob

    def encode_hard(self, image_torch: torch.Tensor, threshold: float = 0.5):
        """Encode and keep only Gaussians whose existence prob exceeds the threshold."""
        blob = self.encode(image_torch)
        tree_level = blob["layer_tree_levels"][-1][0]
        exist_prob = tree_level["exist_prob"]
        filter_indexes = torch.nonzero(
            exist_prob.squeeze() >= threshold,
            as_tuple=False,
        )[:, 0]
        filtered = {}
        for key in (
            "xy",
            "covs",
            "per_texel_feature",
            "rot_feature",
            "exist_prob",
            "weights",
        ):
            filtered[key] = torch.index_select(
                tree_level[key],
                dim=1,
                index=filter_indexes,
            )
        return filtered

    def splat(self, blob, feature_map_sizes=None, img_size=None):
        tree_splat = getattr(self.layout_splat, "tree_splat", None)
        original_img_size = None
        if img_size is not None and tree_splat is not None:
            original_img_size = copy.deepcopy(tree_splat.img_size)
            tree_splat.img_size = img_size
        try:
            layout, _ = self.layout_splat(blob)
        finally:
            if (
                img_size is not None
                and tree_splat is not None
                and original_img_size is not None
            ):
                tree_splat.img_size = original_img_size
        layout["global_step"] = self.global_step
        if feature_map_sizes is not None:
            layout["feature_map_sizes"] = feature_map_sizes
        return layout

    def decode(self, layout):
        gen_output, _ = self.generator(layout)
        recon = gen_output["recon"][-1].flatten(1, 2)
        return recon

    def compute_hard_mean_feature(self, tree_level):
        per_texel_feature = tree_level["per_texel_feature"]
        exist_prob = tree_level["exist_prob"]
        hard_exist_prob = exist_prob.round()
        weighted_sum = (per_texel_feature * hard_exist_prob).sum(dim=1)
        denom = hard_exist_prob.sum(dim=1).clamp_min(1e-8)
        mean_feature = weighted_sum / denom
        return F.normalize(mean_feature, dim=-1)

    def recon(self, image: torch.Tensor):
        with torch.no_grad():
            image = image.to(self.device)
            blob = self.encode(image)
            layout = self.splat(blob)
            return self.decode(layout)


def build_content_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_image(image_path: Path, image_size: int) -> torch.Tensor:
    transform = build_content_transform(image_size)
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
    return tensor
