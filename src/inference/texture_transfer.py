import copy
import dataclasses
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from src.inference.core import build_content_transform
from src.inference.demo import (
    ApplicationConfig,
    ApplicationRegistery,
    ApplicationType,
    BaseApplication,
    Demo,
    DemoConfig,
    load_image_tensor,
)


@dataclasses.dataclass
class TextureTransferConfig(ApplicationConfig):
    src_path: Path
    tgt_path: Path
    current_folder: Path
    demo_root: Path = Path("./demo/texture_transfer")
    transfer_mode: str = "mean_shift"

    def __post_init__(self):
        assert (
            self.transfer_mode == "replace" or self.transfer_mode == "mean_shift"
        ), "there is only two modes: 'replace', 'mean_shift'."

    def prepare_demo_directory(self, folder_name: Union[str, Path]) -> Path:
        folder = str(folder_name)
        output_dir = self.demo_root / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


@ApplicationRegistery.register_application(ApplicationType.TEXTURE_TRANSFER)
class TextureTransfer(BaseApplication):
    def __init__(self, config: TextureTransferConfig) -> None:
        super().__init__()
        self.config = config
        self.inference = self.get_new_inference()
        self._image_transform = build_content_transform(
            self.inference.layout_splat.hparams.splat_func.hparams.img_size
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        tensor = load_image_tensor(path, self._image_transform)
        return tensor.to(self.inference.device)

    def _mean_shift_features(
        self,
        src_tree_level: dict,
        tgt_tree_level: dict,
    ) -> torch.Tensor:
        src_app_feat = self.inference.compute_hard_mean_feature(src_tree_level)
        tgt_texel_feats = tgt_tree_level["per_texel_feature"]
        tgt_app_feat = self.inference.compute_hard_mean_feature(tgt_tree_level)
        return F.normalize(
            (tgt_texel_feats - tgt_app_feat[:, None, :]) + src_app_feat[:, None, :],
            dim=-1,
        )

    def _replace_per_texel_features(
        self,
        src_tree_level: dict,
        num_target_blobs: int,
    ) -> torch.Tensor:
        src_feats = src_tree_level["per_texel_feature"][0]
        exist_prob = src_tree_level["exist_prob"][0, :, 0]
        candidate_indexes = torch.nonzero(exist_prob > 0, as_tuple=False).squeeze(1)
        if candidate_indexes.numel() == 0:
            candidate_indexes = torch.arange(
                src_feats.shape[0],
                device=src_feats.device,
                dtype=torch.long,
            )

        random_choice = torch.randint(
            low=0,
            high=candidate_indexes.numel(),
            size=(num_target_blobs,),
            device=src_feats.device,
        )
        sampled_indexes = candidate_indexes[random_choice]
        sampled_feats = torch.index_select(src_feats, dim=0, index=sampled_indexes)
        return F.normalize(sampled_feats.unsqueeze(0), dim=-1)

    def _compute_transferred_features(
        self,
        src_tree_level: dict,
        tgt_tree_level: dict,
    ) -> torch.Tensor:
        mode = getattr(self.config, "transfer_mode", "mean_shift").lower()
        if mode == "mean_shift":
            return self._mean_shift_features(src_tree_level, tgt_tree_level)
        if mode == "replace":
            num_target_blobs = tgt_tree_level["per_texel_feature"].shape[1]
            return self._replace_per_texel_features(src_tree_level, num_target_blobs)
        raise ValueError(f"Unsupported transfer_mode '{self.config.transfer_mode}'.")

    def texture_transfer(self, src_img: torch.Tensor, tgt_img: torch.Tensor):
        with torch.no_grad():
            src_blob = self.inference.encode(src_img)
            src_tree_level = src_blob["layer_tree_levels"][-1][0]

            tgt_blob = self.inference.encode(tgt_img)
            tgt_tree_level = tgt_blob["layer_tree_levels"][-1][0]

            new_feat = self._compute_transferred_features(
                src_tree_level, tgt_tree_level
            )

            transferred_blob = copy.deepcopy(tgt_blob)
            transferred_blob["layer_tree_levels"][-1][0]["per_texel_feature"] = new_feat

            layout = self.inference.splat(transferred_blob)
            recon = self.inference.decode(layout)
            return recon

    def run(self):
        src_img = self._load_image(self.config.src_path)
        tgt_img = self._load_image(self.config.tgt_path)
        recon = self.texture_transfer(src_img, tgt_img)

        output_dir = self.config.prepare_demo_directory(self.config.current_folder)
        output_path = output_dir / "reconstruction.png"
        vutils.save_image(recon.cpu(), str(output_path), normalize=True)
        return output_path


if __name__ == "__main__":
    src_path = Path("dataset/textures/68_1.png")
    tgt_path = Path("dataset/textures/87_25.png")

    tmp_config = TextureTransferConfig(
        src_path=src_path,
        tgt_path=tgt_path,
        current_folder=Path("demo"),
        save_visuals=True,
        transfer_mode="mean_shift",
    )

    demo_config = DemoConfig(ApplicationType.TEXTURE_TRANSFER, tmp_config)
    demo = Demo(demo_config)
    demo.run()
