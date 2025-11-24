import copy
import dataclasses
from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from src.inference.demo import (
    ApplicationConfig,
    ApplicationRegistery,
    ApplicationType,
    BaseApplication,
    Demo,
    DemoConfig,
    load_image_tensor,
)


def _normalize_patch_shape(patch_shape: Sequence[int]) -> Tuple[int, int]:
    if len(patch_shape) != 2:
        raise ValueError("patch_shape must contain exactly two elements (H, W).")
    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("patch dimensions must be positive.")
    return patch_h, patch_w


def _normalize_step(
    step: Sequence[Union[float, int]], patch_shape: Tuple[int, int]
) -> Tuple[int, int]:
    if len(step) != 2:
        raise ValueError("step must contain exactly two elements.")

    def _convert(value: Union[float, int], patch: int) -> int:
        if isinstance(value, float):
            stride = int(patch * value)
        else:
            stride = int(value)
        if stride <= 0:
            raise ValueError("Stride must be positive.")
        return stride

    return _convert(step[0], patch_shape[0]), _convert(step[1], patch_shape[1])


def _pad_to_minimum_size(img: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    if not img.is_contiguous():
        img = img.contiguous()

    pad_h = max(patch_h - img.size(2), 0)
    pad_w = max(patch_w - img.size(3), 0)

    if pad_h == 0 and pad_w == 0:
        return img

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))


def _compute_start_positions(length: int, window: int, stride: int) -> list[int]:
    if length < window:
        raise ValueError("Window cannot be larger than the padded dimension.")

    max_start = length - window
    positions = list(range(0, max_start + 1, stride))
    if not positions:
        positions = [0]
    elif positions[-1] != max_start:
        positions.append(max_start)
    return positions


def _sliding_window_view(img: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    if not img.is_contiguous():
        img = img.contiguous()

    stride_b, stride_c, stride_h, stride_w = img.stride()
    view_shape = (
        img.size(0),
        img.size(1),
        img.size(2) - patch_h + 1,
        img.size(3) - patch_w + 1,
        patch_h,
        patch_w,
    )
    view_strides = (stride_b, stride_c, stride_h, stride_w, stride_h, stride_w)
    return img.as_strided(view_shape, view_strides)


def _crop_to_shape(
    img: torch.Tensor, target_shape: Sequence[int], patch_h: int, patch_w: int
) -> torch.Tensor:
    height, width = int(target_shape[0]), int(target_shape[1])
    if height < patch_h:
        pad_top = (patch_h - height) // 2
        pad_bottom = patch_h - height - pad_top
        img = img[:, :, pad_top : img.size(2) - pad_bottom, :]
    if width < patch_w:
        pad_left = (patch_w - width) // 2
        pad_right = patch_w - width - pad_left
        img = img[:, :, :, pad_left : img.size(3) - pad_right]
    return img


def _validate_image_tensor(img: torch.Tensor) -> None:
    if img.dim() != 4:
        raise ValueError("img must be a 4D BCHW tensor.")


def _validate_patches_tensor(patches: torch.Tensor) -> None:
    if patches.dim() != 5:
        raise ValueError(
            "patches must be a 5D tensor (N, B, C, patch_h, patch_w) or batch_first equivalent."
        )


def extract_patches_2d(
    img: torch.Tensor,
    patch_shape: Sequence[int],
    step: Sequence[Union[float, int]] = (1.0, 1.0),
    batch_first: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """
    Extract 2D patches from a 4D BCHW tensor while ensuring the final row/column
    of patches always aligns with the image boundary.
    """
    _validate_image_tensor(img)

    patch_h, patch_w = _normalize_patch_shape(patch_shape)
    stride_h, stride_w = _normalize_step(step, (patch_h, patch_w))
    padded_img = _pad_to_minimum_size(img, patch_h, patch_w)

    row_starts = _compute_start_positions(padded_img.size(2), patch_h, stride_h)
    col_starts = _compute_start_positions(padded_img.size(3), patch_w, stride_w)

    row_selector = torch.as_tensor(
        row_starts, dtype=torch.long, device=padded_img.device
    )
    col_selector = torch.as_tensor(
        col_starts, dtype=torch.long, device=padded_img.device
    )

    windows = _sliding_window_view(padded_img, patch_h, patch_w)
    windows = windows.index_select(2, row_selector)
    windows = windows.index_select(3, col_selector)

    # Arrange patches as (num_rows, num_cols, batch, channels, patch_h, patch_w)
    patches = windows.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(
        -1, padded_img.size(0), padded_img.size(1), patch_h, patch_w
    )

    if batch_first:
        patches = patches.permute(1, 0, 2, 3, 4)

    return patches, len(row_starts), len(col_starts)


def _compute_patch_overlap_bounds(
    idx_row: int,
    idx_col: int,
    steps: Sequence[Union[float, int]],
    patch_shape: Sequence[Union[float, int]],
    image_shape: Sequence[int],
    curr_xy: torch.Tensor,
) -> Tuple[float, float, float, float, torch.Tensor]:
    """
    Compute the bounding box describing the overlapping region between the current patch
    and the previously added patches, following the same normalized coordinate system
    used during style transfer inference.
    """
    if len(steps) != 2:
        raise ValueError("steps must have exactly two elements (vertical, horizontal).")
    if len(patch_shape) != 2:
        raise ValueError("patch_shape must have exactly two elements (height, width).")
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width).")

    patch_h, patch_w = float(patch_shape[0]), float(patch_shape[1])
    step_h, step_w = float(steps[0]), float(steps[1])
    img_h, img_w = float(image_shape[0]), float(image_shape[1])

    span_y = 6.0 * patch_h / img_h
    span_x = 6.0 * patch_w / img_w
    step_y = 6.0 * step_h / img_h
    step_x = 6.0 * step_w / img_w

    if curr_xy.dim() < 3 or curr_xy.size(0) == 0:
        raise ValueError("curr_xy must have shape (B, N, 2) with B > 0.")

    curr_xy_slice = curr_xy[0]

    if idx_row == 0 and idx_col == 0:
        upper_patch_y = 1e5
        bottom_patch_y = -1e5
        left_patch_x = 1e5
        right_patch_x = -1e5
        curr_overlapped = torch.logical_and(
            curr_xy_slice[:, 0] < right_patch_x, curr_xy_slice[:, 1] < bottom_patch_y
        )
    elif idx_row == 0:
        upper_patch_y = -3.0
        bottom_patch_y = -3.0 + span_y
        left_patch_x = step_x * idx_col - 3.0
        right_patch_x = span_x + step_x * (idx_col - 1) - 3.0
        curr_overlapped = curr_xy_slice[:, 0] < right_patch_x
    elif idx_col == 0:
        upper_patch_y = step_y * idx_row - 3.0
        bottom_patch_y = span_y + step_y * (idx_row - 1) - 3.0
        left_patch_x = -3.0
        right_patch_x = -3.0 + span_x
        curr_overlapped = curr_xy_slice[:, 1] < bottom_patch_y
    else:
        upper_patch_y = step_y * idx_row - 3.0
        bottom_patch_y = span_y + step_y * (idx_row - 1) - 3.0
        left_patch_x = step_x * idx_col - 3.0
        right_patch_x = span_x + step_x * (idx_col - 1) - 3.0
        curr_overlapped = torch.logical_or(
            curr_xy_slice[:, 0] < right_patch_x, curr_xy_slice[:, 1] < bottom_patch_y
        )

    return left_patch_x, upper_patch_y, right_patch_x, bottom_patch_y, curr_overlapped


@dataclasses.dataclass
class StyleTransferConfig(ApplicationConfig):
    style_folder: str
    content_patch_shape: tuple[int, int]
    patch_steps: tuple[int, int]
    content_size: int
    style_size: tuple[int, int]
    content_image_paths: list[Path]
    current_folder: Path
    demo_root: Path = Path("./demo/style_transfer")
    content_transform: transforms.Compose = dataclasses.field(init=False, repr=False)
    style_transform: transforms.Compose = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.content_transform = transforms.Compose(
            [
                transforms.Resize(self.content_size),
                transforms.CenterCrop(self.content_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.style_transform = transforms.Compose(
            [
                transforms.Resize(self.style_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def prepare_demo_directory(self, folder_name: Union[str, Path]) -> Path:
        folder = str(folder_name)
        output_dir = self.demo_root / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class StyleTransferHelper:
    @staticmethod
    def compute_added_patch_overlap_mask(
        added_xy: torch.Tensor,
        left: float,
        upper: float,
        right: float,
        bottom: float,
    ) -> torch.Tensor:
        """Return mask of added patches that fall inside the provided bounding box."""
        return (
            (added_xy[0, :, 0] > left)
            & (added_xy[0, :, 1] > upper)
            & (added_xy[0, :, 0] < right)
            & (added_xy[0, :, 1] < bottom)
        )

    @staticmethod
    def compute_patch_overlap_bounds(
        idx_row: int,
        idx_col: int,
        steps: Sequence[Union[float, int]],
        patch_shape: Sequence[Union[float, int]],
        image_shape: Sequence[int],
        curr_xy: torch.Tensor,
    ) -> Tuple[float, float, float, float, torch.Tensor]:
        """Wrapper around `compute_patch_overlap_bounds` scoped for style transfer usage."""
        return _compute_patch_overlap_bounds(
            idx_row=idx_row,
            idx_col=idx_col,
            steps=steps,
            patch_shape=patch_shape,
            image_shape=image_shape,
            curr_xy=curr_xy,
        )

    @staticmethod
    def patch_gaussian_classification(
        curr_patch: dict[str, torch.Tensor],
        added_patches: dict[str, torch.Tensor],
        curr_overlapped: torch.Tensor,
        added_overlapped: torch.Tensor,
    ) -> Tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """
        Split Gaussians into overlapped and non-overlapped subsets for both
        the current patch and the already-added patches.
        """
        curr_overlapped_indexes = curr_overlapped.nonzero(as_tuple=False).squeeze(1)
        added_overlapped_indexes = added_overlapped.nonzero(as_tuple=False).squeeze(1)
        curr_nonoverlapped_indexes = (
            (~curr_overlapped).nonzero(as_tuple=False).squeeze(1)
        )
        added_nonoverlapped_indexes = (
            (~added_overlapped).nonzero(as_tuple=False).squeeze(1)
        )

        nonoverlapped_curr_patch: dict[str, torch.Tensor] = {}
        nonoverlapped_added_patches: dict[str, torch.Tensor] = {}
        overlapped_curr_patch: dict[str, torch.Tensor] = {}
        overlapped_added_patches: dict[str, torch.Tensor] = {}

        for gauss_key, gauss_param in curr_patch.items():
            nonoverlapped_curr_patch[gauss_key] = torch.index_select(
                gauss_param, dim=1, index=curr_nonoverlapped_indexes
            )
            nonoverlapped_added_patches[gauss_key] = torch.index_select(
                added_patches[gauss_key], dim=1, index=added_nonoverlapped_indexes
            )
            overlapped_curr_patch[gauss_key] = torch.index_select(
                gauss_param, dim=1, index=curr_overlapped_indexes
            )
            overlapped_added_patches[gauss_key] = torch.index_select(
                added_patches[gauss_key], dim=1, index=added_overlapped_indexes
            )

        return (
            nonoverlapped_curr_patch,
            nonoverlapped_added_patches,
            overlapped_curr_patch,
            overlapped_added_patches,
        )


@ApplicationRegistery.register_application(ApplicationType.STYLE_TRANSFER)
class StyleTransfer(BaseApplication):
    """Production-ready orchestrator for running style-transfer demos."""

    MAX_STYLE_IMAGES = 30

    def __init__(self, config: StyleTransferConfig) -> None:
        super().__init__()
        self.config = config
        self.content_transform = config.content_transform
        self.style_transform = config.style_transform

        self.paper_demos = self.get_new_inference()

    def run(self) -> None:
        """Run the refactored high-resolution pipeline backed by TexAutoEncoderInference."""
        self._run_for_all_pairs(self._run_local_highres)

    def _run_for_all_pairs(
        self,
        executor: Callable[[torch.Tensor, torch.Tensor, str, str, str], None],
    ) -> None:
        folder_name = str(self.config.current_folder)
        output_dir = self.config.prepare_demo_directory(folder_name)
        style_paths = self._collect_style_image_paths()
        content_paths = [Path(path) for path in self.config.content_image_paths]
        if not content_paths:
            raise ValueError("config.content_image_paths is empty; nothing to process.")

        for content_path in content_paths:
            content_tensor, content_name = self._load_content_image(content_path)
            self._save_snapshot(content_tensor, output_dir / f"{content_name}.png")
            content_tensor = self._move_to_device(content_tensor)

            for style_path in style_paths:
                style_tensor, style_name = self._load_style_image(style_path)
                self._save_snapshot(style_tensor, output_dir / f"{style_name}.png")
                style_tensor = self._move_to_device(style_tensor)

                result_name = f"{content_name}_{style_name}"
                executor(
                    content_tensor, style_tensor, folder_name, style_name, result_name
                )

    def _collect_style_image_paths(self) -> list[Path]:
        style_dir = Path(self.config.style_folder)
        if not style_dir.exists():
            raise FileNotFoundError(f"Style folder does not exist: {style_dir}")
        style_images = sorted(path for path in style_dir.iterdir() if path.is_file())
        if not style_images:
            raise ValueError(f"No style images were found under {style_dir}")
        return style_images[: self.MAX_STYLE_IMAGES]

    def _load_content_image(self, image_path: Path) -> tuple[torch.Tensor, str]:
        tensor = load_image_tensor(image_path, self.content_transform)
        return tensor, image_path.stem

    def _load_style_image(self, image_path: Path) -> tuple[torch.Tensor, str]:
        tensor = load_image_tensor(image_path, self.style_transform)
        return tensor, image_path.stem

    def _save_snapshot(self, tensor: torch.Tensor, output_path: Path) -> None:
        if output_path.exists():
            return
        vutils.save_image(tensor.detach().cpu(), str(output_path), normalize=True)

    def _save_style_visuals(
        self,
        layout_style: dict,
        style_name: str,
        output_dir: Path,
    ) -> None:
        if not self.config.save_visuals:
            return
        recon_style = self.paper_demos.decode(layout_style)
        vis_gaussian_contri = layout_style["layer_level_layouts"][-1][0][
            "vis_gaussian_contri"
        ]
        vutils.save_image(
            vis_gaussian_contri,
            str(output_dir / f"gaussian_{style_name}.png"),
            normalize=True,
        )
        vutils.save_image(
            recon_style,
            str(output_dir / f"recon_{style_name}.png"),
            normalize=True,
        )

    def _save_highres_visuals(
        self,
        layout: dict,
        recon_highres: torch.Tensor,
        result_name: str,
        output_dir: Path,
        tree_level_content: dict,
    ) -> None:
        if not self.config.save_visuals:
            return
        vutils.save_image(
            recon_highres,
            str(output_dir / f"recon_highres_{result_name}.png"),
            normalize=True,
        )
        gaussian_contri = layout["layer_level_layouts"][-1][0]["gaussian_contri"]
        generator = torch.Generator(device=self.device).manual_seed(42)
        rand_color = torch.rand(
            tree_level_content["xy"].shape[1],
            3,
            device=self.device,
            generator=generator,
        )
        vis_gaussian_contri = torch.einsum(
            "nc,bhwnd->bchwd", rand_color, gaussian_contri[:, :, :, 1:]
        ).squeeze(-1)
        vutils.save_image(
            vis_gaussian_contri,
            str(output_dir / f"gaussain_highres_{result_name}.png"),
            normalize=True,
        )

    def _resolve_patch_seams(
        self,
        *,
        added_patches: dict[str, torch.Tensor],
        nonoverlapped_curr_patch: dict[str, torch.Tensor],
        nonoverlapped_added_patches: dict[str, torch.Tensor],
        overlapped_curr_patch: dict[str, torch.Tensor],
        overlapped_added_patches: dict[str, torch.Tensor],
        interp_factor: float = 0.5,
    ) -> None:
        """Blend overlapping Gaussians between existing and current patches to hide seams."""
        has_overlap = (
            overlapped_curr_patch["xy"].shape[1] != 0
            and overlapped_added_patches["xy"].shape[1] != 0
        )
        if not has_overlap:
            for gauss_key, gauss_param in nonoverlapped_curr_patch.items():
                added_patches[gauss_key] = torch.cat(
                    (added_patches[gauss_key], gauss_param), dim=1
                )
            return

        valid_sxy = overlapped_curr_patch["xy"][0]
        valid_txy = overlapped_added_patches["xy"][0]
        valid_scovs = overlapped_curr_patch["covs"][0]
        valid_tcovs = overlapped_added_patches["covs"][0]
        valid_sfeat = overlapped_curr_patch["per_texel_feature"][0]
        valid_tfeat = overlapped_added_patches["per_texel_feature"][0]
        valid_srotfeat = overlapped_curr_patch["rot_feature"][0]
        valid_trotfeat = overlapped_added_patches["rot_feature"][0]

        cost_matrix = (
            torch.cdist(valid_sxy.unsqueeze(0), valid_txy.unsqueeze(0))
            .squeeze(0)
            .cpu()
            .numpy()
        )
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        row_idx = torch.as_tensor(row_idx, device=self.device, dtype=torch.long)
        col_idx = torch.as_tensor(col_idx, device=self.device, dtype=torch.long)

        interp_xy = valid_sxy[row_idx] * interp_factor + valid_txy[col_idx] * (
            1 - interp_factor
        )
        interp_feat = F.normalize(
            valid_sfeat[row_idx] * interp_factor
            + valid_tfeat[col_idx] * (1 - interp_factor),
            p=2.0,
            dim=-1,
        )
        interp_covs = valid_scovs[row_idx] * interp_factor + valid_tcovs[col_idx] * (
            1 - interp_factor
        )
        interp_rotfeat = F.normalize(
            valid_srotfeat[row_idx] * interp_factor
            + valid_trotfeat[col_idx] * (1 - interp_factor),
            p=2.0,
            dim=-1,
        )

        overlap_count = row_idx.shape[0]
        merged_patch = {
            "xy": interp_xy[None],
            "covs": interp_covs[None],
            "per_texel_feature": interp_feat[None],
            "rot_feature": interp_rotfeat[None],
            "exist_prob": torch.ones(1, overlap_count, 1, device=self.device),
            "weights": torch.ones(1, overlap_count, 1, device=self.device),
        }

        for gauss_key, gauss_param in nonoverlapped_curr_patch.items():
            added_patches[gauss_key] = torch.cat(
                (
                    nonoverlapped_added_patches[gauss_key],
                    gauss_param,
                    merged_patch[gauss_key],
                ),
                dim=1,
            )

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        target_device = torch.device(self.device)
        if tensor.device == target_device:
            return tensor
        return tensor.to(target_device)

    def _run_local_highres(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        folder_name: str,
        style_name: str,
        result_name: str,
    ) -> None:
        output_dir = self.config.prepare_demo_directory(folder_name)

        result = self.style_transfer_highres(content_img, style_img)
        layout_style = result["layout_style"]
        layout = result["layout"]
        recon_highres = result["recon_highres"]
        tree_level_content = result["tree_level_content"]

        self._save_style_visuals(layout_style, style_name, output_dir)
        self._save_highres_visuals(
            layout,
            recon_highres,
            result_name,
            output_dir,
            tree_level_content,
        )

    def _process_content_patch(
        self,
        patch_idx: int,
        num_cols: int,
        content_patches: torch.Tensor,
        scale_coe: torch.Tensor,
        scale_coe_vec: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[tuple[int, int], dict[str, torch.Tensor]]:
        idx_row = patch_idx // num_cols
        idx_col = patch_idx % num_cols

        filtered_patch = self.paper_demos.encode_hard(content_patches[:, patch_idx])

        translation = filtered_patch["xy"].new_tensor(
            [
                (6 / (image_width / self.config.patch_steps[1])) * idx_col
                + (-3 + 6 / (2 * scale_coe_vec[0])),
                (6 / (image_height / self.config.patch_steps[0])) * idx_row
                + (-3 + 6 / (2 * scale_coe_vec[0])),
            ]
        )[None, None]
        filtered_patch["xy"] = (filtered_patch["xy"] / scale_coe) + translation

        L = torch.linalg.cholesky(filtered_patch["covs"])
        covs_scale = torch.diag_embed(1 / scale_coe_vec).to(
            device=filtered_patch["covs"].device,
            dtype=filtered_patch["covs"].dtype,
        )
        scaled_L = L @ covs_scale
        filtered_patch["covs"] = scaled_L @ scaled_L.transpose(-1, -2)

        return (idx_row, idx_col), filtered_patch

    @torch.no_grad()
    def style_transfer_highres(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
    ) -> dict[str, Any]:
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        H, W = content_img.shape[-2:]

        content_patches, _, num_cols = extract_patches_2d(
            content_img,
            patch_shape=self.config.content_patch_shape,
            step=self.config.patch_steps,
            batch_first=True,
        )
        content_patches = content_patches.to(self.device)
        num_patches = content_patches.shape[1]

        style_blob = self.paper_demos.encode(style_img)
        layout_style = self.paper_demos.splat(style_blob)

        tree_level_style = style_blob["layer_tree_levels"][-1][0]
        style_feature = self.paper_demos.compute_hard_mean_feature(tree_level_style)

        scale_coe = torch.tensor(
            [
                W / self.config.content_patch_shape[1],
                H / self.config.content_patch_shape[0],
            ],
            dtype=content_img.dtype,
            device=self.device,
        )[None, None]
        scale_coe_vec = scale_coe.squeeze()

        processed_patches = [
            self._process_content_patch(
                patch_idx=i,
                num_cols=num_cols,
                content_patches=content_patches,
                scale_coe=scale_coe,
                scale_coe_vec=scale_coe_vec,
                image_width=W,
                image_height=H,
            )
            for i in range(num_patches)
        ]
        filtered_patches = dict(processed_patches)

        added_patches = {}
        first_patch = filtered_patches[(0, 0)]
        for gauss_key, gauss_param in first_patch.items():
            tmp_shape = list(gauss_param.shape)
            tmp_shape[1] = 0
            added_patches[gauss_key] = torch.zeros(
                tmp_shape, device=self.device, dtype=gauss_param.dtype
            )

        for i in range(num_patches):
            idx_row = i // num_cols
            idx_col = i % num_cols
            curr_patch = filtered_patches[(idx_row, idx_col)]

            (
                left_patch_x,
                upper_patch_y,
                right_patch_x,
                bottom_patch_y,
                curr_overlapped,
            ) = StyleTransferHelper.compute_patch_overlap_bounds(
                idx_row=idx_row,
                idx_col=idx_col,
                steps=self.config.patch_steps,
                patch_shape=self.config.content_patch_shape,
                image_shape=(H, W),
                curr_xy=curr_patch["xy"],
            )

            added_overlapped = StyleTransferHelper.compute_added_patch_overlap_mask(
                added_patches["xy"],
                left_patch_x,
                upper_patch_y,
                right_patch_x,
                bottom_patch_y,
            )

            (
                nonoverlapped_curr_patch,
                nonoverlapped_added_patches,
                overlapped_curr_patch,
                overlapped_added_patches,
            ) = StyleTransferHelper.patch_gaussian_classification(
                curr_patch=curr_patch,
                added_patches=added_patches,
                curr_overlapped=curr_overlapped,
                added_overlapped=added_overlapped,
            )

            self._resolve_patch_seams(
                added_patches=added_patches,
                nonoverlapped_curr_patch=nonoverlapped_curr_patch,
                nonoverlapped_added_patches=nonoverlapped_added_patches,
                overlapped_curr_patch=overlapped_curr_patch,
                overlapped_added_patches=overlapped_added_patches,
            )

        tree_level_content = added_patches
        content_feature = self.paper_demos.compute_hard_mean_feature(tree_level_content)
        new_per_texel_feature = F.normalize(
            tree_level_content["per_texel_feature"]
            - content_feature[:, None, :]
            + style_feature[:, None, :],
            dim=-1,
        )
        tree_level_content["per_texel_feature"] = new_per_texel_feature

        new_blob = copy.deepcopy(style_blob)
        new_blob["layer_tree_levels"][-1][0] = tree_level_content

        layout = self.paper_demos.splat(new_blob, img_size=256)
        layout["layer_level_layouts"][-1][0]["score_img"] = F.interpolate(
            layout["layer_level_layouts"][-1][0]["score_img"], size=512, mode="bilinear"
        )

        recon_highres = self.paper_demos.decode(layout)

        return {
            "layout_style": layout_style,
            "layout": layout,
            "recon_highres": recon_highres,
            "tree_level_content": tree_level_content,
        }


if __name__ == "__main__":
    content_image_path = "dataset/content_imgs/empire_state_building.jpg"

    tmp_config = StyleTransferConfig(
        style_folder="dataset/textures",
        content_patch_shape=(128, 128),
        patch_steps=(96, 96),
        content_size=684,
        style_size=(128, 128),
        content_image_paths=[Path(content_image_path)],
        current_folder=Path("demo"),
        save_visuals=True,
    )

    demo_config = DemoConfig(ApplicationType.STYLE_TRANSFER, tmp_config)
    demo = Demo(demo_config)
    demo.run()
