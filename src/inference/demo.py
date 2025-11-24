import dataclasses
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Optional, Tuple, Type

import torch
from PIL import Image
from torchvision import transforms

from src.inference.core import TexAutoEncoderInference


class ApplicationType(str, Enum):
    TEXTURE_TRANSFER = "texture_transfer"
    STYLE_TRANSFER = "style_transfer"


@dataclasses.dataclass
class ApplicationConfig:
    save_visuals: bool


@dataclasses.dataclass
class DemoConfig:
    app_type: ApplicationType
    app_config: ApplicationConfig


@dataclasses.dataclass
class TextureTransferConfig(ApplicationConfig):
    class TransferMode(str, Enum):
        SHIFT = "shift"
        REPLACE = "replace"

    mode: TransferMode


class ApplicationRegistery:
    _registry: ClassVar[Dict[ApplicationType, Type["BaseApplication"]]] = {}

    @classmethod
    def register_application(cls, application_type: ApplicationType):
        def decorator(application_cls: Type[BaseApplication]) -> Type[BaseApplication]:
            if not issubclass(application_cls, BaseApplication):
                raise TypeError(
                    "Registered application must inherit from BaseApplication."
                )
            cls._registry[application_type] = application_cls
            return application_cls

        return decorator

    @classmethod
    def registry(cls) -> MappingProxyType:
        return MappingProxyType(cls._registry)


class BaseApplication:
    _new_inference_class: ClassVar[Optional[TexAutoEncoderInference]] = None

    def __init__(self) -> None:
        pass

    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def get_new_inference(cls) -> TexAutoEncoderInference:
        if cls._new_inference_class is None:
            cls._new_inference_class = cls._create_new_inference_class()
        return cls._new_inference_class

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _create_new_inference_class(cls) -> TexAutoEncoderInference:
        ckpt_path = "./ckpts/finetuned-model256.ckpt"
        return TexAutoEncoderInference(ckpt_path=Path(ckpt_path))


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    """Open image from disk, enforce RGB, apply transform and add batch dimension."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
    return tensor


class Demo:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._application = self._build_application(config.app_type, config.app_config)

    @staticmethod
    def available_applications() -> Tuple[ApplicationType, ...]:
        return tuple(ApplicationRegistery.registry().keys())

    @property
    def application(self) -> BaseApplication:
        return self._application

    def _build_application(
        self,
        application_type: ApplicationType,
        app_config: ApplicationConfig,
    ) -> BaseApplication:
        try:
            application_cls = ApplicationRegistery.registry()[application_type]
        except KeyError as exc:
            available = ", ".join(
                app_type.value for app_type in self.available_applications()
            )
            raise ValueError(
                f"{application_type} is not registered. "
                f"Available applications: {available or 'none'}."
            ) from exc
        return application_cls(app_config)

    def run(self, **kwargs) -> Any:
        return self._application.run(**kwargs)

    def run_with(
        self,
        application_type: ApplicationType,
        *,
        init_kwargs: Optional[Dict[str, Any]] = None,
        **run_kwargs: Any,
    ) -> Any:
        application = self._build_application(application_type, init_kwargs)
        return application.run(**run_kwargs)
