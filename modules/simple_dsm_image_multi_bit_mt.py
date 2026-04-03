import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Protocol, TypeAlias, cast

from modules.generated_output_paths import mul_array_path


DTypeLike: TypeAlias = np.dtype[np.generic] | type[np.generic]


class _ApplyBatchFn(Protocol):
    def __call__(
        self,
        array: NDArray[np.float64],
        axis: int,
        batch_processor: Callable[..., np.ndarray],
        *,
        output_dtype: DTypeLike,
        max_workers: int | None = None,
        batch_args: tuple[object, ...] = (),
    ) -> np.ndarray: ...


_COMMON_SPEC = importlib.util.spec_from_file_location(
    "dsm_mt_common",
    Path(__file__).with_name("dsm_mt_common.py"),
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError("Unable to load dsm_mt_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)
apply_batch_along_axis_parallel = cast(_ApplyBatchFn, _COMMON_MODULE.apply_batch_along_axis_parallel)


def _packed_dtype(bit: int):
    if bit <= 8:
        return np.uint8
    if bit <= 16:
        return np.uint16
    if bit <= 32:
        return np.uint32
    if bit <= 64:
        return np.uint64
    raise ValueError("bit must be 64 or less")


def _mdsm_batch(
    signals: NDArray[np.float64],
    order: int,
    bit: int,
    output_dtype: DTypeLike,
) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be at least 1")
    if bit < 1:
        raise ValueError("bit must be at least 1")

    levels = (1 << bit) - 1
    integrator = np.zeros((order, signals.shape[1]), dtype=np.float64)
    output = np.zeros(signals.shape, dtype=output_dtype)

    for index in range(signals.shape[0]):
        integrator[0] += signals[index]
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = np.clip(np.rint(integrator[order - 1] * levels), 0, levels)
        if levels > 0:
            integrator -= quantized / levels
        output[index] = quantized.astype(output_dtype)

    return output
def dsm_conv_image_modulation(
    image_path: str,
    order: int = 1,
    channel_bit: int = 3,
    max_workers: int | None = None,
) -> tuple[float, int, int]:
    if channel_bit < 1:
        raise ValueError("bit must be at least 1")
    if order < 1:
        raise ValueError("order must be at least 1")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}, mode: {image.mode}")

    dim = 2
    bit = dim * channel_bit
    image_array = np.array(image, dtype=np.float64)
    maximum = float(np.max(image_array))
    image_array = image_array / maximum
    normalized_image = np.float_power(image_array, 1 / dim)
    print(f"maximum: {maximum}")

    packed_dtype = _packed_dtype(channel_bit)
    return_dtype = _packed_dtype(bit)

    print(f"Image as NumPy array shape: {normalized_image.shape}, dtype: {normalized_image.dtype}")

    w_array = apply_batch_along_axis_parallel(
        normalized_image,
        axis=0,
        batch_processor=_mdsm_batch,
        output_dtype=packed_dtype,
        max_workers=max_workers,
        batch_args=(order, channel_bit, packed_dtype),
    )
    print("Image as w array done.")

    h_array = apply_batch_along_axis_parallel(
        normalized_image,
        axis=1,
        batch_processor=_mdsm_batch,
        output_dtype=packed_dtype,
        max_workers=max_workers,
        batch_args=(order, channel_bit, packed_dtype),
    )
    print("Image as h array done.")

    rt_array = (w_array.astype(return_dtype) * h_array.astype(return_dtype)).astype(return_dtype)

    np.save(mul_array_path(image_path), rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, channel_bit
