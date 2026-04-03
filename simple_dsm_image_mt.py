import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Protocol, TypeAlias, cast


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


def _dsm_batch(signals: NDArray[np.float64], order: int) -> NDArray[np.int8]:
    if order < 1:
        raise ValueError("order must be at least 1")

    integrator = np.zeros((order, signals.shape[1]), dtype=np.float64)
    output = np.zeros(signals.shape, dtype=np.int8)

    for index in range(signals.shape[0]):
        integrator[0] += signals[index]
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = (integrator[order - 1] >= 0).astype(np.int8)
        integrator -= quantized
        output[index] = quantized

    return output
def dsm_conv_image_modulation(
    image_path: str,
    order: int = 1,
    max_workers: int | None = None,
) -> tuple[float, float, int]:
    image = Image.open(image_path)
    print(f"Image size: {image.size}, mode: {image.mode}")

    image_width = image.size[0]
    image_height = image.size[1]
    dim = 2

    image_array = np.asarray(image, dtype=np.float64)
    image_array = np.float_power(image_array, 1 / dim)
    maximum = float(np.max(image_array))
    median = float((np.max(image_array) + np.min(image_array)) / 2)
    sqrt_image_array = image_array / maximum

    print(f"maximum: {maximum}")
    print(f"Image as NumPy array shape: {sqrt_image_array.shape}, dtype: {sqrt_image_array.dtype}")

    w_array = apply_batch_along_axis_parallel(
        sqrt_image_array,
        axis=0,
        batch_processor=_dsm_batch,
        output_dtype=np.int8,
        max_workers=max_workers,
        batch_args=(order,),
    )
    print(f"w_array shape: {w_array.shape}, dtype: {w_array.dtype}")

    h_array = apply_batch_along_axis_parallel(
        sqrt_image_array,
        axis=1,
        batch_processor=_dsm_batch,
        output_dtype=np.int8,
        max_workers=max_workers,
        batch_args=(order,),
    )
    print(f"h_array shape: {h_array.shape}, dtype: {h_array.dtype}")

    row_indices = np.arange(image_height)
    column_indices = np.arange(image_width)
    mul_array = (w_array[:, column_indices, :] * h_array[row_indices, :, :]).astype(np.int8)

    np.save("mul_array_" + image_path + ".npy", mul_array)
    print(f"mul_array shape: {mul_array.shape}, dtype: {mul_array.dtype}")
    return median, maximum, dim
