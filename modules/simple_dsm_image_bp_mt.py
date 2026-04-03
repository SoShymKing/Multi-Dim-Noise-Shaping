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


def _dsm_batch(signals: NDArray[np.float64], order: int) -> NDArray[np.uint8]:
    if order < 1:
        raise ValueError("order must be at least 1")

    integrator = np.zeros((order, signals.shape[1]), dtype=np.float64)
    output = np.zeros(signals.shape, dtype=np.uint8)

    for index in range(signals.shape[0]):
        integrator[0] += signals[index]
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = (integrator[order - 1] >= 0).astype(np.uint8)
        integrator -= quantized
        output[index] = quantized

    return output
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


def dsm_conv_image_modulation(
    image_path: str,
    order: int = 1,
    bit: int = 3,
    max_workers: int | None = None,
) -> tuple[float, int, int]:
    if bit < 1:
        raise ValueError("bit must be at least 1")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}, mode: {image.mode}")

    dim = 2
    image_array = np.asarray(image, dtype=np.float64)
    maximum = float(np.max(image_array))

    working_array = image_array.copy()
    img: list[NDArray[np.float64]] = []
    for i in range(bit - 1):
        local_max = maximum / np.power(2, bit - 1 - i)
        current_plane = working_array % local_max
        working_array -= current_plane
        current_plane /= local_max
        img.append(np.float_power(current_plane, 1 / dim))
    img.append(np.float_power(working_array / maximum, 1 / dim))

    print(f"maximum: {maximum}")
    return_dtype = _packed_dtype(bit)
    rt_array = np.zeros_like(image_array, dtype=return_dtype)

    print(f"Image as NumPy array shape: {image_array.shape}, dtype: {image_array.dtype}")

    for i in range(bit):
        bit_plane = img[i].astype(np.float64)
        w_array = apply_batch_along_axis_parallel(
            bit_plane,
            axis=0,
            batch_processor=_dsm_batch,
            output_dtype=np.uint8,
            max_workers=max_workers,
            batch_args=(order,),
        )
        h_array = apply_batch_along_axis_parallel(
            bit_plane,
            axis=1,
            batch_processor=_dsm_batch,
            output_dtype=np.uint8,
            max_workers=max_workers,
            batch_args=(order,),
        )
        mul_array = w_array * h_array
        rt_array |= mul_array.astype(return_dtype) << i

    np.save("mul_array_" + image_path + ".npy", rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, bit
