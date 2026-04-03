from PIL import Image
import numpy as np
from numpy.typing import NDArray

from modules.generated_output_paths import clean_bp_mul_array_path


def _dsm_1d(signal: NDArray[np.float64], order: int) -> NDArray[np.uint8]:
    if order < 1:
        raise ValueError("order must be at least 1")

    integrator = np.zeros(order, dtype=np.float64)
    output = np.zeros(signal.shape[0], dtype=np.uint8)

    for index, value in enumerate(signal):
        integrator[0] += value
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = 1 if integrator[order - 1] >= 0 else 0
        integrator -= quantized
        output[index] = quantized

    return output


def _apply_dsm_along_axis(array: NDArray[np.float64], axis: int, order: int) -> NDArray[np.uint8]:
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    output = np.zeros_like(reshaped, dtype=np.uint8)

    for index in range(reshaped.shape[1]):
        output[:, index] = _dsm_1d(reshaped[:, index], order)

    restored = output.reshape(moved.shape)
    return np.moveaxis(restored, 0, axis)


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


def _default_packed_path(image_path: str) -> str:
    return clean_bp_mul_array_path(image_path)


def dsm_conv_image_modulation(
    image_path: str,
    order: int = 1,
    bit: int = 3,
    packed_output_path: str | None = None,
):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}, mode: {image.mode}")

    dim = 2
    image_array = np.asarray(image, dtype=np.float64)
    maximum = float(np.max(image_array))

    working_array = image_array.copy()
    img: list[NDArray[np.float64]] = []
    for i in range(bit-1):
        local_max =  np.float_power(maximum,(i+1)/bit)
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
        w_array = _apply_dsm_along_axis(bit_plane, axis=0, order=order)
        h_array = _apply_dsm_along_axis(bit_plane, axis=1, order=order)
        mul_array = w_array * h_array
        rt_array |= mul_array.astype(return_dtype) << i

    output_path = packed_output_path or _default_packed_path(image_path)
    np.save(output_path, rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, bit
