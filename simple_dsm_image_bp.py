from PIL import Image
import numpy as np
from numpy.typing import NDArray


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


def dsm_conv_image_modulation(image_path, order=1, bit=3):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}, mode: {image.mode}")

    dim = 2
    image_array = np.array(image, dtype=np.float64)
    image_array = np.float_power(image_array, 1 / dim)
    maximum = float(np.max(image_array))
    print(f"maximum: {maximum}")

    if maximum > 0:
        normalized_image = image_array / maximum
    else:
        normalized_image = np.zeros_like(image_array)

    levels = (1 << bit) - 1
    packed_dtype = _packed_dtype(bit)
    quantized_levels = np.rint(normalized_image * levels).astype(packed_dtype)
    rt_array = np.zeros_like(quantized_levels, dtype=packed_dtype)

    print(f"Image as NumPy array shape: {normalized_image.shape}, dtype: {normalized_image.dtype}")

    for level in range(bit):
        bit_plane = ((quantized_levels >> level) & 1).astype(np.float64)
        w_array = _apply_dsm_along_axis(bit_plane, axis=0, order=order)
        h_array = _apply_dsm_along_axis(bit_plane, axis=1, order=order)
        mul_array = w_array * h_array
        rt_array |= mul_array.astype(packed_dtype) << level

    np.save("mul_array_" + image_path + ".npy", rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, bit
