from PIL import Image
import numpy as np
from numpy.typing import NDArray

from modules.generated_output_paths import mul_array_path

def dsm_1d_multibit(signal, order, bit, output_dtype):
    if order < 1:
        raise ValueError("order must be at least 1")
    if bit < 1:
        raise ValueError("bit must be at least 1")

    L = (1 << bit) - 1
    integrator = np.zeros(order, dtype=np.float64)
    output = np.zeros(signal.shape[0], dtype=output_dtype)
    for i, x in enumerate(signal):
        integrator[0] += x
        for k in range(1, order):
            integrator[k] += integrator[k - 1]
        y = np.clip(np.rint(integrator[order - 1] * L), 0, L)
        if L > 0:
            integrator -= y / L
        output[i] = output_dtype(y)
    return output




def _apply_mdsm_along_axis(
    array: NDArray[np.float64],
    axis: int,
    order: int,
    bit: int,
    output_dtype,
) -> NDArray[np.generic]:
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    output = np.zeros_like(reshaped, dtype=output_dtype)

    for index in range(reshaped.shape[1]):
        output[:, index] = dsm_1d_multibit(reshaped[:, index], order, bit, output_dtype)

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


def dsm_conv_image_modulation(image_path, order=1, channel_bit=3):
    if not isinstance(channel_bit, int):
        raise ValueError("bit must be an integer")
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

    w_array = _apply_mdsm_along_axis(normalized_image, axis=0, order=order, bit=channel_bit, output_dtype=packed_dtype)
    print(f"Image as w array done.")
    h_array = _apply_mdsm_along_axis(normalized_image, axis=1, order=order, bit=channel_bit, output_dtype=packed_dtype)
    print(f"Image as h array done.")

    combined_array = w_array.astype(return_dtype) * h_array.astype(return_dtype)
    
    rt_array = np.zeros_like(normalized_image, dtype=return_dtype)
    rt_array[:] = combined_array.astype(return_dtype)

    np.save(mul_array_path(image_path), rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, channel_bit
