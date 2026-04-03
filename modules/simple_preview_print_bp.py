from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def _default_packed_input_path(image_path: str) -> str:
    return "mul_array_" + image_path + ".npy"


def _default_output_path(image_path: str) -> str:
    stem = Path(image_path).stem
    return f"output_preview_bp_{stem}.bmp"


def _blur_image(image: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    if radius <= 0:
        return image.astype(np.float64)

    blurred: NDArray[np.float64] = np.empty_like(image, dtype=np.float64)
    channel_count = cast(int, image.shape[2])
    sigma = float(radius)
    for channel in range(channel_count):
        blurred[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=sigma, mode="nearest")
    return blurred


def _matched_weight_sum(
    loaded_array: NDArray[np.uint64],
    maximum: float,
    bit: int,
) -> NDArray[np.float64]:
    reconstructed_image: NDArray[np.float64] = np.zeros_like(loaded_array, dtype=np.float64)
    for index in range(bit):
        exponent = int(bit - 1 - index)
        local_max: float = maximum / float(2 ** exponent)
        bit_plane: NDArray[np.float64] = ((loaded_array >> index) & 1).astype(np.float64)
        reconstructed_image += bit_plane * local_max
    return reconstructed_image


def preview_dsm_print(
    image_path: str,
    maximum: float = 256,
    bit: int = 3,
    blur_radius: float = 4.0,
    packed_input_path: str | None = None,
    output_image_path: str | None = None,
) -> str:
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array: NDArray[np.uint64] = np.asarray(
        np.load(packed_input_path or _default_packed_input_path(image_path)),
        dtype=np.uint64,
    )
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")

    print(f"Preview input shape: {loaded_array.shape}, dtype: {loaded_array.dtype}")

    reconstructed_image: NDArray[np.float64] = _matched_weight_sum(loaded_array, maximum, bit)

    preview_image = _blur_image(reconstructed_image, blur_radius)
    current_maximum = float(np.max(preview_image))
    if current_maximum > 0.0:
        preview_image *= 255.0 / current_maximum
    scaled_maximum = float(np.max(preview_image))

    final_image: NDArray[np.uint8] = np.clip(np.rint(preview_image), 0, 255).astype(np.uint8)
    print(f"Preview image shape: {final_image.shape}, dtype: {final_image.dtype}")
    print(f"Preview blur radius used: {blur_radius:.3f}")
    print(f"Preview max before scaling: {current_maximum:.3f}")
    print(f"Preview max after scaling: {scaled_maximum:.3f}")

    output_path = output_image_path or _default_output_path(image_path)
    Image.fromarray(final_image, mode="RGB").save(output_path)
    return output_path
