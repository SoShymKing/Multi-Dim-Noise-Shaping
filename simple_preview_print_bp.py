from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageFilter
from numpy.typing import NDArray


def _default_packed_input_path(image_path: str) -> str:
    return "mul_array_" + image_path + ".npy"


def _default_output_path(image_path: str) -> str:
    stem = Path(image_path).stem
    return f"output_preview_bp_{stem}.bmp"


def _blur_plane(plane: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    if radius <= 0:
        return plane.astype(np.float64)

    blurred: NDArray[np.float64] = np.empty_like(plane, dtype=np.float64)
    channel_count = cast(int, plane.shape[2])
    for channel in range(channel_count):
        grayscale = Image.fromarray(
            np.rint(np.clip(plane[:, :, channel], 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="L",
        )
        filtered = grayscale.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred[:, :, channel] = np.asarray(filtered, dtype=np.float64) / 255.0
    return blurred


def preview_dsm_print(
    image_path: str,
    maximum: float = 256,
    bit: int = 3,
    blur_radius: float = 1.5,
    packed_input_path: str | None = None,
    output_image_path: str | None = None,
    avoid_clip: bool = True,
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

    reconstructed_image: NDArray[np.float64] = np.zeros_like(loaded_array, dtype=np.float64)
    for index in range(bit):
        local_max = maximum / (2 ** (bit - 1 - index))
        bit_plane: NDArray[np.float64] = ((loaded_array >> index) & 1).astype(np.float64)
        preview_plane = _blur_plane(bit_plane, blur_radius)
        reconstructed_image += preview_plane * local_max

    if avoid_clip:
        current_maximum = float(np.max(reconstructed_image))
        if current_maximum > 255.0:
            reconstructed_image *= 255.0 / current_maximum

    final_image: NDArray[np.uint8] = np.clip(np.rint(reconstructed_image), 0, 255).astype(np.uint8)
    print(f"Preview image shape: {final_image.shape}, dtype: {final_image.dtype}")

    output_path = output_image_path or _default_output_path(image_path)
    Image.fromarray(final_image, mode="RGB").save(output_path)
    return output_path
