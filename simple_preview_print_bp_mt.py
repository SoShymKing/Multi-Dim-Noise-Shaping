from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


def _default_packed_input_path(image_path: str) -> str:
    return "mul_array_" + image_path + ".npy"


def _default_output_path(image_path: str) -> str:
    stem = Path(image_path).stem
    return f"output_preview_bp_mt_{stem}.bmp"


def _resolve_worker_count(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    elif type(max_workers) is int and max_workers >= 1:
        requested_workers = max_workers
    else:
        raise ValueError("max_workers must be a positive integer or None")

    return max(1, min(requested_workers, task_count))


def _build_ranges(axis_length: int, worker_count: int) -> list[tuple[int, int]]:
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


def _blur_plane(plane: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    if radius <= 0:
        return plane.astype(np.float64)

    blurred = np.empty_like(plane, dtype=np.float64)
    channel_count = int(plane.shape[2])
    for channel in range(channel_count):
        grayscale = Image.fromarray(
            np.rint(np.clip(plane[:, :, channel], 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="L",
        )
        filtered = grayscale.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred[:, :, channel] = np.asarray(filtered, dtype=np.float64) / 255.0
    return blurred


def _preview_bit_range(
    loaded_array: NDArray[np.uint64],
    start_index: int,
    end_index: int,
    maximum: float,
    bit: int,
    blur_radius: float,
) -> tuple[int, int, NDArray[np.float64]]:
    reconstructed_chunk = np.zeros_like(loaded_array, dtype=np.float64)
    for index in range(start_index, end_index):
        local_max: float = maximum / float(2 ** (bit - 1 - index))
        bit_plane: NDArray[np.float64] = ((loaded_array >> index) & 1).astype(np.float64)
        preview_plane = _blur_plane(bit_plane, blur_radius)
        reconstructed_chunk += preview_plane * local_max
    return start_index, end_index, reconstructed_chunk


def preview_dsm_print(
    image_path: str,
    maximum: float = 256,
    bit: int = 3,
    blur_radius: float = 1.5,
    packed_input_path: str | None = None,
    output_image_path: str | None = None,
    avoid_clip: bool = True,
    max_workers: int | None = None,
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

    worker_count = _resolve_worker_count(max_workers, bit)
    bit_ranges = _build_ranges(bit, worker_count)

    contributions: dict[int, NDArray[np.float64]] = {}
    if worker_count == 1:
        start_index, _, reconstructed_image = _preview_bit_range(
            loaded_array,
            0,
            bit,
            maximum,
            bit,
            blur_radius,
        )
        contributions[start_index] = reconstructed_image
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _preview_bit_range,
                    loaded_array,
                    start_index,
                    end_index,
                    maximum,
                    bit,
                    blur_radius,
                )
                for start_index, end_index in bit_ranges
            ]

            for future in as_completed(futures):
                start_index, _, reconstructed_chunk = future.result()
                contributions[start_index] = reconstructed_chunk

        reconstructed_image = np.zeros_like(loaded_array, dtype=np.float64)
        for start_index in sorted(contributions):
            reconstructed_image += contributions[start_index]

    if avoid_clip:
        current_maximum = float(np.max(reconstructed_image))
        if current_maximum > 255.0:
            reconstructed_image *= 255.0 / current_maximum

    final_image: NDArray[np.uint8] = np.clip(np.rint(reconstructed_image), 0, 255).astype(np.uint8)
    print(f"Preview image shape: {final_image.shape}, dtype: {final_image.dtype}")

    output_path = output_image_path or _default_output_path(image_path)
    Image.fromarray(final_image, mode="RGB").save(output_path)
    return output_path
