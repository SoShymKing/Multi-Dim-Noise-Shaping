from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

from PIL import Image
import numpy as np


def _resolve_worker_count(max_workers, axis_length):
    if axis_length <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    elif isinstance(max_workers, int) and max_workers >= 1:
        requested_workers = max_workers
    else:
        raise ValueError("max_workers must be a positive integer or None")

    return max(1, min(requested_workers, axis_length))


def _build_ranges(axis_length, worker_count):
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


def _reconstruct_chunk(loaded_array, start_index, end_index, levels, maximum, dim):
    if levels > 0 and maximum > 0:
        sqrt_reconstructed = loaded_array[start_index:end_index].astype(np.float64) / levels * maximum
    else:
        sqrt_reconstructed = np.zeros_like(loaded_array[start_index:end_index], dtype=np.float64)

    reconstructed_chunk = np.power(sqrt_reconstructed, dim)
    np.clip(reconstructed_chunk, 0, 255, out=reconstructed_chunk)
    return start_index, end_index, reconstructed_chunk.astype(np.uint8)


def raw_dsm_print(image_path, maximum=255, dim=2, bit=3, max_workers=None):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load("mul_array_" + image_path + ".npy")
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")

    print(f"Image as NumPy array shape: {loaded_array.shape}")

    image_height = loaded_array.shape[0]
    reconstructed_image = np.zeros_like(loaded_array, dtype=np.uint8)
    levels = (1 << bit) - 1

    worker_count = _resolve_worker_count(max_workers, image_height)
    index_ranges = _build_ranges(image_height, worker_count)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _reconstruct_chunk,
                loaded_array,
                start_index,
                end_index,
                levels,
                maximum,
                dim,
            )
            for start_index, end_index in index_ranges
        ]

        for future in as_completed(futures):
            start_index, end_index, reconstructed_chunk = future.result()
            reconstructed_image[start_index:end_index] = reconstructed_chunk

    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode="RGB")
    image.save("output.bmp")
