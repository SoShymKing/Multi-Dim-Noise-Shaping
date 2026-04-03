from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

import numpy as np

from modules.generated_output_paths import bp_raw_output_path, mul_array_path


def _resolve_worker_count(max_workers, task_count: int) -> int:
    if task_count <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    elif isinstance(max_workers, int) and max_workers >= 1:
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


def _reconstruct_chunk(
    loaded_array,
    start_index: int,
    end_index: int,
    maximum: float,
    bit: int,
):
    chunk_input = loaded_array[start_index:end_index]
    chunk_output = np.zeros_like(chunk_input, dtype=np.float64)

    for i in range(bit):
        local_max = maximum / np.power(2, bit - 1 - i)
        chunk_output += ((chunk_input >> i) & 1) * local_max

    chunk_output = np.rint(chunk_output).astype(np.uint32)
    return start_index, end_index, chunk_output


def raw_dsm_print(image_path, maximum=255, dim=2, bit=3, max_workers=None):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load(mul_array_path(image_path))
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")
    if not np.issubdtype(loaded_array.dtype, np.integer):
        loaded_array = loaded_array.astype(np.uint64)
    print(f"Image as NumPy array shape: {loaded_array.shape}")

    reconstructed_image = np.zeros_like(loaded_array, dtype=np.uint32)
    worker_count = _resolve_worker_count(max_workers, loaded_array.shape[0])
    index_ranges = _build_ranges(loaded_array.shape[0], worker_count)

    if worker_count == 1:
        _, _, reconstructed_image = _reconstruct_chunk(loaded_array, 0, loaded_array.shape[0], maximum, bit)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _reconstruct_chunk,
                    loaded_array,
                    start_index,
                    end_index,
                    maximum,
                    bit,
                )
                for start_index, end_index in index_ranges
            ]

            for future in as_completed(futures):
                start_index, end_index, chunk = future.result()
                reconstructed_image[start_index:end_index] = chunk

    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    np.save(bp_raw_output_path(), reconstructed_image)
