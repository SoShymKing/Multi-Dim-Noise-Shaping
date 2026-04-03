from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

from PIL import Image
import numpy as np

from modules.generated_output_paths import default_output_bmp_path, mul_array_path


def _resolve_worker_count(max_workers, axis_length):
    if axis_length <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    else:
        requested_workers = max_workers

    return max(1, min(requested_workers, axis_length))


def _build_ranges(axis_length, worker_count):
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


def _reconstruct_chunk(loaded_array, start_index, end_index, scale_value):
    chunk = loaded_array[start_index:end_index].astype(np.float64) * scale_value
    np.clip(chunk, 0, 255, out=chunk)
    return start_index, end_index, chunk.astype(np.uint8)


def raw_dsm_print(image_path, median=127, maximum=255, dim=2, max_workers=None):
    loaded_array = np.load(mul_array_path(image_path)).astype(np.uint8)
    print(f"Image as NumPy array shape: {loaded_array.shape}")

    new_width = loaded_array.shape[0]
    new_height = loaded_array.shape[1]
    reconstructed_image = np.zeros((new_width, new_height, 3), dtype=np.uint8)
    scale_value = min(np.power(maximum, dim), 255)

    worker_count = _resolve_worker_count(max_workers, new_width)
    index_ranges = _build_ranges(new_width, worker_count)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _reconstruct_chunk,
                loaded_array,
                start_index,
                end_index,
                scale_value,
            )
            for start_index, end_index in index_ranges
        ]

        for future in as_completed(futures):
            start_index, end_index, chunk = future.result()
            reconstructed_image[start_index:end_index] = chunk

    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode='RGB')
    image.save(default_output_bmp_path())
    # image.save("raw_image_mul_" + image_path + ".bmp")
