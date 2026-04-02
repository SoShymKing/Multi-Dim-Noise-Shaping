from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

from PIL import Image
import numpy as np
from numpy.typing import NDArray


def _resolve_worker_count(max_workers, task_count):
    if task_count <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    elif isinstance(max_workers, int) and max_workers >= 1:
        requested_workers = max_workers
    else:
        raise ValueError("max_workers must be a positive integer or None")

    return max(1, min(requested_workers, task_count))


def _build_ranges(axis_length, worker_count):
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


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


def _mdsm_batch(signals: NDArray[np.float64], order: int, bit: int, output_dtype) -> NDArray:
    if order < 1:
        raise ValueError("order must be at least 1")
    if bit < 1:
        raise ValueError("bit must be at least 1")

    levels = (1 << bit) - 1
    integrator = np.zeros((order, signals.shape[1]), dtype=np.float64)
    output = np.zeros(signals.shape, dtype=output_dtype)

    for index in range(signals.shape[0]):
        integrator[0] += signals[index]
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = np.clip(np.rint(integrator[order - 1] * levels), 0, levels)
        if levels > 0:
            integrator -= quantized / levels
        output[index] = quantized.astype(output_dtype)

    return output


def _mdsm_chunk_job(reshaped, start_index, end_index, order, bit, output_dtype):
    chunk_output = _mdsm_batch(reshaped[:, start_index:end_index], order, bit, output_dtype)
    return start_index, end_index, chunk_output


def _apply_mdsm_along_axis_parallel(array: NDArray[np.float64], axis: int, order: int, bit: int, output_dtype, max_workers=None) -> NDArray:
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    output = np.zeros_like(reshaped, dtype=output_dtype)

    worker_count = _resolve_worker_count(max_workers, reshaped.shape[1])
    index_ranges = _build_ranges(reshaped.shape[1], worker_count)

    if worker_count == 1:
        output[:, :] = _mdsm_batch(reshaped, order, bit, output_dtype)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _mdsm_chunk_job,
                    reshaped,
                    start_index,
                    end_index,
                    order,
                    bit,
                    output_dtype,
                )
                for start_index, end_index in index_ranges
            ]

            for future in as_completed(futures):
                start_index, end_index, chunk_output = future.result()
                output[:, start_index:end_index] = chunk_output

    restored = output.reshape(moved.shape)
    return np.moveaxis(restored, 0, axis)


def dsm_conv_image_modulation(image_path, order=1, channel_bit=3, max_workers=None):
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

    w_array = _apply_mdsm_along_axis_parallel(
        normalized_image,
        axis=0,
        order=order,
        bit=channel_bit,
        output_dtype=packed_dtype,
        max_workers=max_workers,
    )
    print("Image as w array done.")

    h_array = _apply_mdsm_along_axis_parallel(
        normalized_image,
        axis=1,
        order=order,
        bit=channel_bit,
        output_dtype=packed_dtype,
        max_workers=max_workers,
    )
    print("Image as h array done.")

    rt_array = (w_array.astype(return_dtype) * h_array.astype(return_dtype)).astype(return_dtype)

    np.save("mul_array_" + image_path + ".npy", rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, channel_bit
