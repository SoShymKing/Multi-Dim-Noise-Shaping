from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os

from PIL import Image
import numpy as np
from numpy.typing import NDArray


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


def _dsm_batch(signals: NDArray[np.float64], order: int) -> NDArray[np.uint8]:
    if order < 1:
        raise ValueError("order must be at least 1")

    integrator = np.zeros((order, signals.shape[1]), dtype=np.float64)
    output = np.zeros(signals.shape, dtype=np.uint8)

    for index in range(signals.shape[0]):
        integrator[0] += signals[index]
        for level in range(1, order):
            integrator[level] += integrator[level - 1]

        quantized = (integrator[order - 1] >= 0).astype(np.uint8)
        integrator -= quantized
        output[index] = quantized

    return output


def _dsm_chunk_job(
    reshaped: NDArray[np.float64],
    start_index: int,
    end_index: int,
    order: int,
) -> tuple[int, int, NDArray[np.uint8]]:
    chunk_output = _dsm_batch(reshaped[:, start_index:end_index], order)
    return start_index, end_index, chunk_output


def _apply_dsm_along_axis_parallel(
    array: NDArray[np.float64],
    axis: int,
    order: int,
    max_workers: int | None = None,
) -> NDArray[np.uint8]:
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    output = np.zeros_like(reshaped, dtype=np.uint8)

    worker_count = _resolve_worker_count(max_workers, reshaped.shape[1])
    index_ranges = _build_ranges(reshaped.shape[1], worker_count)

    if worker_count == 1:
        output[:, :] = _dsm_batch(reshaped, order)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _dsm_chunk_job,
                    reshaped,
                    start_index,
                    end_index,
                    order,
                )
                for start_index, end_index in index_ranges
            ]

            for future in as_completed(futures):
                start_index, end_index, chunk_output = future.result()
                output[:, start_index:end_index] = chunk_output

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


def dsm_conv_image_modulation(
    image_path: str,
    order: int = 1,
    bit: int = 3,
    max_workers: int | None = None,
) -> tuple[int, int, int]:
    if bit < 1:
        raise ValueError("bit must be at least 1")

    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}, mode: {image.mode}")

    dim = 2
    image_array = np.asarray(image, dtype=np.float64)
    maximum = float(np.max(image_array))

    working_array = image_array.copy()
    img: list[NDArray[np.float64]] = []
    for i in range(bit - 1):
        local_max = maximum / np.power(2, bit - 1 - i)
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
        w_array = _apply_dsm_along_axis_parallel(bit_plane, axis=0, order=order, max_workers=max_workers)
        h_array = _apply_dsm_along_axis_parallel(bit_plane, axis=1, order=order, max_workers=max_workers)
        mul_array = w_array * h_array
        rt_array |= mul_array.astype(return_dtype) << i

    np.save("mul_array_" + image_path + ".npy", rt_array)
    print(f"mul_array shape: {rt_array.shape}, dtype: {rt_array.dtype}")
    return maximum, dim, bit
