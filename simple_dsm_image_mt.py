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
    else:
        requested_workers = max_workers

    return max(1, min(requested_workers, axis_length))


def _build_ranges(axis_length, worker_count):
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


def _horizontal_pass_chunk(sqrt_image_array, channel, start_column, end_column, convert_w, order):
    w_chunk = np.zeros((convert_w, end_column - start_column), dtype=np.int8)

    for local_column, column in enumerate(range(start_column, end_column)):
        integrator = [0.0] * order
        for row in range(convert_w):
            integrator[0] += sqrt_image_array[row, column, channel]
            
            for level in range(1, order):
                integrator[level] += integrator[level - 1]
                
            y = 1 if integrator[order-1] >= 0 else 0
            
            for level in range(0, order):
                integrator[level] -= y
                
            w_chunk[row, local_column] = y

    return channel, start_column, end_column, w_chunk


def _vertical_pass_chunk(sqrt_image_array, channel, start_row, end_row, convert_h, order):
    h_chunk = np.zeros((end_row - start_row, convert_h), dtype=np.int8)

    for local_row, row in enumerate(range(start_row, end_row)):
        integrator = [0.0] * order
        for column in range(convert_h):
            integrator[0] += sqrt_image_array[row, column, channel]
            
            for level in range(1, order):
                integrator[level] += integrator[level - 1]
            
            y = 1 if integrator[order-1] >= 0 else 0

            for level in range(0, order):
                integrator[level] -= y

            h_chunk[local_row, column] = y

    return channel, start_row, end_row, h_chunk


def _compute_horizontal_pass(sqrt_image_array, image_width, convert_w, order, max_workers):
    worker_count = _resolve_worker_count(max_workers, image_width)
    column_ranges = _build_ranges(image_width, worker_count)
    w_array = np.zeros((convert_w, image_width, 3), dtype=np.int8)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _horizontal_pass_chunk,
                sqrt_image_array,
                channel,
                start_column,
                end_column,
                convert_w,
                order,
            )
            for channel in range(3)
            for start_column, end_column in column_ranges
        ]

        for future in as_completed(futures):
            channel, start_column, end_column, w_chunk = future.result()
            w_array[:, start_column:end_column, channel] = w_chunk

    return w_array


def _compute_vertical_pass(sqrt_image_array, image_height, convert_h, order, max_workers):
    worker_count = _resolve_worker_count(max_workers, image_height)
    row_ranges = _build_ranges(image_height, worker_count)
    h_array = np.zeros((image_height, convert_h, 3), dtype=np.int8)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _vertical_pass_chunk,
                sqrt_image_array,
                channel,
                start_row,
                end_row,
                convert_h,
                order,
            )
            for channel in range(3)
            for start_row, end_row in row_ranges
        ]

        for future in as_completed(futures):
            channel, start_row, end_row, h_chunk = future.result()
            h_array[start_row:end_row, :, channel] = h_chunk

    return h_array


def dsm_conv_image_modulation(image_path, order=1, max_workers=None):
    image = Image.open(image_path)
    print(f"Image size: {image.size}, mode: {image.mode}")

    convert_h = image.size[0]
    convert_w = image.size[1]
    dim = 2

    image_array = np.array(image, dtype=np.float64)
    image_array = np.float_power(image_array, 1 / dim)
    maximum = np.max(image_array)
    median = (np.max(image_array) + np.min(image_array)) / 2
    image_array = image_array / maximum
    sqrt_image_array = image_array

    print(f"maximum: {maximum}")
    print(f"Image as NumPy array shape: {sqrt_image_array.shape}, dtype: {sqrt_image_array.dtype}")

    w_array = _compute_horizontal_pass(
        sqrt_image_array=sqrt_image_array,
        image_width=image.size[0],
        convert_w=convert_w,
        order=order,
        max_workers=max_workers,
    )
    print(f"w_array shape: {w_array.shape}, dtype: {w_array.dtype}")

    h_array = _compute_vertical_pass(
        sqrt_image_array=sqrt_image_array,
        image_height=image.size[1],
        convert_h=convert_h,
        order=order,
        max_workers=max_workers,
    )
    print(f"h_array shape: {h_array.shape}, dtype: {h_array.dtype}")

    row_indices = np.arange(convert_w)
    column_indices = np.arange(convert_h)
    mul_array = (
        w_array[:, column_indices, :] * h_array[row_indices, :, :]
    ).astype(np.int8)

    np.save("mul_array_" + image_path + ".npy", mul_array)
    print(f"mul_array shape: {mul_array.shape}, dtype: {mul_array.dtype}")
    return median, maximum, dim
