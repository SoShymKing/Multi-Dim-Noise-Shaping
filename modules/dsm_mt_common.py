from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def resolve_worker_count(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1

    if max_workers is None:
        requested_workers = os.cpu_count() or 1
    elif type(max_workers) is int and max_workers >= 1:
        requested_workers = max_workers
    else:
        raise ValueError("max_workers must be a positive integer or None")

    return max(1, min(requested_workers, task_count))


def build_ranges(axis_length: int, worker_count: int) -> list[tuple[int, int]]:
    chunk_size = max(1, math.ceil(axis_length / worker_count))
    return [
        (start, min(axis_length, start + chunk_size))
        for start in range(0, axis_length, chunk_size)
    ]


def _batch_chunk_job(
    reshaped: NDArray[np.float64],
    start_index: int,
    end_index: int,
    batch_processor: Callable[..., np.ndarray[Any, Any]],
    batch_args: tuple[object, ...],
) -> tuple[int, int, np.ndarray[Any, Any]]:
    chunk_output = batch_processor(reshaped[:, start_index:end_index], *batch_args)
    return start_index, end_index, chunk_output


def apply_batch_along_axis_parallel(
    array: NDArray[np.float64],
    axis: int,
    batch_processor: Callable[..., np.ndarray[Any, Any]],
    *,
    output_dtype: np.dtype[Any] | type[np.generic],
    max_workers: int | None = None,
    batch_args: tuple[object, ...] = (),
) -> np.ndarray[Any, Any]:
    moved = np.moveaxis(array, axis, 0)
    reshaped = moved.reshape(moved.shape[0], -1)
    output = np.zeros(reshaped.shape, dtype=output_dtype)

    worker_count = resolve_worker_count(max_workers, reshaped.shape[1])
    index_ranges = build_ranges(reshaped.shape[1], worker_count)

    if worker_count == 1:
        output[:, :] = batch_processor(reshaped, *batch_args)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _batch_chunk_job,
                    reshaped,
                    start_index,
                    end_index,
                    batch_processor,
                    batch_args,
                )
                for start_index, end_index in index_ranges
            ]

            for future in as_completed(futures):
                start_index, end_index, chunk_output = future.result()
                output[:, start_index:end_index] = chunk_output

    restored = output.reshape(moved.shape)
    return np.moveaxis(restored, 0, axis)
