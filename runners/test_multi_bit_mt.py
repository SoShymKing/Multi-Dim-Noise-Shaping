import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import simple_dsm_image_multi_bit_mt
from modules import simple_raw_print_multi_bit_mt


def main(image_path="sample.jpg", dsm_workers=12, raw_workers=12):
    start_time = time.perf_counter()

    maximum, dim, channel_bit = simple_dsm_image_multi_bit_mt.dsm_conv_image_modulation(
        image_path,
        max_workers=dsm_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total dsm time: {elapsed_time:.2f} seconds")

    simple_raw_print_multi_bit_mt.raw_dsm_print(
        image_path,
        cast(int, maximum),
        dim,
        channel_bit,
        max_workers=raw_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
