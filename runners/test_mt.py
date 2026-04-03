import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import simple_dsm_image_mt
from modules import simple_raw_print_mt


def main(image_path="sample.jpg", dsm_workers=12, raw_workers=12):
    start_time = time.perf_counter()

    median, maximum, dim = simple_dsm_image_mt.dsm_conv_image_modulation(
        image_path,
        max_workers=dsm_workers,
    )
    simple_raw_print_mt.raw_dsm_print(
        image_path,
        cast(int, median),
        cast(int, maximum),
        dim,
        max_workers=raw_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
