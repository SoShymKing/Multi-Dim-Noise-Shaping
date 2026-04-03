import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import simple_dsm_image
from modules import simple_raw_print


def main(image_path="sample.jpg", dsm_workers=12):
    start_time = time.perf_counter()

    median, maximum, dim = simple_dsm_image.dsm_conv_image_modulation(image_path)
    simple_raw_print.raw_dsm_print(image_path, median, maximum, dim)

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
