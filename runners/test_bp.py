import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import simple_dsm_image_bp
from modules import simple_preview_print_bp
from modules import simple_raw_print_bp


def main(image_path="sample.jpg", dsm_workers=12):
    start_time = time.perf_counter()

    maximum, dim, bit = simple_dsm_image_bp.dsm_conv_image_modulation(image_path)
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Total dsm time: {elapsed_time:.2f} seconds")
    simple_preview_print_bp.preview_dsm_print(image_path, maximum, dim, bit)

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
