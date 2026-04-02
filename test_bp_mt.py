import time

import simple_dsm_image_bp_mt
import simple_preview_print_bp_mt


def main(image_path: str = "sample.jpg", dsm_workers: int = 12, preview_workers: int = 12) -> None:
    start_time = time.perf_counter()

    maximum, _dim, bit = simple_dsm_image_bp_mt.dsm_conv_image_modulation(
        image_path,
        max_workers=dsm_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total dsm time: {elapsed_time:.2f} seconds")

    _ = simple_preview_print_bp_mt.preview_dsm_print(
        image_path,
        maximum=maximum,
        bit=bit,
        max_workers=preview_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
