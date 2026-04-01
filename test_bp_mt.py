import time

import simple_dsm_image_bp_mt
import simple_raw_print_bp_mt


def main(image_path="sample.jpg", order=1, bit=3, dsm_workers=12, raw_workers=12):
    start_time = time.perf_counter()

    maximum, dim, bit = simple_dsm_image_bp_mt.dsm_conv_image_modulation(
        image_path,
        order=order,
        bit=bit,
        max_workers=dsm_workers,
    )
    simple_raw_print_bp_mt.raw_dsm_print(
        image_path,
        maximum=maximum,
        dim=dim,
        bit=bit,
        max_workers=raw_workers,
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
