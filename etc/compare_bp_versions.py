from argparse import ArgumentParser
from pathlib import Path
import shutil
from typing import Any

from PIL import Image

import lpf_compare
import simple_dsm_image_bp
import simple_raw_print_bp
import simple_dsm_image_bp_clean
import simple_raw_print_bp_clean


def _prepare_input_image(image_path: str, max_edge: int) -> str:
    source = Path(image_path)
    prepared_path = Path(f"bp_compare_input_{source.stem}.bmp")

    with Image.open(source) as image:
        rgb_image = image.convert("RGB")
        if max(rgb_image.size) > max_edge:
            scale = max_edge / float(max(rgb_image.size))
            resized = rgb_image.resize(
                (max(1, round(rgb_image.size[0] * scale)), max(1, round(rgb_image.size[1] * scale))),
                Image.Resampling.LANCZOS,
            )
        else:
            resized = rgb_image.copy()
        resized.save(prepared_path)

    return str(prepared_path)


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(f"[{label}]")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Max abs error: {metrics['max_abs_error']:.6f}")
    print(f"  Mean signed error: {metrics['mean_signed_error']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.6f}")
    print(f"  P50 abs error: {metrics['p50_abs_error']:.6f}")
    print(f"  P95 abs error: {metrics['p95_abs_error']:.6f}")
    print(f"  P99 abs error: {metrics['p99_abs_error']:.6f}")
    for channel_name, channel_metrics in metrics["per_channel"].items():
        print(
            f"  {channel_name}: MAE={channel_metrics['mae']:.6f}, "
            f"MSE={channel_metrics['mse']:.6f}, "
            f"RMSE={channel_metrics['rmse']:.6f}, "
            f"MaxAbs={channel_metrics['max_abs_error']:.6f}, "
            f"MeanSigned={channel_metrics['mean_signed_error']:.6f}"
        )


def main() -> None:
    parser = ArgumentParser(description="Compare current BP output with a new clean BP implementation.")
    _ = parser.add_argument("--image", default="sample.jpg", help="Source image path")
    _ = parser.add_argument("--order", type=int, default=1, help="DSM order")
    _ = parser.add_argument("--bit", type=int, default=3, help="Bit depth")
    _ = parser.add_argument("--max-edge", type=int, default=192, help="Resize long edge before evaluation")
    _ = parser.add_argument("--radius", type=float, default=1.5, help="LPF radius for comparison")
    _ = parser.add_argument("--save-lpf", action="store_true", help="Save LPF images and diffs")
    args = parser.parse_args()

    prepared_input = _prepare_input_image(args.image, args.max_edge)
    prepared_stem = Path(prepared_input).stem
    current_output = f"output_current_bp_{prepared_stem}.bmp"
    clean_output = f"output_clean_bp_{prepared_stem}.bmp"

    current_maximum, current_dim, current_bit = simple_dsm_image_bp.dsm_conv_image_modulation(
        prepared_input,
        order=args.order,
        bit=args.bit,
    )
    simple_raw_print_bp.raw_dsm_print(prepared_input, int(round(current_maximum)), current_dim, current_bit)
    _ = shutil.copyfile("output.bmp", current_output)

    clean_maximum, clean_dim, clean_bit = simple_dsm_image_bp_clean.dsm_conv_image_modulation(
        prepared_input,
        order=args.order,
        bit=args.bit,
    )
    simple_raw_print_bp_clean.raw_dsm_print(
        prepared_input,
        maximum=clean_maximum,
        dim=clean_dim,
        bit=clean_bit,
        output_image_path=clean_output,
    )

    current_vs_reference = lpf_compare.compare_images(
        current_output,
        prepared_input,
        radius=args.radius,
        save_outputs=args.save_lpf,
    )
    clean_vs_reference = lpf_compare.compare_images(
        clean_output,
        prepared_input,
        radius=args.radius,
        save_outputs=args.save_lpf,
    )
    clean_vs_current = lpf_compare.compare_images(
        clean_output,
        current_output,
        radius=args.radius,
        save_outputs=args.save_lpf,
    )

    print(f"Prepared input: {prepared_input}")
    print(f"Current BP output: {current_output}")
    print(f"Clean BP output: {clean_output}")
    print(f"LPF radius: {args.radius}")
    _print_metrics("current bp vs reference", current_vs_reference)
    _print_metrics("clean bp vs reference", clean_vs_reference)
    _print_metrics("clean bp vs current bp", clean_vs_current)


if __name__ == "__main__":
    main()
