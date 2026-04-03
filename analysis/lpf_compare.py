from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageFilter
import numpy as np


def _load_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32) / 255.0


def _apply_lpf_rgb(image_array: np.ndarray, radius: float) -> np.ndarray:
    image_uint8 = np.clip(np.rint(image_array * 255.0), 0, 255).astype(np.uint8)
    filtered_image = Image.fromarray(image_uint8, mode="RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    filtered_array = np.asarray(filtered_image, dtype=np.float32) / 255.0

    if not np.isfinite(filtered_array).all():
        raise ValueError("LPF output contains non-finite values")

    return filtered_array


def _metrics(reference: np.ndarray, target: np.ndarray) -> dict:
    difference = reference - target
    absolute_difference = np.abs(difference)
    mse = float(np.mean(np.square(difference)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_difference))
    max_abs_error = float(np.max(absolute_difference))
    mean_signed_error = float(np.mean(difference))
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))

    per_channel = {}
    for channel_index, channel_name in enumerate(("R", "G", "B")):
        channel_difference = difference[:, :, channel_index]
        channel_abs_difference = absolute_difference[:, :, channel_index]
        channel_mse = float(np.mean(np.square(channel_difference)))
        per_channel[channel_name] = {
            "mae": float(np.mean(channel_abs_difference)),
            "mse": channel_mse,
            "rmse": float(np.sqrt(channel_mse)),
            "max_abs_error": float(np.max(channel_abs_difference)),
            "mean_signed_error": float(np.mean(channel_difference)),
        }

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "mean_signed_error": mean_signed_error,
        "psnr": psnr,
        "p50_abs_error": float(np.percentile(absolute_difference, 50)),
        "p95_abs_error": float(np.percentile(absolute_difference, 95)),
        "p99_abs_error": float(np.percentile(absolute_difference, 99)),
        "per_channel": per_channel,
    }


def _save_rgb(image_array: np.ndarray, output_path: str) -> None:
    image_uint8 = np.clip(np.rint(image_array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(output_path)


def compare_images(image_a_path: str, image_b_path: str, radius: float = 1.5, save_outputs: bool = False) -> dict:
    image_a = _load_rgb(image_a_path)
    image_b = _load_rgb(image_b_path)

    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Image shapes differ: {image_a_path}={image_a.shape}, {image_b_path}={image_b.shape}"
        )

    filtered_a = _apply_lpf_rgb(image_a, radius)
    filtered_b = _apply_lpf_rgb(image_b, radius)
    metrics = _metrics(filtered_a, filtered_b)

    if save_outputs:
        stem_a = Path(image_a_path).stem
        stem_b = Path(image_b_path).stem
        _save_rgb(filtered_a, f"{stem_a}_lpf.bmp")
        _save_rgb(filtered_b, f"{stem_b}_lpf.bmp")
        _save_rgb(np.abs(filtered_a - filtered_b), f"lpf_diff_{stem_a}_vs_{stem_b}.bmp")

    return metrics


def main() -> None:
    parser = ArgumentParser(description="Apply LPF to two images and compare them.")
    parser.add_argument("--img-a", default="image_mul_sample.bmp", help="First image path")
    parser.add_argument("--img-b", default="sample.jpg", help="Second image path")
    parser.add_argument("--radius", type=float, default=1.5, help="Gaussian blur radius")
    parser.add_argument("--save", action="store_true", help="Save LPF outputs and abs-diff image")
    args = parser.parse_args()

    metrics = compare_images(args.img_a, args.img_b, radius=args.radius, save_outputs=args.save)
    print(f"Compared: {args.img_a} vs {args.img_b}")
    print(f"LPF radius: {args.radius}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Max abs error: {metrics['max_abs_error']:.6f}")
    print(f"Mean signed error: {metrics['mean_signed_error']:.6f}")
    print(f"PSNR: {metrics['psnr']:.6f}")
    print(f"P50 abs error: {metrics['p50_abs_error']:.6f}")
    print(f"P95 abs error: {metrics['p95_abs_error']:.6f}")
    print(f"P99 abs error: {metrics['p99_abs_error']:.6f}")

    for channel_name, channel_metrics in metrics["per_channel"].items():
        print(
            f"{channel_name}: MAE={channel_metrics['mae']:.6f}, "
            f"MSE={channel_metrics['mse']:.6f}, "
            f"RMSE={channel_metrics['rmse']:.6f}, "
            f"MaxAbs={channel_metrics['max_abs_error']:.6f}, "
            f"MeanSigned={channel_metrics['mean_signed_error']:.6f}"
        )


if __name__ == "__main__":
    main()
