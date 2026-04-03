from PIL import Image
import numpy as np


def raw_dsm_print(image_path, maximum=255, dim=2, channel_bit=3):
    if channel_bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load("mul_array_" + image_path + ".npy")
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded multi-bit array must have shape (height, width, 3)")

    print(f"Image as NumPy array shape: {loaded_array.shape}")

    levels = np.power((1 << channel_bit) - 1, dim)
    if levels > 0 and maximum > 0:
        clipped_levels = np.clip(loaded_array.astype(np.float64), 0, levels)
        sqrt_reconstructed = clipped_levels / levels * maximum
    else:
        sqrt_reconstructed = np.zeros_like(loaded_array, dtype=np.float64)

    reconstructed_image = sqrt_reconstructed
    np.clip(reconstructed_image, 0, 255, out=reconstructed_image)
    reconstructed_image = reconstructed_image.astype(np.uint8)

    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode="RGB")
    image.save("output.bmp")
