from PIL import Image
import numpy as np


def raw_dsm_print(image_path, maximum=255, dim=2, bit=3):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load("mul_array_" + image_path + ".npy")
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")

    print(f"Image as NumPy array shape: {loaded_array.shape}")

    levels = (1 << bit) - 1
    if levels > 0 and maximum > 0:
        sqrt_reconstructed = loaded_array.astype(np.float64) / levels * maximum
    else:
        sqrt_reconstructed = np.zeros_like(loaded_array, dtype=np.float64)

    reconstructed_image = np.power(sqrt_reconstructed, dim)
    np.clip(reconstructed_image, 0, 255, out=reconstructed_image)
    reconstructed_image = reconstructed_image.astype(np.uint8)

    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode="RGB")
    image.save("output.bmp")
