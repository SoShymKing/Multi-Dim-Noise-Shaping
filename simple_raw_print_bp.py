from PIL import Image
import numpy as np


def raw_dsm_print(image_path, maximum=255, dim=2, bit=3):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load("mul_array_" + image_path + ".npy")
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")
    if not np.issubdtype(loaded_array.dtype, np.integer):
        loaded_array = loaded_array.astype(np.uint64)
    print(f"Image as NumPy array shape: {loaded_array.shape}")

    reconstructed_image = np.zeros_like(loaded_array, dtype=np.float64)
    for i in range(bit):
        local_max = maximum/np.power(2,bit-1-i)
        reconstructed_image += ((loaded_array >> i) & 1) * local_max

    reconstructed_image = np.clip(np.rint(reconstructed_image), 0, 255).astype(np.uint8)
    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode="RGB")
    image.save("output.bmp")
