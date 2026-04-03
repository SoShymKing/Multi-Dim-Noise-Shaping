from PIL import Image
import numpy as np

from modules.generated_output_paths import clean_bp_mul_array_path, preview_output_path


def _default_packed_path(image_path: str) -> str:
    return clean_bp_mul_array_path(image_path)


def _default_output_path(image_path: str) -> str:
    return preview_output_path("output_clean_bp", image_path)


def raw_dsm_print(
    image_path: str,
    maximum: float = 255,
    dim: int = 2,
    bit: int = 3,
    packed_input_path: str | None = None,
    output_image_path: str | None = None,
) -> str:
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load(packed_input_path or _default_packed_path(image_path))
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")

    if not np.issubdtype(loaded_array.dtype, np.integer):
        loaded_array = loaded_array.astype(np.uint64)

    print(f"Image as NumPy array shape: {loaded_array.shape}")

    levels = (1 << bit) - 1
    if levels > 0 and maximum > 0:
        reconstructed_image = loaded_array.astype(np.float64) / levels * maximum
    else:
        reconstructed_image = np.zeros_like(loaded_array, dtype=np.float64)

    reconstructed_image = np.clip(np.rint(reconstructed_image), 0, 255).astype(np.uint8)

    output_path = output_image_path or _default_output_path(image_path)
    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    Image.fromarray(reconstructed_image, mode="RGB").save(output_path)
    return output_path
