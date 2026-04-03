from PIL import Image
import numpy as np

from modules.generated_output_paths import bp_raw_output_path, mul_array_path


def raw_dsm_print(image_path, maximum=255, dim=2, bit=3):
    if bit < 1:
        raise ValueError("bit must be at least 1")

    loaded_array = np.load(mul_array_path(image_path))
    if loaded_array.ndim != 3 or loaded_array.shape[2] != 3:
        raise ValueError("Loaded bit-plane array must have shape (height, width, 3)")
    if not np.issubdtype(loaded_array.dtype, np.integer):
        loaded_array = loaded_array.astype(np.uint64)
    print(f"Image as NumPy array shape: {loaded_array.shape}")

    reconstructed_image = np.zeros_like(loaded_array, dtype=np.float64)
    _local_max = 0
    for i in range(bit):
        local_max = maximum/np.power(2, bit-1-i) # - ((maximum/np.power(2, bit-i)) if i>0 else 0)
        # _local_max += local_max
        reconstructed_image += ((loaded_array >> i) & 1) * local_max
    # reconstructed_image = maximum * reconstructed_image / _local_max
    reconstructed_image = np.rint(reconstructed_image).astype(np.uint32)
    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    np.save(bp_raw_output_path(), reconstructed_image)
