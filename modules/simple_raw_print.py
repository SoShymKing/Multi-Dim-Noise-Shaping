from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def raw_dsm_print(image_path, median=127, maximum=255, dim=2):    
    loaded_array = np.load("mul_array_" + image_path + ".npy").astype(np.uint8)
    print(f"Image as NumPy array shape: {loaded_array.shape}")
    new_width = loaded_array.shape[0]
    new_height = loaded_array.shape[1]
    reconstructed_image = np.zeros((new_width, new_height,3), dtype=np.uint8)

    for p in range(3):
        for i in range(new_width):
            for j in range(new_height):
                reconstructed_image[i, j, p] = loaded_array[i, j, p] * min(np.power(maximum,dim),255)
    np.clip(reconstructed_image, 0, 255, out=reconstructed_image)
    print(f"Raw image mul shape: {reconstructed_image.shape}, dtype: {reconstructed_image.dtype}")
    image = Image.fromarray(reconstructed_image, mode='RGB')
    image.save("output.bmp")
    # image.save("raw_image_mul_" + image_path + ".bmp")
