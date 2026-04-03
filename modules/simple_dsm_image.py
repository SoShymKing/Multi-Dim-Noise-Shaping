from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from modules.generated_output_paths import mul_array_path

def dsm_conv_image_modulation(image_path, order=1):
    image = Image.open(image_path)
    print(f"Image size: {image.size}, mode: {image.mode}")

    convert_h = image.size[0]
    convert_w = image.size[1]
    dim = 2
    max_value = np.float_power(256, 1/dim)
    w_array = np.zeros((convert_w, image.size[0], 3), dtype=np.int8)
    wr_array = np.zeros((convert_w, image.size[0], 3), dtype=np.int8)
    h_array = np.zeros((image.size[1], convert_h, 3), dtype=np.int8)
    hr_array = np.zeros((image.size[1], convert_h, 3), dtype=np.int8)
    mul_array = np.zeros((convert_w, convert_h, 3), dtype=np.int8)
    conv_array = np.zeros((convert_w*3, convert_h*3, 3), dtype=np.float32)
    
    image_array = np.array(image, dtype=np.float64)
    image_array= np.float_power(image_array, 1/dim)
    maximum = np.max(image_array) # np.percentile(image_array,99) 256
    median = (np.max(image_array) + np.min(image_array)) / 2
    image_array = image_array/maximum
    sqrt_image_array = image_array 
    print(f"maximum: {maximum}")
    
    print(f"Image as NumPy array shape: {sqrt_image_array.shape}, dtype: {sqrt_image_array.dtype}")
    integ_j = {}
    integ_i = {}
    integ_jr = {}
    integ_ir = {}

    for k in range(3):
        for i in range(image.size[0]):
            for l in range(order):
                integ_j[l] = 0.
                integ_jr[l] = 0.
            for j in range(convert_w):
                integ_j[0] += sqrt_image_array[j, i, k]
                integ_jr[0] += sqrt_image_array[sqrt_image_array.shape[0] - j - 1,
                                    sqrt_image_array.shape[1] -  i - 1,sqrt_image_array.shape[2] -  k - 1]
                
                for l in range(1, order):
                    integ_j[l] += integ_j[l - 1]
                    integ_jr[l] += integ_jr[l - 1]
                    
                y = 1 if integ_j[order-1]>=0 else 0
                yr = 1 if integ_jr[order-1]>=0 else 0
                
                for l in range(0, order):
                    integ_j[l] -= y
                    integ_jr[l] -= yr
                    
                w_array[j, i, k] = y
                wr_array[wr_array.shape[0] - j - 1, wr_array.shape[1] - i - 1, wr_array.shape[2] - k - 1] = yr
                    

    print(f"w_array shape: {w_array.shape}, dtype: {w_array.dtype}")

    for k in range(3):
        for j in range(image.size[1]):
            for l in range(order):
                integ_i[l] = 0.
                integ_ir[l] = 0.
            for i in range(convert_h):
                integ_i[0] += sqrt_image_array[j, i, k]
                integ_ir[0] += sqrt_image_array[sqrt_image_array.shape[0] - 1 - j,
                                    sqrt_image_array.shape[1] - 1 - i,sqrt_image_array.shape[2] - 1 - k]
                
                for l in range(1, order):
                    integ_i[l] += integ_i[l - 1]
                    integ_ir[l] += integ_ir[l - 1]
                    
                
                y = 1 if integ_i[order-1]>=0 else 0
                yr = 1 if integ_ir[order-1]>=0 else 0
                
                for l in range(0, order):
                    integ_i[l] -= y
                    integ_ir[l] -= yr
                    
                hr_array[hr_array.shape[0] - 1 - j, hr_array.shape[1] - 1 - i, hr_array.shape[2] - 1 - k] = yr
                h_array[j, i, k] = y
    print(f"h_array shape: {h_array.shape}, dtype: {h_array.dtype}")

    for k in range(3):
        for j in range(convert_w):
            for i in range(convert_h):
                mul_array[j, i, k] = (w_array[j, i, k] * h_array[j, i, k] )
    np.save(mul_array_path(image_path), mul_array)
    print(f"mul_array shape: {mul_array.shape}, dtype: {mul_array.dtype}")
    return median, maximum, dim
