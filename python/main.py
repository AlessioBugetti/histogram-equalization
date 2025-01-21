#
# SPDX-License-Identifier: GPL-3.0-only
#
# Author: Alessio Bugetti <alessiobugetti98@gmail.com>
#

import os
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

BLOCK_SIZE = 1024
NUM_BINS = 256
NUM_ITERATIONS = 1000


def cuda_histogram_equalization(input_image):
    if input_image.ndim != 2:
        raise ValueError("The input image must be in grayscale (2 dimensions)")

    original_shape = input_image.shape
    pixel_count = input_image.size

    input_image = input_image.flatten()
    output_image = np.zeros_like(input_image, dtype=np.uint8)

    histogram = np.zeros(NUM_BINS, dtype=np.uint32)
    cdf = np.zeros(NUM_BINS, dtype=np.uint32)

    try:
        with open("../c++/kernel.cu", 'r') as f:
            cuda_code = f.read()
    except FileNotFoundError:
        print("Error: kernel.cu file not found")
        return

    module = SourceModule(cuda_code)

    calculate_histogram = module.get_function("CalculateHistogram")
    kogge_stone_scan_double_buffer = module.get_function("KoggeStoneScanDoubleBuffer")
    normalize_cdf = module.get_function("NormalizeCdf")
    equalize_histogram = module.get_function("EqualizeHistogram")

    device_input = cuda.mem_alloc(input_image.nbytes)
    device_output = cuda.mem_alloc(output_image.nbytes)
    device_histogram = cuda.mem_alloc(histogram.nbytes)
    device_cdf = cuda.mem_alloc(cdf.nbytes)

    cuda.memcpy_htod(device_input, input_image)
    cuda.memcpy_htod(device_histogram, histogram)

    grid_dim = ((pixel_count + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    calculate_histogram(
        device_input, device_histogram, np.uint32(pixel_count),
        block=(BLOCK_SIZE, 1, 1), grid=grid_dim
    )

    kogge_stone_scan_double_buffer(
        device_cdf, device_histogram,
        block=(NUM_BINS, 1, 1), grid=(1, 1, 1)
    )

    host_cdf = np.zeros(NUM_BINS, dtype=np.uint32)

    cuda.memcpy_dtoh(host_cdf, device_cdf)

    cdf_min = next((value for value in host_cdf if value > 0), 0)

    normalize_cdf(
        device_cdf, np.float32(cdf_min), np.uint32(pixel_count),
        block=(NUM_BINS, 1, 1), grid=(1, 1, 1)
    )

    equalize_histogram(
        device_output, device_input, device_cdf, np.uint32(pixel_count),
        block=(BLOCK_SIZE, 1, 1), grid=grid_dim
    )

    cuda.memcpy_dtoh(output_image, device_output)

    device_histogram.free()
    device_cdf.free()
    device_input.free()
    device_output.free()

    return output_image.reshape(original_shape)


if __name__ == "__main__":
    data_directory = "../data/"

    for file_name in os.listdir(data_directory):
        image_path = os.path.join(data_directory, file_name)
        input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        totalCudaTime = 0
        for iteration in range(NUM_ITERATIONS):
            start = time.perf_counter_ns()
            output_image = cuda_histogram_equalization(input_image)
            stop = time.perf_counter_ns()
            totalCudaTime += stop - start

        meanCudaTime = totalCudaTime / NUM_ITERATIONS

        print(f"Image: {file_name}")
        print(f"Average CUDA Time (Kogge-Stone Double Buffer): {meanCudaTime} ns")
        print("------------------------")
