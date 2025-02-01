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

BLOCK_SIZE = 512
NUM_BINS = 256
NUM_ITERATIONS = 1000

def sequential_histogram_equalization(input_image):
    if input_image.ndim != 2:
        raise ValueError("The input image must be in grayscale (2 dimensions)")
    pixel_count = input_image.size

    histogram = np.zeros(NUM_BINS, dtype=np.int32)

    for pixel in input_image.flatten():
        histogram[pixel] += 1

    cdf = np.empty_like(histogram)
    cdf[0] = histogram[0]
    for i in range(1, len(histogram)):
        cdf[i] = cdf[i - 1] + histogram[i]

    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = (cdf - cdf_min) * (NUM_BINS - 1) / (pixel_count - cdf_min)
    cdf_normalized = np.round(cdf_normalized).astype(np.uint8)

    output_image = cdf_normalized[input_image]

    return output_image


def cuda_histogram_equalization(input_image, scan_type):
    if input_image.ndim != 2:
        raise ValueError("The input image must be in grayscale (2 dimensions)")

    pixel_count = input_image.size

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
    kogge_stone_scan = module.get_function("KoggeStoneScan")
    kogge_stone_scan_double_buffer = module.get_function("KoggeStoneScanDoubleBuffer")
    brent_kung_scan = module.get_function("BrentKungScan")
    normalize_cdf = module.get_function("NormalizeCdf")
    equalize_histogram = module.get_function("EqualizeHistogram")

    device_input = cuda.mem_alloc(input_image.nbytes)
    device_output = cuda.mem_alloc(output_image.nbytes)
    device_histogram = cuda.mem_alloc(histogram.nbytes)
    device_cdf = cuda.mem_alloc(cdf.nbytes)

    cuda.memcpy_htod(device_input, input_image)
    cuda.memcpy_htod(device_histogram, histogram)

    grid_dim = ((pixel_count + BLOCK_SIZE - 1) // BLOCK_SIZE // 4, 1, 1)

    calculate_histogram(
        device_input, device_histogram, np.uint32(pixel_count),
        block=(BLOCK_SIZE, 1, 1), grid=grid_dim
    )

    if scan_type == 'KoggeStone':
        kogge_stone_scan(
            device_cdf, device_histogram,
            block=(NUM_BINS, 1, 1), grid=(1, 1, 1)
        )
    elif scan_type == 'KoggeStoneDoubleBuffer':
        kogge_stone_scan_double_buffer(
            device_cdf, device_histogram,
            block=(NUM_BINS, 1, 1), grid=(1, 1, 1)
        )
    elif scan_type == 'BrentKung':
        brent_kung_scan(
            device_cdf, device_histogram,
            block=(NUM_BINS//2, 1, 1), grid=(1, 1, 1)
        )
    else:
        raise ValueError("Invalid scan type specified")

    normalize_cdf(
        device_cdf, np.uint32(pixel_count),
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

    return output_image


if __name__ == "__main__":
    data_directory = "../data/"

    for file_name in os.listdir(data_directory):
        image_path = os.path.join(data_directory, file_name)
        input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Warm-up
        for i in range(10):
            sequential_histogram_equalization(input_image)
            cuda_histogram_equalization(input_image, scan_type='KoggeStone')
            cuda_histogram_equalization(input_image, scan_type='KoggeStoneDoubleBuffer')
            cuda_histogram_equalization(input_image, scan_type='BrentKung')

        totalSequentialTime = 0
        totalCudaTimeKoggeStone = 0
        totalCudaTimeKoggeStoneDoubleBuffer = 0
        totalCudaTimeBrentKung = 0

        for iteration in range(NUM_ITERATIONS):
            startSequential = time.perf_counter_ns()
            output_image_sequential = sequential_histogram_equalization(input_image)
            stopSequential = time.perf_counter_ns()
            totalSequentialTime += stopSequential - startSequential

            startCudaKoggeStone = time.perf_counter_ns()
            output_image_cuda = cuda_histogram_equalization(input_image, "KoggeStone")
            stopCudaKoggeStone = time.perf_counter_ns()
            totalCudaTimeKoggeStone += stopCudaKoggeStone - startCudaKoggeStone

            startCudaKoggeStoneDoubleBuffer = time.perf_counter_ns()
            output_image_cuda = cuda_histogram_equalization(input_image, "KoggeStoneDoubleBuffer")
            stopCudaKoggeStoneDoubleBuffer = time.perf_counter_ns()
            totalCudaTimeKoggeStoneDoubleBuffer += stopCudaKoggeStoneDoubleBuffer - startCudaKoggeStoneDoubleBuffer

            startCudaBrentKung = time.perf_counter_ns()
            output_image_cuda = cuda_histogram_equalization(input_image, "BrentKung")
            stopCudaBrentKung = time.perf_counter_ns()
            totalCudaTimeBrentKung += stopCudaBrentKung - startCudaBrentKung

        meanSequentialTime = totalSequentialTime / (NUM_ITERATIONS * 1000000)
        meanCudaTimeKoggeStone = totalCudaTimeKoggeStone / (NUM_ITERATIONS * 1000000)
        meanCudaTimeKoggeStoneDoubleBuffer = totalCudaTimeKoggeStoneDoubleBuffer / (NUM_ITERATIONS * 1000000)
        meanCudaTimeBrentKung = totalCudaTimeBrentKung / (NUM_ITERATIONS * 1000000)

        speedupKoggeStone = meanSequentialTime / meanCudaTimeKoggeStone
        speedupKoggeStoneDoubleBuffer = meanSequentialTime / meanCudaTimeKoggeStoneDoubleBuffer
        speedupBrentKung = meanSequentialTime / meanCudaTimeBrentKung

        print(f"Image: {file_name}")
        print(f"Average Sequential Time: {meanSequentialTime:.6g} ms")
        print(f"Average CUDA Time (Kogge-Stone): {meanCudaTimeKoggeStone:.6g} ms")
        print(f"Speedup (Kogge-Stone): {speedupKoggeStone}")
        print(f"Average CUDA Time (Kogge-Stone Double Buffer): {meanCudaTimeKoggeStoneDoubleBuffer:.6g} ms")
        print(f"Speedup (Kogge-Stone Double Buffer): {speedupKoggeStoneDoubleBuffer:.6g}")
        print(f"Average CUDA Time (Brent-Kung): {meanCudaTimeBrentKung:.6g} ms")
        print(f"Speedup (Brent-Kung): {speedupBrentKung:.6g}")
        print("------------------------")

