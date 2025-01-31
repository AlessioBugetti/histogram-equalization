/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Author: Alessio Bugetti <alessiobugetti98@gmail.com>
 */

#include "kernel.cu"

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 1024
#define NUM_ITERATIONS 1000
#define CHECK(call)                                                                                \
    {                                                                                              \
        const cudaError_t error_code = call;                                                       \
        if (error_code != cudaSuccess)                                                             \
        {                                                                                          \
            fprintf(stderr, "CUDA Error:\n");                                                      \
            fprintf(stderr, "    File:       %s\n", __FILE__);                                     \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                                     \
            fprintf(stderr, "    Error code: %d\n", error_code);                                   \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code));               \
            exit(1);                                                                               \
        }                                                                                          \
    }

void
SequentialHistogramEqualization(const unsigned char* input,
                                unsigned char* output,
                                const unsigned int pixelCount)
{
    if (input == nullptr || output == nullptr || pixelCount == 0)
    {
        return;
    }

    unsigned int histogram[NUM_BINS] = {0};
    unsigned int cdf[NUM_BINS] = {0};

    bool cdfMinIsSet = false;
    unsigned int cdfMin = std::numeric_limits<unsigned int>::max();

    for (unsigned int i = 0; i < pixelCount; i++)
    {
        histogram[input[i]]++;
    }

    for (unsigned int i = 0; i < NUM_BINS; i++)
    {
        cdf[i] = histogram[i];
        if (i > 0)
        {
            cdf[i] += cdf[i - 1];
        }
        if (!cdfMinIsSet && cdf[i] > 0)
        {
            cdfMin = cdf[i];
            cdfMinIsSet = true;
        }
    }

    if (pixelCount == cdfMin)
    {
        for (unsigned int i = 0; i < pixelCount; i++)
        {
            output[i] = input[i];
        }
        return;
    }

    for (unsigned int& value : cdf)
    {
        value = static_cast<unsigned int>(
            round(static_cast<double>(value - cdfMin) / (pixelCount - cdfMin) * (NUM_BINS - 1)));
    }

    for (unsigned int i = 0; i < pixelCount; i++)
    {
        output[i] = static_cast<unsigned char>(cdf[input[i]]);
    }
}

enum class ScanType
{
    KoggeStone,
    KoggeStoneDoubleBuffer,
    BrentKung
};

void
CudaHistogramEqualization(const unsigned char* hostInput,
                          unsigned char* hostOutput,
                          const unsigned int pixelCount,
                          const ScanType scanType)
{
    unsigned char *deviceInput, *deviceOutput;
    unsigned int *deviceHistogram, *deviceCdf;

    const size_t imageSize = pixelCount * sizeof(unsigned char);
    constexpr size_t histogramSize = NUM_BINS * sizeof(unsigned int);

    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput), imageSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), imageSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceHistogram), histogramSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCdf), histogramSize));

    CHECK(cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(deviceHistogram, 0, histogramSize));

    dim3 gridDimImage((pixelCount + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 blockDimImage(BLOCK_SIZE, 1, 1);

    CalculateHistogram<<<gridDimImage, blockDimImage>>>(deviceInput, deviceHistogram, pixelCount);

    switch (scanType)
    {
    case ScanType::KoggeStone:
        KoggeStoneScan<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);
        break;
    case ScanType::KoggeStoneDoubleBuffer:
        KoggeStoneScanDoubleBuffer<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);
        break;
    case ScanType::BrentKung:
        BrentKungScan<<<1, NUM_BINS / 2>>>(deviceCdf, deviceHistogram);
        break;
    default:
        std::cerr << "Error: Invalid scan type specified!" << std::endl;
        CHECK(cudaFree(deviceInput));
        CHECK(cudaFree(deviceOutput));
        CHECK(cudaFree(deviceHistogram));
        CHECK(cudaFree(deviceCdf));
        return;
    }

    NormalizeCdf<<<1, NUM_BINS>>>(deviceCdf, pixelCount);
    EqualizeHistogram<<<gridDimImage, blockDimImage>>>(deviceOutput,
                                                       deviceInput,
                                                       deviceCdf,
                                                       pixelCount);

    CHECK(cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
    CHECK(cudaFree(deviceHistogram));
    CHECK(cudaFree(deviceCdf));
}

int
main()
{
    const std::string data_directory = "../../data/";

    for (const auto& entry : std::filesystem::directory_iterator(data_directory))
    {
        std::string image_path = entry.path().string();

        cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat output(input.size(), input.type());

        if (input.empty())
        {
            std::cerr << "Error: image " << image_path << " not found!" << std::endl;
            continue;
        }

        const auto pixelCount = input.rows * input.cols;

        // Warmup phase
        for (int i = 0; i < 10; i++)
        {
            SequentialHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::KoggeStone);
            CudaHistogramEqualization(input.ptr(),
                                      output.ptr(),
                                      pixelCount,
                                      ScanType::KoggeStoneDoubleBuffer);
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::BrentKung);
        }

        double totalSequentialTime = 0;
        double totalCudaTimeKoggeStone = 0;
        double totalCudaTimeKoggeStoneDoubleBuffer = 0;
        double totalCudaTimeBrentKung = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime += std::chrono::duration<double, std::milli>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::KoggeStone);
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeKoggeStone +=
                std::chrono::duration<double, std::milli>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(),
                                      output.ptr(),
                                      pixelCount,
                                      ScanType::KoggeStoneDoubleBuffer);
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeKoggeStoneDoubleBuffer +=
                std::chrono::duration<double, std::milli>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::BrentKung);
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeBrentKung +=
                std::chrono::duration<double, std::milli>(stop - start).count();
        }

        double meanSequentialTime = totalSequentialTime / NUM_ITERATIONS;
        double meanCudaTimeKoggeStone = totalCudaTimeKoggeStone / NUM_ITERATIONS;
        double meanCudaTimeKoggeStoneDoubleBuffer =
            totalCudaTimeKoggeStoneDoubleBuffer / NUM_ITERATIONS;
        double meanCudaTimeBrentKung = totalCudaTimeBrentKung / NUM_ITERATIONS;

        double speedupKoggeStone = meanSequentialTime / meanCudaTimeKoggeStone;
        double speedupKoggeStoneDoubleBuffer =
            meanSequentialTime / meanCudaTimeKoggeStoneDoubleBuffer;
        double speedupBrentKung = meanSequentialTime / meanCudaTimeBrentKung;

        std::cout << std::setprecision(6);
        std::cout << "Image: " << image_path << std::endl;
        std::cout << "Average Sequential Time: " << meanSequentialTime << " ms" << std::endl;
        std::cout << "Average CUDA Time (Kogge-Stone): " << meanCudaTimeKoggeStone << " ms"
                  << std::endl;
        std::cout << "Speedup (Kogge-Stone): " << speedupKoggeStone << std::endl;
        std::cout << "Average CUDA Time (Kogge-Stone Double Buffer): "
                  << meanCudaTimeKoggeStoneDoubleBuffer << " ms" << std::endl;
        std::cout << "Speedup (Kogge-Stone Double Buffer): " << speedupKoggeStoneDoubleBuffer
                  << std::endl;
        std::cout << "Average CUDA Time (Brent-Kung): " << meanCudaTimeBrentKung << " ms"
                  << std::endl;
        std::cout << "Speedup (Brent-Kung): " << speedupBrentKung << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return 0;
}