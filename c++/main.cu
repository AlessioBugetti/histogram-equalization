/*
* SPDX-License-Identifier: GPL-3.0-only
 *
 * Author: Alessio Bugetti <alessiobugetti98@gmail.com>
 */

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include "kernel.cu"

#define BLOCK_SIZE 1024
#define NUM_ITERATIONS 1000

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
        if (!cdfMinIsSet)
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

    cudaMalloc(reinterpret_cast<void**>(&deviceInput), imageSize);
    cudaMalloc(reinterpret_cast<void**>(&deviceOutput), imageSize);
    cudaMalloc(reinterpret_cast<void**>(&deviceHistogram), histogramSize);
    cudaMalloc(reinterpret_cast<void**>(&deviceCdf), histogramSize);

    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(deviceHistogram, 0, histogramSize);

    unsigned int gridDim = (pixelCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CalculateHistogram<<<gridDim, BLOCK_SIZE>>>(deviceInput, deviceHistogram, pixelCount);

    switch (scanType)
    {
    case ScanType::KoggeStone:
        KoggeStoneScan<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);
        break;
    case ScanType::KoggeStoneDoubleBuffer:
        KoggeStoneScanDoubleBuffer<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);
        break;
    case ScanType::BrentKung:
        BrentKungScan<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);
        break;
    default:
        std::cerr << "Error: Invalid scan type specified!" << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        cudaFree(deviceHistogram);
        cudaFree(deviceCdf);
        return;
    }

    unsigned int hostCdf[NUM_BINS];
    cudaMemcpy(hostCdf, deviceCdf, histogramSize, cudaMemcpyDeviceToHost);

    unsigned int cdfMin = 0;
    for (unsigned int value : hostCdf)
    {
        if (value > 0)
        {
            cdfMin = value;
            break;
        }
    }

    NormalizeCdf<<<1, NUM_BINS>>>(deviceCdf, cdfMin, pixelCount);
    EqualizeHistogram<<<gridDim, BLOCK_SIZE>>>(deviceOutput, deviceInput, deviceCdf, pixelCount);

    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceHistogram);
    cudaFree(deviceCdf);
}

int
main()
{
    const std::string data_directory = "../../data/";

    std::vector<std::tuple<std::string, double, double, double, double, double, double, double>>
        results;

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

        long totalSequentialTime = 0;
        long totalCudaTimeKoggeStone = 0;
        long totalCudaTimeKoggeStoneDoubleBuffer = 0;
        long totalCudaTimeBrentKung = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::KoggeStone);
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeKoggeStone +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(),
                                      output.ptr(),
                                      pixelCount,
                                      ScanType::KoggeStoneDoubleBuffer);
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeKoggeStoneDoubleBuffer +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

            start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::BrentKung);
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTimeBrentKung +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        double meanSequentialTime = static_cast<double>(totalSequentialTime) / NUM_ITERATIONS;
        double meanCudaTimeKoggeStone =
            static_cast<double>(totalCudaTimeKoggeStone) / NUM_ITERATIONS;
        double meanCudaTimeKoggeStoneDoubleBuffer =
            static_cast<double>(totalCudaTimeKoggeStoneDoubleBuffer) / NUM_ITERATIONS;
        double meanCudaTimeBrentKung = static_cast<double>(totalCudaTimeBrentKung) / NUM_ITERATIONS;

        double speedupKoggeStone = meanSequentialTime / meanCudaTimeKoggeStone;
        double speedupKoggeStoneDoubleBuffer =
            meanSequentialTime / meanCudaTimeKoggeStoneDoubleBuffer;
        double speedupBrentKung = meanSequentialTime / meanCudaTimeBrentKung;

        results.emplace_back(image_path,
                             meanSequentialTime,
                             meanCudaTimeKoggeStone,
                             speedupKoggeStone,
                             meanCudaTimeKoggeStoneDoubleBuffer,
                             speedupKoggeStoneDoubleBuffer,
                             meanCudaTimeBrentKung,
                             speedupBrentKung);
    }

    for (const auto& result : results)
    {
        std::cout << "Image: " << std::get<0>(result) << std::endl;
        std::cout << "Average Sequential Time: " << std::get<1>(result) << " ns" << std::endl;
        std::cout << "Average CUDA Time (Kogge-Stone): " << std::get<2>(result) << " ns"
                  << std::endl;
        std::cout << "Speedup (Kogge-Stone): " << std::get<3>(result) << std::endl;
        std::cout << "Average CUDA Time (Kogge-Stone Double Buffer): " << std::get<4>(result)
                  << " ns" << std::endl;
        std::cout << "Speedup (Kogge-Stone Double Buffer): " << std::get<5>(result) << std::endl;
        std::cout << "Average CUDA Time (Brent-Kung): " << std::get<6>(result) << " ns"
                  << std::endl;
        std::cout << "Speedup (Brent-Kung): " << std::get<7>(result) << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return 0;
}