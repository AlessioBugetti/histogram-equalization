#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 1024
#define NUM_BINS 256

void
SequentialHistogramEqualization(const unsigned char* input,
                                unsigned char* output,
                                const unsigned int pixelCount)
{
    unsigned int histogram[NUM_BINS] = {0};
    unsigned int cdf[NUM_BINS] = {0};

    int cdfMin = -1;

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
        if (cdfMin == -1 && cdf[i] > 0)
        {
            cdfMin = static_cast<int>(cdf[i]);
        }
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

__global__ void
CalculateHistogram(const unsigned char* input,
                   unsigned int* histogram,
                   const unsigned int pixelCount)
{
    __shared__ unsigned int cache[NUM_BINS];
    if (threadIdx.x < NUM_BINS)
    {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    while (tid < pixelCount)
    {
        atomicAdd(&(cache[input[tid]]), 1);
        tid += stride;
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS)
    {
        atomicAdd(&(histogram[threadIdx.x]), cache[threadIdx.x]);
    }
}

__global__ void
CalculateCdf(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int tmp[NUM_BINS];
    const unsigned int tid = threadIdx.x;

    if (tid < NUM_BINS)
    {
        tmp[tid] = histogram[tid];
    }

    for (unsigned int stride = 1; stride < NUM_BINS; stride *= 2)
    {
        __syncthreads();
        if (tid >= stride)
        {
            tmp[tid] += tmp[tid - stride];
        }
    }
    __syncthreads();

    if (tid < NUM_BINS)
    {
        cdf[tid] = tmp[tid];
    }
}

__global__ void
NormalizeCdf(unsigned int* cdf, const unsigned int cdfMin, const unsigned int pixelCount)
{
    if (const unsigned int tid = threadIdx.x; tid < NUM_BINS)
    {
        cdf[tid] = ((cdf[tid] - cdfMin) * 255) / (pixelCount - cdfMin);
    }
}

__global__ void
EqualizeHistogram(unsigned char* output,
                  const unsigned char* input,
                  const unsigned int* cdf,
                  const unsigned int pixelCount)
{
    if (const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < pixelCount)
    {
        output[tid] = cdf[input[tid]];
    }
}

void
CudaHistogramEqualization(const unsigned char* hostInput,
                          unsigned char* hostOutput,
                          const unsigned int pixelCount)
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

    CalculateCdf<<<1, NUM_BINS>>>(deviceCdf, deviceHistogram);

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
    const std::string data_directory = "../data/";

    std::vector<std::tuple<std::string, double, double, double>> results;

    for (const auto& entry : std::filesystem::directory_iterator(data_directory))
    {
        constexpr int iterations = 10000;
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
        long totalCudaTime = 0;

        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        for (int i = 0; i < iterations; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            auto stop = std::chrono::high_resolution_clock::now();
            totalCudaTime +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        double meanSequentialTime = static_cast<double>(totalSequentialTime) / iterations;
        double meanCudaTime = static_cast<double>(totalCudaTime) / iterations;

        double speedup = meanSequentialTime / meanCudaTime;

        results.emplace_back(image_path, meanSequentialTime, meanCudaTime, speedup);
    }

    for (const auto& result : results)
    {
        std::cout << "Image: " << std::get<0>(result) << std::endl;
        std::cout << "Average Sequential Time: " << std::get<1>(result) << " ns" << std::endl;
        std::cout << "Average CUDA Time: " << std::get<2>(result) << " ns" << std::endl;
        std::cout << "Speedup: " << std::get<3>(result) << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return 0;
}
