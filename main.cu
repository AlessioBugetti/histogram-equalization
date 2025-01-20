#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 1024
#define NUM_BINS 256
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
KoggeStoneScan(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cache[NUM_BINS];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_BINS)
    {
        cache[threadIdx.x] = histogram[tid];
    }
    else
    {
        cache[threadIdx.x] = 0;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int temp = cache[threadIdx.x];
        if (threadIdx.x >= stride)
        {
            temp += cache[threadIdx.x - stride];
        }
        __syncthreads();

        if (threadIdx.x >= stride)
        {
            cache[threadIdx.x] = temp;
        }
    }

    if (tid < NUM_BINS)
    {
        cdf[tid] = cache[threadIdx.x];
    }
}

__global__ void
KoggeStoneScanDoubleBuffer(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cache[NUM_BINS];
    __shared__ unsigned int cacheAux[NUM_BINS];

    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < NUM_BINS)
    {
        cache[threadIdx.x] = histogram[tid];
        cacheAux[threadIdx.x] = histogram[tid];
    }
    else
    {
        cache[threadIdx.x] = 0;
        cacheAux[threadIdx.x] = 0;
    }

    unsigned int* inputBuffer = cache;
    unsigned int* outputBuffer = cacheAux;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        unsigned int* temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;

        if (threadIdx.x >= stride)
        {
            outputBuffer[threadIdx.x] =
                inputBuffer[threadIdx.x] + inputBuffer[threadIdx.x - stride];
        }
        else
        {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x];
        }
    }

    if (tid < NUM_BINS)
    {
        cdf[tid] = outputBuffer[threadIdx.x];
    }
}

__global__ void
BrentKungScan(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cache[NUM_BINS];
    unsigned int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BINS)
    {
        cache[threadIdx.x] = histogram[tid];
    }

    if (tid + blockDim.x < NUM_BINS)
    {
        cache[threadIdx.x + blockDim.x] = histogram[tid + blockDim.x];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < NUM_BINS)
        {
            cache[index] += cache[index - stride];
        }
    }

    for (unsigned int stride = NUM_BINS / 4; stride > 0; stride /= 2)
    {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < NUM_BINS)
        {
            cache[index + stride] += cache[index];
        }
    }

    __syncthreads();
    if (tid < NUM_BINS)
    {
        cdf[tid] = cache[threadIdx.x];
    }
    if (tid + blockDim.x < NUM_BINS)
    {
        cdf[tid + blockDim.x] = cache[threadIdx.x + blockDim.x];
    }
}

__global__ void
NormalizeCdf(unsigned int* cdf, const unsigned int cdfMin, const unsigned int pixelCount)
{
    if (const unsigned int tid = threadIdx.x; tid < NUM_BINS)
    {
        cdf[tid] = __double2int_rn(static_cast<double>(cdf[tid] - cdfMin) / (pixelCount - cdfMin) *
                                   (NUM_BINS - 1));
    }
}

__global__ void
EqualizeHistogram(unsigned char* output,
                  const unsigned char* input,
                  const unsigned int* cdf,
                  const unsigned int pixelCount)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    while (tid < pixelCount)
    {
        output[tid] = cdf[input[tid]];
        tid += stride;
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
    const std::string data_directory = "../data/";

    std::vector<std::tuple<std::string, double, double, double>> results;

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
        long totalCudaTime = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialHistogramEqualization(input.ptr(), output.ptr(), pixelCount);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            CudaHistogramEqualization(input.ptr(), output.ptr(), pixelCount, ScanType::KoggeStone);
            auto stop = std::chrono::high_resolution_clock::now();
            totalCudaTime +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        double meanSequentialTime = static_cast<double>(totalSequentialTime) / NUM_ITERATIONS;
        double meanCudaTime = static_cast<double>(totalCudaTime) / NUM_ITERATIONS;

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