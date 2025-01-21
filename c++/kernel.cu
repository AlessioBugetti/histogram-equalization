/*
* SPDX-License-Identifier: GPL-3.0-only
 *
 * Author: Alessio Bugetti <alessiobugetti98@gmail.com>
 */

#define NUM_BINS 256

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