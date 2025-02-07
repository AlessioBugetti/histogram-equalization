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
    __shared__ unsigned int histShared[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histShared[bin] = 0;
    }
    __syncthreads();

    unsigned int accumulator = 0;
    unsigned int prevBin = NUM_BINS;

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = tid; index < pixelCount; index += blockDim.x * gridDim.x)
    {
        unsigned int currBin = input[index];
        if (currBin != prevBin)
        {
            if (prevBin != NUM_BINS)
            {
                atomicAdd(&histShared[prevBin], accumulator);
            }
            accumulator = 1;
            prevBin = currBin;
        }
        else
        {
            accumulator++;
        }
    }
    if (accumulator > 0)
    {
        atomicAdd(&histShared[prevBin], accumulator);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        unsigned int binValue = histShared[bin];
        if (binValue > 0)
        {
            atomicAdd(&histogram[bin], binValue);
        }
    }
}

__global__ void
KoggeStoneScan(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cdfShared[NUM_BINS];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_BINS)
    {
        cdfShared[threadIdx.x] = histogram[tid];
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int temp = 0;
        if (threadIdx.x >= stride)
        {
            temp = cdfShared[threadIdx.x - stride];
        }
        __syncthreads();

        if (threadIdx.x >= stride)
        {
            cdfShared[threadIdx.x] += temp;
        }
    }

    if (tid < NUM_BINS)
    {
        cdf[tid] = cdfShared[threadIdx.x];
    }
}

__global__ void
KoggeStoneScanDoubleBuffer(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cdfShared[NUM_BINS];
    __shared__ unsigned int cdfSharedAux[NUM_BINS];

    unsigned int* inputBuffer = cdfShared;
    unsigned int* outputBuffer = cdfSharedAux;

    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < NUM_BINS)
    {
        cdfShared[threadIdx.x] = histogram[tid];
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();

        if (threadIdx.x >= stride)
        {
            outputBuffer[threadIdx.x] =
                inputBuffer[threadIdx.x] + inputBuffer[threadIdx.x - stride];
        }
        else
        {
            outputBuffer[threadIdx.x] = inputBuffer[threadIdx.x];
        }

        unsigned int* temp = inputBuffer;
        inputBuffer = outputBuffer;
        outputBuffer = temp;
    }

    if (tid < NUM_BINS)
    {
        cdf[tid] = outputBuffer[threadIdx.x];
    }
}

__global__ void
BrentKungScan(unsigned int* cdf, const unsigned int* histogram)
{
    __shared__ unsigned int cdfShared[NUM_BINS];
    const unsigned int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BINS)
    {
        cdfShared[threadIdx.x] = histogram[tid];
    }

    if (tid + blockDim.x < NUM_BINS)
    {
        cdfShared[threadIdx.x + blockDim.x] = histogram[tid + blockDim.x];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < NUM_BINS)
        {
            cdfShared[index] += cdfShared[index - stride];
        }
    }

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < NUM_BINS)
        {
            cdfShared[index + stride] += cdfShared[index];
        }
    }

    __syncthreads();
    if (tid < NUM_BINS)
    {
        cdf[tid] = cdfShared[threadIdx.x];
    }
    if (tid + blockDim.x < NUM_BINS)
    {
        cdf[tid + blockDim.x] = cdfShared[threadIdx.x + blockDim.x];
    }
}

__global__ void
NormalizeCdf(unsigned int* cdf, const unsigned int pixelCount)
{
    __shared__ unsigned int cdfMinIndex;

    const unsigned int tid = threadIdx.x;

    if (tid == 0)
    {
        cdfMinIndex = NUM_BINS;
    }
    __syncthreads();

    if (tid < NUM_BINS && cdf[tid] != 0)
    {
        atomicMin(&cdfMinIndex, tid);
    }
    __syncthreads();

    __shared__ unsigned int cdfMin;

    if (tid == 0)
    {
        cdfMin = cdf[cdfMinIndex];
    }
    __syncthreads();

    if (tid < NUM_BINS)
    {
        cdf[tid] = ((cdf[tid] - cdfMin) * (NUM_BINS - 1) + (pixelCount - cdfMin) / 2) /
                   (pixelCount - cdfMin);
    }
}

__global__ void
EqualizeHistogram(unsigned char* output,
                  const unsigned char* input,
                  const unsigned int* cdf,
                  const unsigned int pixelCount)
{
    __shared__ unsigned int cdfShared[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        cdfShared[bin] = cdf[bin];
    }
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = tid; index < pixelCount; index += blockDim.x * gridDim.x)
    {
        output[index] = cdfShared[input[index]];
    }
}