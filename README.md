# CUDA-Based Histogram Equalization

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
    - [C++](#c)
    - [Python](#python)
- [Cuda Kernels](#cuda-kernels)
    - [Included Kernels](#included-kernels)
- [Performance](#performance)
- [License](#license)
- [Author](#author)

## Overview
This project implements histogram equalization for grayscale images using CUDA. It leverages GPU parallel processing to achieve high performance, particularly for large datasets or computationally intensive tasks.
The algorithm incorporates different parallel prefix sum (scan) algorithms:
- Kogge-Stone
- Kogge-Stone with Double Buffer
- Brent-Kung

## Overview
Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.

Before Histogram Equalization:

<img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Unequalized_Hawkes_Bay_NZ.jpg" alt="Image before Histogram Equalization" width="300" />

Corresponding histogram (red) and cumulative histogram (black):

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Unequalized_Histogram.svg/2560px-Unequalized_Histogram.svg.png" alt="Corresponding histogram" width="300" />

After Histogram Equalization:

<img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Equalized_Hawkes_Bay_NZ.jpg" alt="Image after Histogram Equalization" width="300" />

Corresponding histogram (red) and cumulative histogram (black):

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Equalized_Histogram.svg/1920px-Equalized_Histogram.svg.png" alt="Equalized histogram" width="300" />

## Repository Structure

```plaintext
.
├── main.py        # Python script for invoking CUDA kernels and managing the workflow
├── main.cu        # CUDA source file containing the implementation of kernels
├── kernel.cu      # CUDA kernel definitions
└── data/          # Directory for input images
```

## Prerequisites

- CUDA-capable NVIDIA GPU
- CUDA Toolkit
- OpenCV
- C++ compiler
- CMake
- Python 3.x (for Python interface)
- Python Libraries:
    - numpy
    - opencv-python
    - pycuda

## Installation
1. Clone the repository:

```sh
git clone https://github.com/AlessioBugetti/histogram-equalization.git
cd histogram-equalization
```
2. Install Python dependencies:

```sh
pip install -r requirements.txt
```
3. Ensure the CUDA environment is set up:
    - Install NVIDIA drivers.
    - Install the CUDA Toolkit. 
    - Verify with ```nvcc --version```.

## Usage

### C++
```sh
# Build the project
mkdir build && cd build
cmake ..
make

# Run the program
./histogram_equalization
```

### Python
```sh
python main.py
```

## Cuda Kernels

### Included Kernels:

- CalculateHistogram: Computes the histogram for the input image.
- KoggeStoneScan: Performs parallel prefix sum using the Kogge-Stone algorithm.
- KoggeStoneScanDoubleBuffer: Performs parallel prefix sum using the Kogge-Stone algorithm with double buffer.
- BrentKungScan: Performs parallel prefix sum using the Brent-Kung algorithm.
- NormalizeCdf: Normalizes the CDF for intensity adjustment.
- EqualizeHistogram: Applies the equalized histogram to the input image.
  Performance
  The implementation includes benchmarking capabilities that measure:

## Performance
- Sequential CPU execution time
- CUDA execution time for each scan algorithm.
- Speedup ratios compared to CPU implementation.
- Measurements are averaged over multiple iterations to ensure reliable results.

## License
This project is licensed under the GPL-3.0-only License. See the [`LICENSE`](../LICENSE) file for more details.

## Author
Alessio Bugetti - alessiobugetti98@gmail.com