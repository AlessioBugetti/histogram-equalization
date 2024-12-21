#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

void SequentialHistogramEqualization(unsigned char* input, unsigned int height, unsigned int width) {
    constexpr unsigned int NUM_BINS = 256;
    unsigned int histogram[NUM_BINS] = {0};
    unsigned int cdf[NUM_BINS] = {0};

    int cdfMin = -1;

    for (unsigned int i = 0; i < height * width; i++) {
        histogram[input[i]]++;
    }

    for (unsigned int i = 0; i < NUM_BINS; i++) {
        cdf[i] = histogram[i];
        if (i > 0) {
            cdf[i] += cdf[i - 1];
        }
        if (cdfMin == -1 && cdf[i] > 0) {
            cdfMin = static_cast<int>(cdf[i]);
        }
    }

    for (unsigned int i = 0; i < NUM_BINS; i++) {
        cdf[i] = static_cast<unsigned int>(
            round((static_cast<double>(cdf[i] - cdfMin) / (height * width - cdfMin)) * 255)
        );
    }

    for (unsigned int i = 0; i < height * width; i++) {
        input[i] = static_cast<unsigned char>(cdf[input[i]]);
    }
}

int main() {
    const std::string input_name = "../image.jpg";

    cv::Mat input = cv::imread(input_name, cv::IMREAD_GRAYSCALE);

    if (input.empty()) {
        std::cerr << "Error: image not found!" << std::endl;
        return -1;
    }

    auto height = static_cast<unsigned int>(input.rows);
    auto width = static_cast<unsigned int>(input.cols);

    SequentialHistogramEqualization(input.ptr(), height, width);

    cv::imshow("Output", input);
    cv::waitKey(0);

    return 0;
}
