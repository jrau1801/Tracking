#include <opencv2/opencv.hpp>
#include "../include/hog.h"
#include <opencv2/core/eigen.hpp>
#include "../libs/Eigen/Dense"
#include <chrono>
using namespace std::chrono;

using std::cerr;
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::destroyAllWindows;

void printEigenMatrix(const Eigen::MatrixXd &matrix) {
    std::cout << "[";
    for (int i = 0; i < matrix.rows(); ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << "[";
        for (int j = 0; j < matrix.cols(); ++j) {
            std::cout << std::setw(6) << matrix(i, j);
            if (j < matrix.cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < matrix.rows() - 1) {
            std::cout << "," << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

int main() {

    Mat image = cv::imread(R"(/Users/Louis/CLionProjects/Tracking/images/frame_15344.jpg)");

    if (image.empty()) {
        cerr << "Error: Couldn't load the image!" << std::endl;
        return -1;
    }

    Mat rImg;
    cv::resize(image, rImg, cv::Size(), 0.25, 0.25);


    cv::Mat grayImage;
    cv::Mat grayImage64;
    cv::cvtColor(rImg, grayImage, cv::COLOR_BGR2GRAY);

    grayImage.convertTo(grayImage64, CV_64F);

    int orientations = 9;
    std::pair<int, int> pixels_per_cell = std::make_pair(8, 8);
    std::pair<int, int> cells_per_block = std::make_pair(2, 2);
    std::string block_norm = "L2-Hys";
    bool visualize = false;
    bool transform_sqrt = true;
    bool feature_vector = true;


    auto start = high_resolution_clock::now();
    auto res = hog(grayImage64,
        orientations,
        pixels_per_cell,
        cells_per_block,
        block_norm,
        visualize,
        transform_sqrt,
        feature_vector);

    auto stop = high_resolution_clock::now();

    std::cout << res.first.rows * res.first.cols << std::endl;

    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken by function: "
         << (duration.count()) << " microseconds" << std::endl;


    std::cout << res.first.size << std::endl;
    std::cout << res.second.size << std::endl;


//    for (int i = 0; i < res.first.size[0]; ++i) {
//           std::cout << res.first.at<double>(i) << " ";
//    }

//    std::cout << res.first << std::endl;


    namedWindow("Image", cv::WINDOW_NORMAL); // Create a resizable window
    imshow("Image", res.second);

    // Wait for a key press
    waitKey(0);

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
