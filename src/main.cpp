#include <chrono>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "../include/HOG.h"

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

    Mat image = cv::imread(R"(/Users/Louis/CLionProjects/Tracking/images/frame_2.png)");

    if (image.empty()) {
        cerr << "Error: Couldn't load the image!" << std::endl;
        return -1;
    }

    Mat rImg;
//    cv::resize(image, rImg, cv::Size(), 0.25, 0.25);


    cv::Mat grayImage;
    cv::Mat grayImage64;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    grayImage.convertTo(grayImage64, CV_64F);

    int orientations = 9;
    std::pair<int, int> pixels_per_cell = std::make_pair(8, 8);
    std::pair<int, int> cells_per_block = std::make_pair(2, 2);
    cv::NormTypes method = cv::NORM_L2;
    bool sobel = true;
    bool visualize = true;
    bool normalize_input = true;
    bool feature_vector = true;

    Eigen::MatrixXd input;
    cv::cv2eigen(grayImage64, input);

    auto start = high_resolution_clock::now();
    auto res = HOG::compute(input,
        orientations,
        pixels_per_cell,
        cells_per_block,
        method,
        sobel,
        visualize,
        normalize_input,
        feature_vector);

    auto stop = high_resolution_clock::now();

    Mat features, hog_image;
    cv::eigen2cv(res.first, features);
    cv::eigen2cv(res.second, hog_image);

    std::cout << features.rows * features.cols << std::endl;

    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken by function: "
         << (duration.count()) << " microseconds" << std::endl;



//    for (int i = 0; i < res.first.size[0]; ++i) {
//           std::cout << res.first.at<double>(i) << " ";
//    }

//    std::cout << res.first << std::endl;


    namedWindow("Image", cv::WINDOW_NORMAL); // Create a resizable window
    imshow("Image", hog_image);

    // Wait for a key press
    waitKey(0);

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
