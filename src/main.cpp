#include <opencv2/opencv.hpp>
#include "../include/hog.h"
#include <opencv2/core/eigen.hpp>
#include "../libs/Eigen/Dense"

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

    Mat image = cv::imread(R"(C:\Users\janra\CLionProjects\Tracking\images\frame_15344.jpg)");

    if (image.empty()) {
        cerr << "Error: Couldn't load the image!" << std::endl;
        return -1;
    }

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    Eigen::MatrixXd converted_image;

    cv::cv2eigen(grayImage, converted_image);


    int orientations = 9;
    std::pair<int, int> pixels_per_cell = std::make_pair(8, 8);
    std::pair<int, int> cells_per_block = std::make_pair(3, 3);
    std::string block_norm = "L2-Hys";
    bool visualize = false;
    bool transform_sqrt = true;
    bool feature_vector = true;

    cv::Mat finished_hog_image = hog(converted_image,
        orientations,
        pixels_per_cell,
        cells_per_block,
        block_norm,
        visualize,
        transform_sqrt,
        feature_vector);


    namedWindow("Image", cv::WINDOW_NORMAL); // Create a resizable window
    imshow("Image", finished_hog_image);

    // Wait for a key press
    waitKey(0);

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
