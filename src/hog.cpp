#include <iostream>
#include <algorithm>
#include "../libs/Eigen/Dense"
#include "../include/hoghistogram.h"
#include "../include/draw_new.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


using cv::Mat;


double calculateSumValue(const Mat &blockArray, const std::string &method) {
    double sumValue = 0.0;
    if (method == "L1" || method == "L1-sqrt") {
        sumValue = cv::sum(cv::abs(blockArray))[0];
    } else if (method == "L2" || method == "L2-Hys") {
        sumValue = cv::sum(blockArray.mul(blockArray))[0];
    } else {
        throw std::invalid_argument("Selected block normalization method is invalid.");
    }
    return sumValue;
}

void normalizeL1(const Mat &blockArray, double sumValue, Mat &normalized_block, double eps) {
    normalized_block = blockArray / std::max(sumValue + eps, std::numeric_limits<double>::epsilon());
}

void normalizeL1Sqrt(const Mat &blockArray, double sumValue, Mat &normalized_block, double eps) {
    Mat temp_normalized_block;
    cv::sqrt(blockArray / std::max(sumValue + eps, std::numeric_limits<double>::epsilon()), temp_normalized_block);
    temp_normalized_block.convertTo(normalized_block, CV_64F); // Convert back to double type
}

void normalizeL2(const Mat &blockArray, double sumValue, Mat &normalized_block, double eps) {
    cv::sqrt(blockArray / std::max(sumValue + eps * eps, std::numeric_limits<double>::epsilon()), normalized_block);
}

void normalizeL2Hys(const Mat &blockArray, double sumValue, Mat &normalized_block, double eps) {
    Mat out;
    cv::sqrt(blockArray / std::max(sumValue + eps * eps, std::numeric_limits<double>::epsilon()), out);
    cv::min(out, 0.2, normalized_block);
    cv::sqrt(normalized_block.mul(normalized_block) /
             max((normalized_block.mul(normalized_block) + eps * eps), std::numeric_limits<double>::epsilon()),
             normalized_block);
}

Mat normalizeBlock(const Mat &block, const std::string &method, double eps = 1e-6) {
    Mat blockArray;
    block.convertTo(blockArray, CV_64F); // Convert block to double

    double sumValue = calculateSumValue(blockArray, method);

    Mat normalized_block;

    if (method == "L1") {
        normalizeL1(blockArray, sumValue, normalized_block, eps);
    } else if (method == "L1-sqrt") {
        normalizeL1Sqrt(blockArray, sumValue, normalized_block, eps);
    } else if (method == "L2") {
        normalizeL2(blockArray, sumValue, normalized_block, eps);
    } else if (method == "L2-Hys") {
        normalizeL2Hys(blockArray, sumValue, normalized_block, eps);
    } else {
        throw std::invalid_argument("Selected block normalization method is invalid.");
    }

    return normalized_block;
}


// Define the equivalent function for _hog_channel_gradient in C++

//std::pair<cv::Mat, cv::Mat> compute_gradients(const cv::Mat &channel) {
//    Eigen::MatrixXd eigenChannel;
//    cv::cv2eigen(channel, eigenChannel); // Convert cv::Mat to Eigen::MatrixXd
//
//    Eigen::MatrixXd g_row(eigenChannel.rows(), eigenChannel.cols());
//    Eigen::MatrixXd g_col(eigenChannel.rows(), eigenChannel.cols());
//
//    g_row.row(0).setZero();
//    g_row.row(g_row.rows() - 1).setZero();
//    g_row.middleRows(1, g_row.rows() - 2) = eigenChannel.block(2, 0, eigenChannel.rows() - 2, eigenChannel.cols()) -
//                                            eigenChannel.block(0, 0, eigenChannel.rows() - 2, eigenChannel.cols());
//
//    g_col.col(0).setZero();
//    g_col.col(g_col.cols() - 1).setZero();
//    g_col.middleCols(1, g_col.cols() - 2) = eigenChannel.block(0, 2, eigenChannel.rows(), eigenChannel.cols() - 2) -
//                                            eigenChannel.block(0, 0, eigenChannel.rows(), eigenChannel.cols() - 2);
//
//    cv::Mat gRowMat, gColMat;
//    cv::eigen2cv(g_row, gRowMat); // Convert Eigen::MatrixXd to cv::Mat
//    cv::eigen2cv(g_col, gColMat); // Convert Eigen::MatrixXd to cv::Mat
//
//    return std::make_pair(gRowMat, gColMat);
//}

//std::pair<cv::Mat, cv::Mat> compute_gradients(const cv::Mat &channel) {
//    cv::Mat grad_x, grad_y;
//
//    // Compute gradients using Sobel in the x and y directions
//    cv::Sobel(channel, grad_x, CV_64F, 1, 0);
//    cv::Sobel(channel, grad_y, CV_64F, 0, 1);
//
//    return std::make_pair(grad_y, grad_x);
//}


std::pair<cv::Mat, cv::Mat> compute_gradients(const cv::Mat &inputImage) {
    cv::Mat grad_x, grad_y;

    // Compute gradients using Prewitt operator
    cv::Mat kernel_x = (cv::Mat_<double>(3, 3) << -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<double>(3, 3) << -1, -1, -1,
            0,  0,  0,
            1,  1,  1);

    cv::filter2D(inputImage, grad_x, -1, kernel_x);
    cv::filter2D(inputImage, grad_y, -1, kernel_y);

    return std::make_pair(grad_y, grad_x);
}




cv::Mat normalize_image(const Mat &image) {
    Mat normalized_image;

    Mat converted_image;
    image.convertTo(converted_image, CV_64F); // Convert to double type
    cv::sqrt(converted_image, normalized_image);

    return normalized_image;
}

std::pair<cv::Mat, Mat> hog(const Mat &image,
                            int orientations = 9,
                            std::pair<int, int> pixels_per_cell = std::make_pair(8, 8),
                            std::pair<int, int> cells_per_block = std::make_pair(3, 3),
                            std::string method = "L2-Hys",
                            bool visualize = false,
                            bool transform_sqrt = true,
                            bool flatten = true) {

    Mat transformed_image = transform_sqrt ? normalize_image(image) : image;

    auto pair = compute_gradients(transformed_image);
    Mat g_row = pair.first;
    Mat g_col = pair.second;

    if (g_row.empty() || g_col.empty()) {
        throw std::runtime_error("Computed gradient matrices are empty!");
    }

    if (g_row.type() != CV_64F) {
        g_row.convertTo(g_row, CV_64F);
    }

    if (g_col.type() != CV_64F) {
        g_col.convertTo(g_col, CV_64F);
    }

    if (g_row.size() != g_col.size()) {
        throw std::runtime_error("Gradient matrices have mismatched sizes!");
    }


    int s_row = image.rows;
    int s_col = image.cols;

    int c_row = pixels_per_cell.first;
    int c_col = pixels_per_cell.second;

    int b_row = cells_per_block.first;
    int b_col = cells_per_block.second;

    int n_cells_row = s_row / c_row;  // number of cells along row-axis
    int n_cells_col = s_col / c_col;  // number of cells along col-axis

    // Assuming n_cells_row, n_cells_col, and orientations are known at runtime
    Mat orientation_histogram = Mat::zeros(n_cells_row, n_cells_col * orientations, CV_64F);


    hog_histograms(g_col,
                   g_row,
                   c_col,
                   c_row,
                   s_col,
                   s_row,
                   n_cells_col,
                   n_cells_row,
                   orientations,
                   orientation_histogram);

    int radius = std::min(c_row, c_col) / 2 - 1;

    std::vector<int> orientations_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientations_arr[i] = i;
    }

    std::vector<double> orientation_bin_midpoints(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientation_bin_midpoints[i] = CV_PI * (i + 0.5) / orientations;
    }

    std::vector<double> dr_arr(orientations);
    std::vector<double> dc_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        dr_arr[i] = radius * std::sin(orientation_bin_midpoints[i]);
        dc_arr[i] = radius * std::cos(orientation_bin_midpoints[i]);
    }

    Mat hog_image = Mat::zeros(s_row, s_col, CV_64F);

    if (visualize) {
        for (int r = 0; r < n_cells_row; ++r) {
            for (int c = 0; c < n_cells_col; ++c) {
                for (int i = 0; i < orientations_arr.size(); ++i) {
                    int o = orientations_arr[i];
                    double dr = dr_arr[i];
                    double dc = dc_arr[i];

                    std::pair<double, double> centre = std::make_pair(r * c_row + c_row / 2, c * c_col + c_col / 2);


                    // Assuming `line` returns a pair of integers as rr and cc
                    auto lines = line(
                            static_cast<int>(centre.first - dc),
                            static_cast<int>(centre.second + dr),
                            static_cast<int>(centre.first + dc),
                            static_cast<int>(centre.second - dr)
                    );


                    auto &rows_array = lines.first;
                    auto &cols_array = lines.second;

                    for (int g = 0; g < rows_array.size(); ++g) {
                        int rr = rows_array[g];
                        int cc = cols_array[g];

                        // Ensure rr and cc are within bounds before accessing hog_image
                        if (rr >= 0 && rr < s_row && cc >= 0 && cc < s_col) {
                            hog_image.at<double>(rr, cc) += orientation_histogram.at<double>(r, c * orientations + o);
                        }
                    }
                }
            }
        }
    }


    int n_blocks_row = (n_cells_row - b_row) + 1;
    int n_blocks_col = (n_cells_col - b_col) + 1;

    if (n_blocks_col <= 0 || n_blocks_row <= 0) {
        int min_row = b_row * c_row;
        int min_col = b_col * c_col;
        throw std::invalid_argument(
                "The input image is too small given the values of pixels_per_cell and cells_per_block. "
                "It should have at least: " + std::to_string(min_row) + " rows and " +
                std::to_string(min_col) + " cols.");
    }

    int total_blocks = orientation_histogram.rows * orientation_histogram.cols / orientations;

// Reshape the orientation histogram to separate each block
    cv::Mat normalized_blocks = orientation_histogram.reshape(1, total_blocks);

// Apply normalization to each block
    for (int i = 0; i < total_blocks; ++i) {
        cv::normalize(normalized_blocks.row(i), normalized_blocks.row(i), 1.0, 0.0, cv::NORM_L2, 6, cv::noArray());
    }

//
//    Mat normalized_blocks;
//
//
//
//    for (int r = 0; r < n_blocks_row; ++r) {
//        for (int c = 0; c < n_blocks_col; ++c) {
//            // Calculate the block range in the orientation histogram
//            Mat block = orientation_histogram.rowRange(r, r + b_row).colRange(c * orientations, (c + 1) * orientations);
//
//            normalized_blocks.push_back(normalizeBlock(block, method));
//        }
//    }


    if (flatten) {
        Mat flattened_feature_vector = normalized_blocks.reshape(1);
    }


    return std::make_pair(normalized_blocks, hog_image);


}


