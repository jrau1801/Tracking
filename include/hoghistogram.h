// hoghistogram.h
#include "../libs/Eigen/Dense"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifndef HOGHISTOGRAM_H  // Include guards to prevent multiple inclusion
#define HOGHISTOGRAM_H


// Declare the function
//void hog_histograms(const Eigen::MatrixXd& g_col,
//                    const Eigen::MatrixXd& g_row,
//                    int c_col,
//                    int c_row,
//                    int s_col,
//                    int s_row,
//                    int n_cells_col,
//                    int n_cells_row,
//                    int orientations,
//                    Eigen::MatrixXd& orientation_histogram);

void hog_histograms(cv::Mat& gradient_columns,
                    cv::Mat& gradient_rows,
                    int cell_columns, int cell_rows,
                    int size_columns, int size_rows,
                    int number_of_cells_columns, int number_of_cells_rows,
                    int number_of_orientations,
                    cv::Mat& orientation_histogram);

#endif  // HOGHISTOGRAM_H
