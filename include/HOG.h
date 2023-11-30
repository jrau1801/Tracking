//
// Created by Louis-Kaan Ay on 28.11.23.
//

#ifndef TRACKING_HOG_H
#define TRACKING_HOG_H

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
//#include <opencv2/core/parallel.hpp>
#include "../include/Line.h"
#include "../include/Gradient.h"


using cv::Mat;


class HOG {

public:
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> compute(const Eigen::MatrixXd &image,
                                                               int orientations = 9,
                                                               std::pair<int, int> pixels_per_cell = std::make_pair(8,
                                                                                                                    8),
                                                               std::pair<int, int> cells_per_block = std::make_pair(3,
                                                                                                                    3),
                                                               unsigned int method = cv::NORM_L2,
                                                               bool sobel = false,
                                                               bool visualize = false,
                                                               bool normalize_input = true,
                                                               bool flatten = true) {

        if (method != cv::NORM_L1 && method != cv::NORM_L2 && method != cv::NORM_L2SQR) {
            throw std::invalid_argument(
                    "The only permitted method values are cv::NORM_L1 (2), cv::NORM_L2 (4) or cv::NORM_L2SQR (5)!");
        }


        Mat mat_image;

        cv::eigen2cv(image, mat_image);

        Mat transformed_image = normalize_input ? normalizeImage(mat_image) : mat_image;

        auto pair = sobel ? Gradient::compute_gradients_sobel(transformed_image) : Gradient::compute_gradients_prewitt(
                transformed_image);

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


        int s_row = mat_image.rows;
        int s_col = mat_image.cols;

        int c_row = pixels_per_cell.first;
        int c_col = pixels_per_cell.second;

        int b_row = cells_per_block.first;
        int b_col = cells_per_block.second;

        int n_cells_row = s_row / c_row;  // number of cells along row-axis
        int n_cells_col = s_col / c_col;  // number of cells along col-axis

        // Assuming n_cells_row, n_cells_col, and orientations are known at runtime
        Mat orientation_histogram = Mat::zeros(n_cells_row, n_cells_col * orientations, CV_64F);


        computeHOGHistograms(g_col,
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

                        auto lines = Line::draw(
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
                                hog_image.at<double>(rr, cc) += orientation_histogram.at<double>(r,
                                                                                                 c * orientations + o);
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


        cv::Mat normalized_blocks;

        for (int r = 0; r < n_blocks_row; ++r) {
            for (int c = 0; c < n_blocks_col; ++c) {
                cv::Rect blockRect(c * orientations, r, b_col * orientations, b_row);
                cv::Mat block = orientation_histogram(blockRect);
                cv::normalize(block, block, 1.0, 0.0, method, 6, cv::noArray());
                normalized_blocks.push_back(block);
            }
        }


        if (flatten) {
            normalized_blocks = normalized_blocks.reshape(1, 1);
            cv::transpose(normalized_blocks, normalized_blocks);
        }

        Eigen::MatrixXd eigen_normalized_blocks, eigen_hog_image;

        cv::cv2eigen(normalized_blocks, eigen_normalized_blocks);
        cv::cv2eigen(hog_image, eigen_hog_image);

        return std::make_pair(eigen_normalized_blocks, eigen_hog_image);

    }

private:
    static Mat normalizeImage(const Mat &image) {
        Mat normalized_image;

        Mat converted_image;
        image.convertTo(converted_image, CV_64F); // Convert to double type
        cv::sqrt(converted_image, normalized_image);

        return normalized_image;
    }

    static double calculateCell(cv::Mat &magnitude,
                                cv::Mat &orientation,
                                double orientation_start,
                                double orientation_end,
                                int cell_columns,
                                int cell_rows,
                                int column_index,
                                int row_index,
                                int size_columns,
                                int size_rows,
                                int range_rows_start,
                                int range_rows_stop,
                                int range_columns_start,
                                int range_columns_stop) noexcept {

        int cell_column, cell_row, cell_row_index, cell_column_index;
        double total = 0.0;

        for (cell_row = range_rows_start; cell_row < range_rows_stop; ++cell_row) {
            cell_row_index = row_index + cell_row;
            if (cell_row_index < 0 || cell_row_index >= size_rows) {
                continue;
            }

            for (cell_column = range_columns_start; cell_column < range_columns_stop; ++cell_column) {
                cell_column_index = column_index + cell_column;
                if (cell_column_index < 0 || cell_column_index >= size_columns ||
                    orientation.at<double>(cell_row_index, cell_column_index) >= orientation_start ||
                    orientation.at<double>(cell_row_index, cell_column_index) < orientation_end) {
                    continue;
                }

                total += magnitude.at<double>(cell_row_index, cell_column_index);
            }
        }

        return total / static_cast<double>(cell_rows * cell_columns);
    }

// Function to compute hog_histograms
    static void computeHOGHistograms(cv::Mat &gradient_columns,
                                     cv::Mat &gradient_rows,
                                     int cell_columns, int cell_rows,
                                     int size_columns, int size_rows,
                                     int number_of_cells_columns, int number_of_cells_rows,
                                     int number_of_orientations,
                                     cv::Mat &orientation_histogram) {



        // Ensure both matrices have the same size and compatible types
        if (gradient_columns.size() != gradient_rows.size()) {
            throw std::invalid_argument("Gradient columns and rows matrices should have the same size.");
        }

// Convert matrices to compatible types (CV_32F or CV_64F) if needed
        if (gradient_columns.type() != CV_32F && gradient_columns.type() != CV_64F) {
            gradient_columns.convertTo(gradient_columns, CV_64F); // Change to CV_64F if needed
        }

        if (gradient_rows.type() != CV_32F && gradient_rows.type() != CV_64F) {
            gradient_rows.convertTo(gradient_rows, CV_64F); // Change to CV_64F if needed
        }

// Ensure both matrices have the same type after conversion
        if (gradient_columns.type() != gradient_rows.type()) {
            throw std::invalid_argument("Gradient columns and rows matrices should have the same type.");
        }


        cv::Mat magnitude;
        cv::magnitude(gradient_columns, gradient_rows, magnitude);

        cv::Mat orientation;
        cv::phase(gradient_columns, gradient_rows, orientation, true);

        int r_0 = cell_rows / 2;
        int c_0 = cell_columns / 2;
        int cc = cell_rows * number_of_cells_rows;
        int cr = cell_columns * number_of_cells_columns;
        int range_rows_stop = (cell_rows + 1) / 2;
        int range_rows_start = -(cell_rows / 2);
        int range_columns_stop = (cell_columns + 1) / 2;
        int range_columns_start = -(cell_columns / 2);
        double number_of_orientations_per_180 = 180.0 / number_of_orientations;

        // Compute orientations integral images
        cv::parallel_for_(cv::Range(0, number_of_orientations), [&](const cv::Range &range) {
                              for (int i = range.start; i < range.end; ++i) {
                                  double orientation_start = number_of_orientations_per_180 * (i + 1);
                                  double orientation_end = number_of_orientations_per_180 * i;
                                  int c = c_0;
                                  int r = r_0;
                                  int r_i = 0;
                                  int c_i = 0;

                                  while (r < cc) {
                                      c_i = 0;
                                      c = c_0;

                                      while (c < cr) {
                                          // Perform calculations similar to 'cell_hog' function
                                          // using the relevant values from magnitude, orientation, etc.
                                          // Update orientation_histogram accordingly

                                          orientation_histogram.at<double>(r_i, c_i * number_of_orientations + i) =
                                                  calculateCell(magnitude, orientation, orientation_start, orientation_end,
                                                                cell_columns, cell_rows, c, r,
                                                                size_columns, size_rows,
                                                                range_rows_start, range_rows_stop,
                                                                range_columns_start, range_columns_stop);

                                          c_i += 1;
                                          c += cell_columns;
                                      }

                                      r_i += 1;
                                      r += cell_rows;
                                  }
                              }
                          }
        );

    }


};

#endif //TRACKING_HOG_H