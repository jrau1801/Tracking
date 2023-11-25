#include <iostream>
#include "../libs/Eigen/Dense"
#include "../include/hoghistogram.h"
#include "../include/draw_new.h"

// Function to convert Eigen::MatrixXd to const std::vector<std::vector<double>>&
const std::vector<std::vector<double>> &EigenMatrixToVector(const Eigen::MatrixXd &matrix) {
    static std::vector<std::vector<double>> data;
    data.resize(matrix.rows(), std::vector<double>(matrix.cols()));

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            data[i][j] = matrix(i, j);
        }
    }

    return data;
}

void printVector(const std::vector<std::vector<double>> &vec) {
    for (const auto &row: vec) {
        for (const auto &val: row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Define the equivalent function for _hog_normalize_block in C++
Eigen::MatrixXd hog_normalize_block(const Eigen::MatrixXd &block, const std::string &method, double eps = 1e-5) {
    Eigen::MatrixXd out;
    if (method == "L1") {
        out = block.array() / (block.array().abs().sum() + eps);
    } else if (method == "L1-sqrt") {
        out = (block.array() / (block.array().abs().sum() + eps)).sqrt();
    } else if (method == "L2") {
        out = (block.array() / (block.array().square().sum() + eps * eps)).sqrt();
    } else if (method == "L2-Hys") {
        out = (block.array() / (block.array().square().sum() + eps * eps)).sqrt();
        out = out.cwiseMin(0.2);
        out = (out.array() / (out.array().square().sum() + eps * eps)).sqrt();
    } else {
        throw std::invalid_argument("Selected block normalization method is invalid.");
    }
    return out;
}

// Define the equivalent function for _hog_channel_gradient in C++
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hog_channel_gradient(const Eigen::MatrixXd &channel) {
    Eigen::MatrixXd g_row(channel.rows(), channel.cols());
    Eigen::MatrixXd g_col(channel.rows(), channel.cols());

    g_row.row(0).setZero();
    g_row.row(g_row.rows() - 1).setZero();
    g_row.middleRows(1, g_row.rows() - 2) = channel.block(2, 0, channel.rows() - 2, channel.cols()) -
                                            channel.block(0, 0, channel.rows() - 2, channel.cols());

    g_col.col(0).setZero();
    g_col.col(g_col.cols() - 1).setZero();
    g_col.middleCols(1, g_col.cols() - 2) = channel.block(0, 2, channel.rows(), channel.cols() - 2) -
                                            channel.block(0, 0, channel.rows(), channel.cols() - 2);

    return {g_row, g_col};
}

void hog(const Eigen::MatrixXd &image,
         int orientations = 9,
         std::pair<int, int> pixels_per_cell = std::make_pair(5, 5),
         std::pair<int, int> cells_per_block = std::make_pair(1, 1),
         std::string block_norm = "L2-Hys",
         bool visualize = false,
         bool transform_sqrt = false,
         bool feature_vector = true) {

    Eigen::MatrixXd transformed_image = image;
    if (transform_sqrt) {
        transformed_image = transformed_image.array().sqrt();
    }

    auto pair = hog_channel_gradient(transformed_image);
    Eigen::MatrixXd g_row = pair.first;
    Eigen::MatrixXd g_col = pair.second;

    int s_row = image.rows();
    int s_col = image.cols();
    int c_row = pixels_per_cell.first;
    int c_col = pixels_per_cell.second;
    int b_row = cells_per_block.first;
    int b_col = cells_per_block.second;

    int n_cells_row = s_row / c_row;  // number of cells along row-axis
    int n_cells_col = s_col / c_col;  // number of cells along col-axis

// Assuming n_cells_row, n_cells_col, and orientations are known at runtime
    std::vector<std::vector<std::vector<double>>> orientation_histogram(
            n_cells_row,
            std::vector<std::vector<double>>(
                    n_cells_col,
                    std::vector<double>(orientations, 0.0)
            )
    );

    std::vector<std::vector<double>> g_row_new = EigenMatrixToVector(g_row);
    std::vector<std::vector<double>> g_col_new = EigenMatrixToVector(g_col);

    // printVector(g_row_new);

    hog_histograms(g_col_new,
                   g_row_new,
                   c_col,
                   c_row,
                   s_col,
                   s_row,
                   n_cells_col,
                   n_cells_row,
                   orientations,
                   orientation_histogram);

    // printVector(g_row_new);



    int radius = std::min(c_row, c_col) / 2 - 1;

    std::vector<int> orientations_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientations_arr[i] = i;
    }

    std::vector<double> orientation_bin_midpoints(orientations);
    for (int i = 0; i < orientations; ++i) {
        orientation_bin_midpoints[i] = M_PI * (i + 0.5) / orientations;
    }

    std::vector<double> dr_arr(orientations);
    std::vector<double> dc_arr(orientations);
    for (int i = 0; i < orientations; ++i) {
        dr_arr[i] = radius * std::sin(orientation_bin_midpoints[i]);
        dc_arr[i] = radius * std::cos(orientation_bin_midpoints[i]);
    }


    std::vector<std::vector<double>> hog_image(s_row, std::vector<double>(s_col, 0.0));

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

                for (size_t i = 0; i < rows_array.size(); ++i) {
                    int rr = rows_array[i];
                    int cc = cols_array[i];

                    // Ensure rr and cc are within bounds before accessing hog_image
                    if (rr >= 0 && rr < s_row && cc >= 0 && cc < s_col) {
                        hog_image[rr][cc] += orientation_histogram[r][c][o];
                    }

                }
            }
        }
    }

    // Printing the contents of the 2D vector
    for (int i = 0; i < s_row; ++i) {
        for (int j = 0; j < s_col; ++j) {
            std::cout << hog_image[i][j] << " ";
        }
        std::cout << std::endl;
    }

    int n_blocks_row = (n_cells_row - b_row) + 1;
    int n_blocks_col = (n_cells_col - b_col) + 1;



    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> normalized_blocks(
            n_blocks_row,
            std::vector<std::vector<std::vector<std::vector<double>>>>(
                    n_blocks_col,
                    std::vector<std::vector<std::vector<double>>>(
                            b_row,
                            std::vector<std::vector<double>>(
                                    b_col,
                                    std::vector<double>(orientations, 0.0)
                            )
                    )
            )
    );



}


int main() {

    Eigen::MatrixXd image(5, 5);
    image << 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25;
    hog(image);

//    // Example usage of hog_normalize_block
//    Eigen::MatrixXd block(3, 3);
//    block << 1, 2, 3,
//            4, 5, 6,
//            7, 8, 9;
//
//    Eigen::MatrixXd normalized_block = hog_normalize_block(block, "L2-Hys");
//
//    // Example usage of hog_channel_gradient
//    Eigen::MatrixXd channel(4, 4);
//    channel << 1, 2, 3, 4,
//            5, 6, 7, 8,
//            9, 10, 11, 12,
//            13, 14, 15, 16;
//
//    auto gradient = hog_channel_gradient(channel);
//    Eigen::MatrixXd g_row = gradient.first;
//    Eigen::MatrixXd g_col = gradient.second;

    // Output or further processing...
//    std::cout << "Normalized Block:\n" << normalized_block << std::endl;
//    std::cout << "Gradient Row:\n" << g_row << std::endl;
//    std::cout << "Gradient Column:\n" << g_col << std::endl;

    return 0;
}