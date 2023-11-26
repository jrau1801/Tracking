#include <iostream>
#include <vector>
#include <cmath>
#include "../include/hoghistogram.h"
#include <opencv2/opencv.hpp>

// Function to compute cell_hog
double cell_hog(const std::vector<std::vector<double>>& magnitude,
                const std::vector<std::vector<double>>& orientation,
                double orientation_start, double orientation_end,
                int cell_columns, int cell_rows,
                int column_index, int row_index,
                int size_columns, int size_rows,
                int range_rows_start, int range_rows_stop,
                int range_columns_start, int range_columns_stop) {
    double total = 0.0;

    for (int cell_row = range_rows_start; cell_row < range_rows_stop; ++cell_row) {
        int cell_row_index = row_index + cell_row;
        if (cell_row_index < 0 || cell_row_index >= size_rows)
            continue;

        for (int cell_column = range_columns_start; cell_column < range_columns_stop; ++cell_column) {
            int cell_column_index = column_index + cell_column;
            if (cell_column_index < 0 || cell_column_index >= size_columns ||
                orientation[cell_row_index][cell_column_index] >= orientation_start ||
                orientation[cell_row_index][cell_column_index] < orientation_end)
                continue;

            total += magnitude[cell_row_index][cell_column_index];
        }
    }

    return total / (cell_rows * cell_columns);
}

// Function to compute hog_histograms
void hog_histograms(const std::vector<std::vector<double>>& g_col,
                    const std::vector<std::vector<double>>& g_row,
                    int c_col,
                    int c_row,
                    int s_col,
                    int s_row,
                    int n_cells_col,
                    int n_cells_row,
                    int orientations,
                    std::vector<std::vector<std::vector<double>>>& orientation_histogram) {

    std::vector<std::vector<double>> magnitude(s_row, std::vector<double>(s_col, 0.0));
    std::vector<std::vector<double>> orientation(s_row, std::vector<double>(s_col, 0.0));

    for (int i = 0; i < s_row; ++i) {
        for (int j = 0; j < s_col; ++j) {
            magnitude[i][j] = std::hypot(g_col[i][j], g_row[i][j]);
            orientation[i][j] = std::fmod(std::atan2(g_row[i][j], g_col[i][j]) * 180.0 / CV_PI, 180.0);
        }
    }

    int r_0 = c_row / 2;
    int c_0 = c_col / 2;
    int cc = c_row * n_cells_row;
    int cr = c_col * n_cells_col;
    int range_rows_stop = (c_row + 1) / 2;
    int range_rows_start = -(c_row / 2);
    int range_columns_stop = (c_col + 1) / 2;
    int range_columns_start = -(c_col / 2);
    double number_of_orientations_per_180 = 180.0 / orientations;

    for (int i = 0; i < orientations; ++i) {
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
                orientation_histogram[r_i][c_i][i] =
                        cell_hog(magnitude, orientation,
                                 orientation_start, orientation_end,
                                 c_col, c_row, c, r,
                                 s_col, s_row,
                                 range_rows_start, range_rows_stop,
                                 range_columns_start, range_columns_stop);
                c_i++;
                c += c_col;
            }
            r_i++;
            r += c_row;
        }
    }

//    for (const auto& row : orientation_histogram) {
//        for (const auto& col : row) {
//            for (double val : col) {
//                std::cout << val << " ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "r_0: " << r_0 << std::endl;
//    std::cout << "c_0: " << c_0 << std::endl;
//    std::cout << "cc: " << cc << std::endl;
//    std::cout << "cr: " << cr << std::endl;
//    std::cout << "range_rows_stop: " << range_rows_stop << std::endl;
//    std::cout << "range_rows_start: " << range_rows_start << std::endl;
//    std::cout << "range_columns_stop: " << range_columns_stop << std::endl;
//    std::cout << "range_columns_start: " << range_columns_start << std::endl;
//    std::cout << "number_of_orientations_per_180: " << number_of_orientations_per_180 << std::endl;
}

//int main() {
//    // Define input matrices (gradient_columns and gradient_rows)
//    // These are just placeholders; replace them with your actual data
//    std::vector<std::vector<double>> gradient_columns = {
//            {1.2, 2.5, 3.1},
//            {0.8, 1.9, 2.7},
//            {2.0, 3.2, 4.0}
//    };
//
//    std::vector<std::vector<double>> gradient_rows = {
//            {0.5, 1.8, 2.2},
//            {1.3, 2.7, 3.5},
//            {2.5, 3.9, 4.7}
//    };
//
//    // Define parameters for histogram computation
//    int cell_columns = 2;
//    int cell_rows = 2;
//    int size_columns = gradient_columns[0].size();
//    int size_rows = gradient_columns.size();
//    int number_of_cells_columns = size_columns / cell_columns;
//    int number_of_cells_rows = size_rows / cell_rows;
//    int number_of_orientations = 4;
//
//    // Prepare orientation_histogram matrix to store results
//    std::vector<std::vector<std::vector<double>>> orientation_histogram(
//            number_of_cells_rows,
//            std::vector<std::vector<double>>(
//                    number_of_cells_columns,
//                    std::vector<double>(number_of_orientations, 0.0)
//            )
//    );
//
//    // Call hog_histograms function
//    hog_histograms(gradient_columns, gradient_rows, cell_columns, cell_rows,
//                   size_columns, size_rows, number_of_cells_columns,
//                   number_of_cells_rows, number_of_orientations, orientation_histogram);
//
//    // Display the resulting orientation_histogram (for demonstration)
//    for (const auto& row : orientation_histogram) {
//        for (const auto& cell : row) {
//            for (const auto& value : cell) {
//                std::cout << value << " ";
//            }
//            std::cout << "| ";
//        }
//        std::cout << std::endl;
//    }
//
//    return 0;
//}
