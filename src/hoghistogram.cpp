#include "../include/hoghistogram.h"
#include <iostream>
// Function to compute cell_hog
double cell_hog(cv::Mat &magnitude,
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
void hog_histograms(cv::Mat &gradient_columns,
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
    for (int i = 0; i < number_of_orientations; ++i) {
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
                        cell_hog(magnitude, orientation, orientation_start, orientation_end,
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

