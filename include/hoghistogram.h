// hoghistogram.h
#include "../libs/Eigen/Dense"  // Include necessary headers

#ifndef HOGHISTOGRAM_H  // Include guards to prevent multiple inclusion
#define HOGHISTOGRAM_H


// Declare the function
void hog_histograms(const std::vector<std::vector<double>>& g_col,
                    const std::vector<std::vector<double>>& g_row,
                    int c_col,
                    int c_row,
                    int s_col,
                    int s_row,
                    int n_cells_col,
                    int n_cells_row,
                    int orientations,
                    std::vector<std::vector<std::vector<double>>>& orientation_histogram);


#endif  // HOGHISTOGRAM_H
