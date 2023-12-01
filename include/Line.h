//
// Created by Louis-Kaan Ay on 28.11.23.
//

#ifndef TRACKING_LINE_H
#define TRACKING_LINE_H

#include <vector>
#include <cmath>

class Line {

private:
    static std::pair<std::vector<int>, std::vector<int>>
    draw_lines(int start_row, int start_col, int end_row, int end_col) {
        bool is_steep = false;
        int current_row = start_row;
        int current_col = start_col;
        int delta_row = std::abs(end_row - start_row);
        int delta_col = std::abs(end_col - start_col);
        int step_row = 0, step_col = 0, delta = 0;

        int max_delta = std::max(delta_col, delta_row);
        std::vector<int> rows_array(max_delta + 1, 0);
        std::vector<int> cols_array(max_delta + 1, 0);

        if (end_col - current_col > 0) {
            step_col = 1;
        } else {
            step_col = -1;
        }

        if (end_row - current_row > 0) {
            step_row = 1;
        } else {
            step_row = -1;
        }

        if (delta_row > delta_col) {
            is_steep = true;
            std::swap(current_col, current_row);
            std::swap(delta_col, delta_row);
            std::swap(step_col, step_row);
        }

        delta = 2 * delta_row - delta_col;

        for (int i = 0; i < delta_col; ++i) {
            if (is_steep) {
                rows_array[i] = current_col;
                cols_array[i] = current_row;
            } else {
                rows_array[i] = current_row;
                cols_array[i] = current_col;
            }

            while (delta >= 0) {
                current_row += step_row;
                delta -= 2 * delta_col;
            }

            current_col += step_col;
            delta += 2 * delta_row;
        }

        rows_array[delta_col] = end_row;
        cols_array[delta_col] = end_col;


        return std::make_pair(rows_array, cols_array);
    }

public:
    static std::pair<std::vector<int>, std::vector<int>> draw(int r0, int c0, int r1, int c1) {
        return draw_lines(r0, c0, r1, c1);
    }

};


#endif //TRACKING_LINE_H
