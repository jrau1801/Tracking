//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_HOGDESCRIPTOR_H
#define TRACKING_HOGDESCRIPTOR_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "../include/HOG.h"

/**
 * @brief Represents functionality for computing HOG descriptors.
 */
class HOGDescriptor {

private:
    int orientations;
    std::pair<int, int> pixels_per_cell;
    std::pair<int, int> cells_per_block;
    unsigned int method;
    bool sobel;
    bool visualize;
    bool normalize_input;
    bool flatten;


public:

    HOGDescriptor(int orientations, const std::pair<int, int> &pixelsPerCell, const std::pair<int, int> &cellsPerBlock,
                  unsigned int method, bool sobel, bool visualize, bool normalizeInput, bool flatten) : orientations(
            orientations), pixels_per_cell(pixelsPerCell), cells_per_block(cellsPerBlock), method(method), sobel(sobel),
                                                                                                        visualize(
                                                                                                                visualize),
                                                                                                        normalize_input(
                                                                                                                normalizeInput),
                                                                                                        flatten(flatten) {}

    HOGDescriptor(const HOGDescriptor &) = default;

    HOGDescriptor(HOGDescriptor &&) = delete;

    HOGDescriptor &operator=(const HOGDescriptor &) = delete;

    HOGDescriptor &operator=(HOGDescriptor &&) = delete;

    ~HOGDescriptor() = default;

    /**
     * @brief Computes Histogram of Oriented Gradients (HOG) descriptors for the given image.
     *
     * @param image Input image.
     * @return Matrix containing HOG descriptors.
     */
    Eigen::MatrixXd compute(const Eigen::MatrixXd& image) {
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res = HOG::compute(image, orientations, pixels_per_cell,
                                                                       cells_per_block,
                                                                       method, sobel, visualize, normalize_input,
                                                                       flatten);
        return res.first;
    }


    /** Getter Methods **/

    int getOrientations() const {
        return orientations;
    }

    const std::pair<int, int> &getPixelsPerCell() const {
        return pixels_per_cell;
    }

    const std::pair<int, int> &getCellsPerBlock() const {
        return cells_per_block;
    }

    unsigned int getMethod() const {
        return method;
    }

    bool isSobel() const {
        return sobel;
    }

    bool isVisualize() const {
        return visualize;
    }

    bool isNormalizeInput() const {
        return normalize_input;
    }

    bool isFlatten() const {
        return flatten;
    }


};


#endif //TRACKING_HOGDESCRIPTOR_H
