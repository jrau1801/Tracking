//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_IMAGEPYRAMID_H
#define TRACKING_IMAGEPYRAMID_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>


/**
 * @brief Represents functionality for generating an image pyramid.
 */
class ImagePyramid {
public:

    ImagePyramid() = default;

    ImagePyramid(const ImagePyramid &) = delete;

    ImagePyramid(ImagePyramid &&) = delete;

    ImagePyramid &operator=(const ImagePyramid &) = delete;

    ImagePyramid &operator=(ImagePyramid &&) = delete;

    ~ImagePyramid() = default;

    /**
    * @brief Generates an image pyramid from the input frame.
    *
    * @param frame Input frame.
    * @param numScales Number of scales in the pyramid.
    * @param downscale Downscale factor.
    * @param minSize Minimum size of the pyramid.
    * @return Vector containing images in the pyramid.
    */
    static std::vector<cv::Mat>
    generate(const cv::Mat &frame, const size_t numScales, const double downscale, const cv::Size minSize) {
        std::vector<cv::Mat> pyramid;
        cv::Mat scaledFrame = frame.clone();
        for (int i = 0; i < numScales; ++i) {
            if (minSize.height > scaledFrame.rows || minSize.width > scaledFrame.cols) {
                break;
            }
            pyramid.push_back(scaledFrame.clone());
            cv::resize(scaledFrame, scaledFrame, cv::Size(), 1 / downscale,
                       1 / downscale);
        }
        return pyramid;
    }
};


#endif //TRACKING_IMAGEPYRAMID_H
