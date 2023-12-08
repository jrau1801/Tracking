//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_NONMAXSUPPRESSION_H
#define TRACKING_NONMAXSUPPRESSION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <numeric>


/**
 * @brief Represents functionality for non-maximum suppression.
 */
class NonMaxSuppression {
public:

    NonMaxSuppression() = default;

    NonMaxSuppression(const NonMaxSuppression &) = delete;

    NonMaxSuppression(NonMaxSuppression &&) = delete;

    NonMaxSuppression &operator=(const NonMaxSuppression &) = delete;

    NonMaxSuppression &operator=(NonMaxSuppression &&) = delete;

    ~NonMaxSuppression() = default;

    /**
     * @brief Suppresses overlapping rectangles based on given scores.
     *
     * @param rects Vector of rectangles.
     * @param scores Scores associated with each rectangle.
     * @param overlapThresh Threshold for overlap suppression.
     * @return Vector containing non-maximally suppressed rectangles.
     */
    static std::vector<cv::Rect>
    suppress(const std::vector<cv::Rect> &rects, const std::vector<float> &scores, float overlapThresh) {
        std::vector<cv::Rect> picked;
        std::vector<int> indices(rects.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });

        while (!indices.empty()) {
            int current_idx = indices.front();
            picked.push_back(rects[current_idx]);
            indices.erase(indices.begin());

            std::vector<int> indices_to_remove;
            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                float intersection_area = (rects[current_idx] & rects[idx]).area();
                float union_area = rects[current_idx].area() + rects[idx].area() - intersection_area;
                float overlap = intersection_area / union_area;

                if (overlap > overlapThresh)
                    indices_to_remove.push_back(i);
            }

            for (int i = indices_to_remove.size() - 1; i >= 0; --i) {
                indices.erase(indices.begin() + indices_to_remove[i]);
            }
        }

        return picked;
    }
};


#endif //TRACKING_NONMAXSUPPRESSION_H
