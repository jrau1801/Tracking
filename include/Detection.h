//
// Created by Louis-Kaan Ay on 30.11.23.
//

#ifndef TRACKING_DETECTION_H
#define TRACKING_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ostream>

class Detection {

public:
    Detection(cv::Rect boundingBox, double score) : boundingBox(boundingBox), score(score) {}


    // Copy Constructor
    Detection(const Detection &d) {
        this->boundingBox = d.boundingBox;
        this->score = d.score;
    }

    ~Detection() = default;

    const cv::Rect &getBoundingBox() const {
        return boundingBox;
    }

    double getScore() const {
        return score;
    }

    Detection& operator=(Detection&& d) noexcept {
        if (this != &d) {
            this->boundingBox = d.boundingBox;
            this->score = d.score;
        }
        return *this;
    }

    bool operator<(const Detection &d) const {
        return this->score < d.score;
    }

    bool operator>(const Detection &d) const {
        return this->score > d.score;
    }

    bool operator<=(const Detection &d) const {
        return this->score <= d.score;
    }

    bool operator>=(const Detection &d) const {
        return this->score >= d.score;
    }

        friend std::ostream &operator<<(std::ostream &os, const Detection &detection) {
        os << "boundingBox: " << detection.boundingBox << " score: " << detection.score;
        return os;
    }

    bool operator==(const Detection &rhs) const {
        return std::tie(boundingBox, score) == std::tie(rhs.boundingBox, rhs.score);
    }

    bool operator!=(const Detection &rhs) const {
        return !(rhs == *this);
    }


private:
    cv::Rect boundingBox;
    double score;
};


#endif //TRACKING_DETECTION_H
