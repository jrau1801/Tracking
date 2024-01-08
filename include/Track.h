//
// Created by Louis-Kaan Ay on 19.12.23.
//

#ifndef TRACKING_TRACK_H
#define TRACKING_TRACK_H

#include <iostream>
#include <vector>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

class Track {
private:
    int id; // Unique identifier for the track
    cv::Rect boundingBox; // Bounding box of the detected person
    cv::KalmanFilter kf;
    double confidenceScore; // Confidence score of the detection
    std::vector<cv::Rect> history; // History of bounding boxes with timestamps
    cv::Point2f velocity; // Velocity of the tracked object
    cv::Point2f acceleration; // Acceleration of the tracked object
    bool isActive; // Track status
    unsigned int age;

    // For later remembering
    cv::Mat hogFeature;
    cv::Mat rgbHistogram;


public:
    // Constructor to initialize Track object
    Track(int trackId, const cv::Rect &bbox, double confidence)
            : id(trackId), boundingBox(bbox), confidenceScore(confidence), isActive(true), age(1) {
        history.emplace_back(bbox); // Add initial bounding box with timestamp to history

        // Initialize Kalman filter for this track
        kf.init(4, 2, 0); // State: [x, y, dx/dt, dy/dt], Measurement: [x, y]
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,     0, 1, 0, 1,     0, 0, 1, 0,    0, 0, 0, 1);
        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    }

    void predictNewPosition() {
        cv::Mat prediction = kf.predict();
        boundingBox.x = static_cast<int>(prediction.at<float>(0));
        boundingBox.y = static_cast<int>(prediction.at<float>(1));
    }


    // Getters and setters for class attributes
    int getId() const {
        return id;
    }

    cv::Rect getBoundingBox() const {
        return boundingBox;
    }

    void setBoundingBox(const cv::Rect &bbox) {
        this->boundingBox = bbox;
        this->history.push_back(bbox);
    }

    double getConfidenceScore() const {
        return confidenceScore;
    }

    void setConfidenceScore(double confidenceScore) {
        this->confidenceScore = confidenceScore;
    }

    const std::vector<cv::Rect> &getHistory() const {
        return history;
    }


    const cv::Point2f &getVelocity() const {
        return velocity;
    }

    void setVelocity(const cv::Point2f &velocity) {
        this->velocity = velocity;
    }

    const cv::Point2f &getAcceleration() const {
        return acceleration;
    }

    void setAcceleration(const cv::Point2f &acceleration) {
        this->acceleration = acceleration;
    }

    bool isActive1() const {
        return isActive;
    }

    void setIsActive(bool isActive) {
        this->isActive = isActive;
    }

    unsigned int getAge() const {
        return age;
    }

    void setAge(unsigned int age) {
        this->age = age;
    }

};


#endif //TRACKING_TRACK_H
