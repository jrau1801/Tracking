//
// Created by Louis-Kaan Ay on 08.12.23.
//

#ifndef TRACKING_PERSONDETECTOR_H
#define TRACKING_PERSONDETECTOR_H

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <numeric>
#include <opencv2/bgsegm.hpp>

#include "../include/HOGDescriptor.h"
#include "../include/SlidingWindow.h"
#include "../include/NonMaxSuppression.h"
#include "../include/ImagePyramid.h"

class PersonDetector {
private:
    cv::Ptr<cv::ml::SVM> svm_model_1;
    cv::Ptr<cv::ml::SVM> svm_model_2;

    HOGDescriptor hogDescriptor;

    const double scale_factor;


    const cv::Size window_size;
    const cv::Size stepSize;
    const double detection_threshold_1;
    const double detection_threshold_2;
    const float overlap_threshold;
    const double downscale;

    void loadModels(const std::string &svm_model_1_path, const std::string &svm_model_2_path) {
        try {
            svm_model_1 = cv::ml::SVM::load(
                    svm_model_1_path);

        } catch (cv::Exception &e) {
            std::cerr << "There was an error while loading the svm_model_1\n" << e.what() << std::endl;
        }

        try {
            svm_model_2 = cv::ml::SVM::load(
                    svm_model_2_path);
        } catch (cv::Exception &e) {
            std::cerr << "There was an error while loading the svm_model_2\n" << e.what() << std::endl;
        }
    }

public:

    PersonDetector(const std::string &svm_model_1_path, const std::string &svm_model_2_path, HOGDescriptor &hogDescriptor,
                   const double scaleFactor,
                   const cv::Size &windowSize, const cv::Size &stepSize, const double detectionThreshold1,
                   const double detectionThreshold2, const float overlapThreshold, const double downscale) :
            hogDescriptor(hogDescriptor),
            scale_factor(scaleFactor),
            window_size(windowSize),
            stepSize(stepSize),
            detection_threshold_1(detectionThreshold1),
            detection_threshold_2(detectionThreshold2),
            overlap_threshold(overlapThreshold),
            downscale(downscale) {
        loadModels(svm_model_1_path, svm_model_2_path);
    }


    PersonDetector(const PersonDetector &) = delete;

    PersonDetector(PersonDetector &&) = delete;

    PersonDetector &operator=(const PersonDetector &) = delete;

    PersonDetector &operator=(PersonDetector &&) = delete;

    ~PersonDetector() = default;

    /**
     * Performs object detection on the given image using a sliding window approach.
     *
     * @param image The input image for object detection.
     * @return A pair of vectors containing detected regions (Rectangles) and corresponding confidence scores.
     */
    std::pair<std::vector<cv::Rect>, std::vector<float>> detect(cv::Mat &image) {

        double scale = 0;
        std::vector<cv::Rect> detections;
        std::vector<float> scores;

        // Generate an image pyramid for multi-scale detection
        std::vector<cv::Mat> pyramid = ImagePyramid::generate(image, 8, downscale, window_size);

        // Generate sliding windows for each scaled frame in the pyramid
        std::vector<std::vector<cv::Rect>> sliding_windows;

        for (const auto &scaledFrame: pyramid) {
            std::vector<cv::Rect> scale_windows;
            for (const auto &window: SlidingWindow::generate(scaledFrame, window_size, stepSize)) {
                scale_windows.push_back(window);
            }
            sliding_windows.push_back(scale_windows);
        }

        // Perform detection on each window in the sliding windows
        for (const auto &scale_windows: sliding_windows) {
            for (const auto &window: scale_windows) {

                // Extract HOG descriptors for the current window
                Eigen::MatrixXd eigen_window;
                cv::cv2eigen(image(window), eigen_window);
                Eigen::MatrixXd descriptors = this->hogDescriptor.compute(eigen_window);
                cv::Mat descriptors_mat;
                cv::eigen2cv(descriptors, descriptors_mat);

                // Confirm object presence using confirm method
                std::pair<float, float> res = confirm(descriptors_mat);

                // If confirmed, compute original coordinates, add detection and score
                if (res.first == 1) {
                    double temp = pow(downscale, scale);
                    int x_orig = (int) (window.x * temp / scale_factor);
                    int y_orig = (int) (window.y * temp / scale_factor);
                    int w_orig = (int) (window_size.width * temp / scale_factor);
                    int h_orig = (int) (window_size.height * temp / scale_factor);

                    detections.push_back(cv::Rect(x_orig, y_orig, w_orig, h_orig));
                    double decision = res.second;
                    scores.push_back(decision);
                }
            }
            scale++;
        }

        // Perform non-maximum suppression to filter detections
        std::vector<cv::Rect> picked = NonMaxSuppression::suppress(detections, scores, overlap_threshold);

        return std::make_pair(picked, scores);
    }

    /**
     * Confirm the presence of an object in the input vector using SVM models.
     *
     * @param vec Input vector for confirmation.
     * @return A pair containing confirmation status (1.0 for confirmed, else 0.0) and decision score.
     */
    std::pair<float, float> confirm(cv::Mat vec) const {
        // Convert input vector to CV_32F format
        vec.convertTo(vec, CV_32F);

        // Reshape the vector
        cv::Mat vecReshaped = vec.reshape(1, 1);

        // Predict using svm_model_1
        cv::Mat predMat_svm_1, decisionMat_1;
        svm_model_1->predict(vecReshaped, predMat_svm_1);
        svm_model_1->predict(vecReshaped, decisionMat_1, cv::ml::StatModel::RAW_OUTPUT);

        float pred_1 = predMat_svm_1.at<float>(0, 0);
        float decision_1 = decisionMat_1.at<float>(0, 0);

        // Check condition based on prediction from svm_model_1
        if (pred_1 == 1 && abs(decision_1) < (1.0 - detection_threshold_1)) {
            // Predict using svm_model_2
            cv::Mat predMat_svm_2, decisionMat_2;
            svm_model_2->predict(vecReshaped, predMat_svm_2);
            svm_model_2->predict(vecReshaped, decisionMat_2, cv::ml::StatModel::RAW_OUTPUT);

            float pred_2 = predMat_svm_2.at<float>(0, 0);
            float decision_2 = decisionMat_2.at<float>(0, 0);


            // Check condition based on prediction from svm_model_2
            if (pred_2 == 1 && abs(decision_2) < (1.0 - detection_threshold_2)) {

                float decision = (decision_1 + decision_2) / 2;

                return std::make_pair(1.0, decision);
            }
        }

        return std::make_pair<float, float>(1.0, 0.0); // Default return if conditions are not met
    }


    const double getScaleFactor() const {
        return scale_factor;
    }

    const cv::Size &getWindowSize() const {
        return window_size;
    }

    const cv::Size &getStepSize() const {
        return stepSize;
    }

    const double getDetectionThreshold1() const {
        return detection_threshold_1;
    }

    const double getDetectionThreshold2() const {
        return detection_threshold_2;
    }

    const float getOverlapThreshold() const {
        return overlap_threshold;
    }

    const double getDownscale() const {
        return downscale;
    }


};


#endif //TRACKING_PERSONDETECTOR_H
