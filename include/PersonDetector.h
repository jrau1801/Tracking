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
#include <opencv2/ximgproc/segmentation.hpp>

#include "../include/HOGDescriptor.h"
#include "../include/SlidingWindow.h"
#include "../include/NonMaxSuppression.h"
#include "../include/ImagePyramid.h"

#ifndef DEBUG_MODE
#define DEBUG_MODE 1 // Set to 1 to enable debug mode, 0 to disable
#endif


using namespace std;

class PersonDetector {
private:
    cv::Ptr<cv::ml::SVM> svm_model_1;

    HOGDescriptor hogDescriptor;
    double scale_factor;
    cv::Size window_size;
    double detection_threshold_1;
    double detection_threshold_2;
    float overlap_threshold;
    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(1000, 25, true);

    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss =
            cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

    void loadModels(const std::string &svm_model_1_path) {

        assert(!svm_model_1_path.empty());

        try {
            svm_model_1 = cv::ml::SVM::load(
                    svm_model_1_path);

        } catch (cv::Exception &e) {
            std::cerr << "There was an error while loading the svm_model_1\n" << e.what() << std::endl;
        }

    }

    static bool isRectInside(const cv::Rect &innerRect, const cv::Rect &outerRect) {
        // Check if the inner rectangle's points lie within the outer rectangle
        return (innerRect.x >= outerRect.x &&
                innerRect.y >= outerRect.y &&
                (innerRect.x + innerRect.width) <= (outerRect.x + outerRect.width) &&
                (innerRect.y + innerRect.height) <= (outerRect.y + outerRect.height));
    }


    void removeRectsInside(std::vector<cv::Rect> &rects) {
        auto it = rects.begin();

        while (it != rects.end()) {
            bool isInside = false;

            for (const cv::Rect &otherRect: rects) {
                if (*it != otherRect && isRectInside(*it, otherRect)) {
                    isInside = true;
                    break;
                }
            }

            if (isInside) {
                it = rects.erase(it); // Remove the rectangle if it's inside another
            } else {
                ++it;
            }
        }
    }


public:

    PersonDetector(const std::string &svm_model_1_path,
                   HOGDescriptor &hogDescriptor,
                   const double scaleFactor,
                   const std::pair<int, int> &windowSize,
                   const double detectionThreshold1,
                   const double detectionThreshold2, const float overlapThreshold) :
            hogDescriptor(hogDescriptor),
            scale_factor(scaleFactor),
            window_size(cv::Size(windowSize.first, windowSize.second)),

            detection_threshold_1(detectionThreshold1),
            detection_threshold_2(detectionThreshold2),
            overlap_threshold(overlapThreshold) {
        loadModels(svm_model_1_path);
    }


    PersonDetector(const PersonDetector &) = default;

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
    std::pair<std::vector<cv::Rect>, std::vector<float>>
    detect(const Eigen::MatrixXd &image, unsigned int minBboxSize = 10000) {

        cv::Mat mat_image;
        cv::eigen2cv(image, mat_image);

        // Check if grey scale
        assert(mat_image.channels() == 1);

        std::vector<cv::Rect> detections;
        std::vector<float> scores;


        cv::Mat fgMask, thresh, motionMask;

        pMOG2->apply(mat_image, fgMask,-1);


        cv::adaptiveThreshold(fgMask, motionMask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 27, 10);

//
//                             cv::threshold(motionMask, motionMask, 127, 255, cv::THRESH_BINARY+cv::THRESH_OTSU);
#if DEBUG_MODE
        cv::imshow("MotionMask None", motionMask);
#endif
        cv::dilate(motionMask, motionMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Size(0, 0),
                   1);
#if DEBUG_MODE
        cv::imshow("MotionMask Dilate", motionMask);
#endif
        // Morphological operations to clean up the mask
        morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

#if DEBUG_MODE
        cv::imshow("MotionMask Opening", motionMask);
#endif
        morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

#if DEBUG_MODE
        cv::imshow("MotionMask Closing", motionMask);
#endif
        // Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        // Find all contours and retrieve the full hierarchy
        findContours(motionMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


        std::vector<std::vector<cv::Point>> approxContours(contours.size());
        for (size_t i = 0; i < contours.size(); ++i) {
            cv::approxPolyDP(contours[i], approxContours[i], 4, true); // Adjust epsilon value as needed
        }

        // ============= Debug ================

#if DEBUG_MODE
        // Display Contours
        cv::Mat contour_image = cv::Mat::zeros(motionMask.size(), CV_8UC3);
        for (size_t i = 0; i < approxContours.size(); ++i) {
            drawContours(contour_image, approxContours, static_cast<int>(i), cv::Scalar(255, 255, 255), 2, cv::LINE_8,
                         hierarchy);
        }

        cv::imshow("Contours", contour_image);
#endif

        std::vector<cv::Rect> contour_detections;

        for (auto &contour: approxContours) {

            cv::Rect rect = cv::boundingRect(contour);

//            int x_orig = (int) (rect.x / scale_factor);
//            int y_orig = (int) (rect.y / scale_factor);
//            int w_orig = (int) (rect.width / scale_factor);
//            int h_orig = (int) (rect.height / scale_factor);
//
//            cv::Rect bbox(x_orig, y_orig, w_orig, h_orig);

            if (rect.area() > minBboxSize) {
//                detections.push_back(bbox);
                contour_detections.push_back(rect);
//                scores.push_back(1.0);

            }
        }

        removeRectsInside(detections);

        for (const cv::Rect &window: contour_detections) {


            cv::Mat roi = mat_image(window);

            // Resize the ROI to the desired window size for SVM input
            cv::Mat resized_roi;
            cv::resize(roi, resized_roi, window_size, cv::INTER_AREA);


            // Extract HOG descriptors for the current window
            Eigen::MatrixXd eigen_window;
            cv::cv2eigen(resized_roi, eigen_window);
            Eigen::MatrixXd descriptors = this->hogDescriptor.compute(eigen_window);
            cv::Mat descriptors_mat;
            cv::eigen2cv(descriptors, descriptors_mat);

            // Confirm object presence using confirm method
            std::pair<float, float> svm_results = confirm(descriptors_mat);

            float isPerson = svm_results.first; // 1 for person, 0 when not
            float score = svm_results.second;

            // If confirmed, compute original coordinates, add detection and score
            if (isPerson == 1) {
                int x_orig = (int) (window.x / scale_factor);
                int y_orig = (int) (window.y / scale_factor);
                int w_orig = (int) (window.width / scale_factor);
                int h_orig = (int) (window.height / scale_factor);

                cv::Rect bbox(x_orig, y_orig, w_orig, h_orig);

                detections.push_back(bbox);
                scores.push_back(score);
            }
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
        if (pred_1 == 1 && (1.0f - abs(decision_1)) > detection_threshold_1) {
//            cout << decision_1 << endl;
            // Predict using svm_model_2
            float decision = (1.0f - abs(decision_1));

            return std::make_pair(1.0, decision);
//            cv::Mat predMat_svm_2, decisionMat_2;
//            svm_model_2->predict(vecReshaped, predMat_svm_2);
//            svm_model_2->predict(vecReshaped, decisionMat_2, cv::ml::StatModel::RAW_OUTPUT);
//
//            float pred_2 = predMat_svm_2.at<float>(0, 0);
//            float decision_2 = decisionMat_2.at<float>(0, 0);
//
//
//            // Check condition based on prediction from svm_model_2
//            if (pred_2 == 1 && (1.0f - abs(decision_2)) > detection_threshold_2) {
//
//                float decision = ((1.0f - abs(decision_1)) + (1.0f - abs(decision_2))) / 2;
//
//                return std::make_pair(1.0, decision);
//            }
        }

        return std::make_pair<float, float>(0.0, 0.0); // Default return if conditions are not met
    }


    const double getScaleFactor() const {
        return scale_factor;
    }

    const cv::Size &getWindowSize() const {
        return window_size;
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


};


#endif //TRACKING_PERSONDETECTOR_H
