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
#include "../include/PersonDetector.h"


using std::cout;
using std::endl;
using std::cerr;
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::destroyAllWindows;

cv::Mat trackOpticalFlow(cv::Mat prevGray, cv::Mat frame, cv::Point2f& prevDot, cv::Rect& bbox) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Point2f dotCenter((bbox.x + bbox.width) / 2.0f, (bbox.y + bbox.height) / 2.0f);

    std::vector<cv::Point2f> prevPoints, nextPoints;
    prevPoints.push_back(prevDot);

    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err);

    if (!status.empty() && status[0]) {
        prevDot = nextPoints[0];
    }

    cv::Point2f movement = prevDot - dotCenter;

    bbox.x += static_cast<int>(movement.x);
    bbox.y += static_cast<int>(movement.y);
    bbox.width += static_cast<int>(movement.x);
    bbox.height += static_cast<int>(movement.y);

    return gray.clone();
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }

    cv::Mat currentFrame, output_frame;


    HOGDescriptor hogDescriptor(9, std::make_pair(8, 8), std::make_pair(3, 3), cv::NORM_L2, false, false, true, true);


    const double scale_factor = 0.19;
    const cv::Size fixedFrameSize(365, 205); // (365, 205)

    const cv::Size size(96, 160);
    const cv::Size stepSize(10, 10); //(10,10)
    const double detection_threshold_1 = 0.8; // inria
    const double detection_threshold_2 = 0.7; // tt
    const float overlap_threshold = 0.3;
    const double downscale = 1.15;

    const double gamma = 1.5;
    const int blurKernelSize = 5;

    int nextObjectID = 0;
    std::map<int, cv::KalmanFilter> objectKalmanFilters;
    std::map<int, cv::Rect> objectRects;

    PersonDetector personDetector("../models/svm_model_inria_96_160_with_flipped.xml", "../models/svm_model_tt_96_160_with_flipped_1000.xml",hogDescriptor, scale_factor, size, stepSize, detection_threshold_1, detection_threshold_2, overlap_threshold, downscale);



    while (true) {
        cap.read(currentFrame);
        if (currentFrame.empty()) {
            break;
        }

        // Apply gamma correction to the currentFrame
        cv::Mat gammaCorrectedFrame;
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar *p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
        }
        cv::LUT(currentFrame, lookupTable, gammaCorrectedFrame);

        // Apply Gaussian blur to the gamma corrected currentFrame
        cv::GaussianBlur(gammaCorrectedFrame, gammaCorrectedFrame, cv::Size(blurKernelSize, blurKernelSize), 0);

        output_frame = gammaCorrectedFrame.clone();

        cv::resize(gammaCorrectedFrame, gammaCorrectedFrame, fixedFrameSize,
                   cv::INTER_AREA);

        cv::cvtColor(gammaCorrectedFrame, gammaCorrectedFrame, cv::COLOR_BGR2GRAY);



        std::pair<std::vector<cv::Rect>, std::vector<float>> res = personDetector.detect(gammaCorrectedFrame);
        std::vector<cv::Rect> picked = res.first;
//
//        std::vector<cv::Rect> mergedBoxes;
//        double mergeThreshold = 0.3; // Adjust this IoU threshold based on your requirements
//
//        for (size_t i = 0; i < picked.size(); ++i) {
//            cv::Rect currentBox = picked[i];
//            bool merged = false;
//
//            // Compare current box with others to find nearby boxes for merging
//            for (size_t j = i + 1; j < picked.size(); ++j) {
//                cv::Rect nextBox = picked[j];
//
//                // Calculate IoU (Intersection over Union) between boxes
//                cv::Rect intersect = currentBox & nextBox;
//                double intersectionArea = intersect.area();
//                double unionArea = currentBox.area() + nextBox.area() - intersectionArea;
//                double iou = intersectionArea / unionArea;
//
//                if (iou > mergeThreshold) {
//                    // Merge the boxes by expanding the bounding box to encapsulate both
//                    currentBox |= nextBox;
//                    merged = true;
//                    // Mark the nearby box as processed
//                    picked[j] = cv::Rect(-1, -1, -1, -1); // Set invalid rect to indicate it's merged
//                }
//            }
//
//            // If current box has been merged, add it to the list of merged boxes
//            if (merged) {
//                mergedBoxes.push_back(currentBox);
//            }
//        }

// Add unmerged boxes to the merged boxes list
//        for (const auto& box : picked) {
//            if (box.area() > 0) {
//                mergedBoxes.push_back(box);
//            }
//        }


        // Update Kalman filters with new detections
        for (size_t i = 0; i < picked.size(); ++i) {
            const cv::Rect &rect = picked[i];
//            const double &score = scores[i]; // Retrieve the score corresponding to the rectangle

            bool foundMatch = false;

            // Check if the detected rect overlaps with existing objects
            for (const auto &obj: objectRects) {
                cv::Rect overlap = rect & obj.second;
                if (overlap.area() > 0) {
                    foundMatch = true;


                    cv::KalmanFilter &kf = objectKalmanFilters[obj.first];
                    cv::Mat measurement = (cv::Mat_<float>(2, 1) << rect.x + rect.width / 2, rect.y +
                                                                                             rect.height / 2);
                    cv::Mat prediction = kf.predict();
                    cv::Mat estimated = kf.correct(measurement);

                    cv::Rect estimatedRect(estimated.at<float>(0) - rect.width / 2,
                                           estimated.at<float>(1) - rect.height / 2,
                                           rect.width, rect.height);

                    objectRects[obj.first] = estimatedRect;

                    cv::rectangle(output_frame, estimatedRect, cv::Scalar(0, 255, 0), 2);
                    cv::putText(output_frame,
                                "ID: " + std::to_string(obj.first),
                                cv::Point(estimatedRect.x, estimatedRect.y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

                    break;
                }
            }

            if (!foundMatch) {

                // Initialize Kalman filter for new object
                cv::KalmanFilter kf(4, 2);
                kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
                kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
                kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.1;
                kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 0.01;
                kf.statePost.at<float>(0) = rect.x + rect.width / 2;
                kf.statePost.at<float>(1) = rect.y + rect.height / 2;
                kf.statePost.at<float>(2) = 0;
                kf.statePost.at<float>(3) = 0;

                objectKalmanFilters[nextObjectID] = kf;
                objectRects[nextObjectID] = rect;

                cv::rectangle(output_frame, rect, cv::Scalar(0, 255, 0), 2);
                cv::putText(output_frame,
                            "ID: " + std::to_string(nextObjectID),
                            cv::Point(rect.x, rect.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

                nextObjectID++;
            }
        }


        cv::imshow("Person Detection", output_frame);
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

