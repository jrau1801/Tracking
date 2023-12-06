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

#include "../include/HOG.h"

using std::cout;
using std::endl;
using std::cerr;
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::destroyAllWindows;

std::vector<cv::Rect> sliding_window(const cv::Mat &image, cv::Size windowSize, cv::Size stepSize) {
    std::vector<cv::Rect> windows;
    for (int y = 0; y < image.rows - windowSize.height; y += stepSize.height) {
        for (int x = 0; x < image.cols - windowSize.width; x += stepSize.width) {
            windows.push_back(cv::Rect(x, y, windowSize.width, windowSize.height));
        }
    }
    return windows;
}

std::vector<cv::Rect>
non_max_suppression(const std::vector<cv::Rect> &rects, const std::vector<double> &scores, float overlapThresh) {
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

std::vector<cv::Mat>
generateImagePyramid(const cv::Mat &frame, const size_t numScales, const double downscale, const cv::Size minSize) {
    std::vector<cv::Mat> pyramid;
    cv::Mat scaledFrame = frame.clone();
    for (int i = 0; i < numScales; ++i) {
        if (minSize.height > scaledFrame.rows || minSize.width > scaledFrame.cols) {
            break;
        }
        pyramid.push_back(scaledFrame.clone());
        cv::resize(scaledFrame, scaledFrame, cv::Size(), 1 / downscale,
                   1 / downscale); // Resizing with a scale factor of 0.75}
    }
    return pyramid;
}

double clip(double value, double minValue, double maxValue) {
    return (value < minValue) ? minValue : (value > maxValue) ? maxValue : value;
}

double normalize(double value, double minVal, double maxVal) {
    return (value - minVal) / (maxVal - minVal);
}


Eigen::MatrixXd computeHOG(Eigen::MatrixXd image) {
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res = HOG::compute(image, 9, std::make_pair(8, 8), std::make_pair(3, 3),
                                                                   cv::NORM_L2, false, false, true);
    return res.first;
}


int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }

    cv::Mat currentFrame, output_frame;

    cv::Ptr<cv::ml::SVM> model_inria;
    cv::Ptr<cv::ml::SVM> model_tt;
    try {
        model_inria = cv::ml::SVM::load(
                "../models/svm_model_inria_96_160_with_flipped.xml");

        model_tt = cv::ml::SVM::load(
                "../models/svm_model_tt_96_160_with_flipped_1000.xml");
    } catch (cv::Exception &e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    }

    const int cameraWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int cameraHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));


    const cv::Size camera_frame_input_size(cameraWidth, cameraHeight);
    const double scale_factor = 0.19;
    const cv::Size fixedFrameSize(static_cast<int>(round(camera_frame_input_size.width * scale_factor)),
                                  static_cast<int>(round(camera_frame_input_size.height * scale_factor))); // (365, 205)
    cv::Mat tempFrame = cv::Mat::zeros(fixedFrameSize, CV_8UC1);

    const cv::Size size(96, 160);
    const cv::Size stepSize(10, 10); //(10,10)
    const double detection_threshold = 0.9;
    const float overlap_threshold = 0.2 ;
    const double downscale = 1.15;


    // precompute sliding window rects
    std::vector<cv::Mat> pyramid = generateImagePyramid(tempFrame, 8, downscale, size);

    std::vector<std::vector<cv::Rect>> sliding_windows;

    for (const auto &scaledFrame: pyramid) {
        std::vector<cv::Rect> scale_windows;
        for (const auto &window: sliding_window(scaledFrame, size, stepSize)) {
            scale_windows.push_back(window);
        }
        sliding_windows.push_back(scale_windows);
    }

    const double gamma = 1.5;
    const int blurKernelSize = 5;

    int nextObjectID = 0;
    std::map<int, cv::KalmanFilter> objectKalmanFilters;
    std::map<int, cv::Rect> objectRects;

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

        double scale = 0;
        std::vector<cv::Rect> detections;
        std::vector<double> scores;

        for (const auto &scale_windows: sliding_windows) {
            for (const auto &window: scale_windows) {

                Eigen::MatrixXd eigen_window;
                cv::cv2eigen(gammaCorrectedFrame(window), eigen_window);

                Eigen::MatrixXd descriptors = computeHOG(eigen_window);

                cv::Mat descriptors_mat;
                cv::eigen2cv(descriptors, descriptors_mat);

                descriptors_mat.convertTo(descriptors_mat,
                                          CV_32F); // Todo change in HOG::compute that features are returned as CV_32F if possible

                cv::Mat descriptorsMatReshaped = descriptors_mat.reshape(1, 1);

                cv::Mat predMat_inria;
                model_inria->predict(descriptorsMatReshaped, predMat_inria);

                float pred_inria = predMat_inria.at<float>(0, 0);
                if (pred_inria == 1) {
                    cv::Mat decisionMat_inria;
                    model_inria->predict(descriptorsMatReshaped, decisionMat_inria, cv::ml::StatModel::RAW_OUTPUT);

                    float decision_inria = decisionMat_inria.at<float>(0,0);

                    cout << decision_inria << endl;

//                    decision_inria = clip(normalize(decision_inria, 1.56338e-314, 1.58213e-314), 0.0, 1.0);

//                    cout << "decision_inria: " << decision_inria << endl;
                    if (abs(decision_inria) < 0.2) {

                        cv::Mat predMat_tt;
                        model_inria->predict(descriptorsMatReshaped, predMat_tt);

                        float pred_tt = predMat_tt.at<float>(0, 0);

                        if (pred_tt == 1) {
                            cv::Mat decisionMat_tt;
                            model_tt->predict(descriptorsMatReshaped, decisionMat_tt, cv::ml::StatModel::RAW_OUTPUT);

                            double decision_tt = decisionMat_tt.at<double>(0, 0);

//                            decision_tt = clip(normalize(decision_tt, 1.56338e-315, 1.58213e-314), 0.0, 1.0);
                            if (abs(decision_tt) < 0.3) {

                                double temp = pow(downscale, scale);
                                int x_orig = (int) (window.x * temp / scale_factor);
                                int y_orig = (int) (window.y * temp / scale_factor);
                                int w_orig = (int) (size.width * temp / scale_factor);
                                int h_orig = (int) (size.height * temp / scale_factor);


                                detections.push_back(cv::Rect(x_orig, y_orig, w_orig, h_orig));

                                double decision = (decision_inria + decision_tt) / 2;
                                scores.push_back(decision);

                            }
                        }
                    }
                }
            }
            scale++;
        }

        std::vector<cv::Rect> picked = non_max_suppression(detections, scores, overlap_threshold);
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

