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

std::vector<cv::Mat> generateImagePyramid(const cv::Mat &frame, const size_t numScales,const double downscale, const cv::Size minSize) {
    std::vector<cv::Mat> pyramid;
    cv::Mat scaledFrame = frame.clone();
    for (int i = 0; i < numScales; ++i) {
        if (minSize.height > scaledFrame.rows || minSize.width > scaledFrame.cols) {
            break;
        }
        pyramid.push_back(scaledFrame.clone());
        cv::resize(scaledFrame, scaledFrame, cv::Size(), 1/ downscale, 1/ downscale); // Resizing with a scale factor of 0.75}
    }
    return pyramid;
}

//double clip(double value, double minValue, double maxValue) {
//    return (value < minValue) ? minValue : (value > maxValue) ? maxValue : value;
//}
//
//double normalize(double value, double minVal, double maxVal) {
//    return (value - minVal) / (maxVal - minVal);
//}


Eigen::MatrixXd computeHOG(Eigen::MatrixXd image) {
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> res = HOG::compute(image, 9, std::make_pair(8, 8), std::make_pair(3, 3),
                                                                   cv::NORM_L2, false, false, true);
    return res.first;
}

/*
int main() {

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }


    cv::Mat currentFrame, output_frame;


//    cv::Mat flow, flowXY[2];


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

    const cv::Size size(96, 160);
    const cv::Size stepSize(10,10); //(10,10)
    const double scale_factor = 0.2;
    const double detection_threshold = 0.9;
    const float overlap_threshold = 0.2;
    const double downscale = 1.15;


    const double gamma = 1.5; // Adjust gamma value as needed
    const int blurKernelSize = 5; // Adjust blur kernel size as needed

    int nextObjectID = 0;
    std::map<int, cv::KalmanFilter> objectKalmanFilters;
    std::map<int, cv::Rect> objectRects;

    while (true) {
        cap.read(currentFrame);
        if (currentFrame.empty()) {
            break;
        }

        cv::resize(currentFrame, currentFrame, cv::Size(1600, 900), cv::INTER_AREA);
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

        cv::resize(gammaCorrectedFrame, gammaCorrectedFrame, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);


        cv::cvtColor(gammaCorrectedFrame, gammaCorrectedFrame, cv::COLOR_BGR2GRAY);

        double scale = 0;
        std::vector<cv::Rect> detections;
        std::vector<double> scores;

        std::vector<cv::Mat> pyramid = generateImagePyramid(gammaCorrectedFrame, 5, downscale, size);

        int pyi = 0;
        for (const auto &scaledFrame: pyramid) {

            // imshow("pyramid " + std::to_string(pyi), scaledFrame);
            pyi++;

            for (const auto &window: sliding_window(scaledFrame, size, stepSize)) {

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

                    float decision_inria = decisionMat_inria.at<float>(0, 0);

                    cout << decision_inria << endl;

                    if (abs(decision_inria) < 0.6 && abs(decision_inria) > 0.1) {

                        cv::Mat predMat_tt;
                        model_inria->predict(descriptorsMatReshaped, predMat_tt);

                        float pred_tt = predMat_tt.at<float>(0, 0);

                        if (pred_tt == 1) {
                            cv::Mat decisionMat_tt;
                            model_tt->predict(descriptorsMatReshaped, decisionMat_tt, cv::ml::StatModel::RAW_OUTPUT);

                            float decision_tt = decisionMat_tt.at<float>(0, 0);

//                            decision_tt = clip(normalize(decision_tt, 1.56338e-315, 1.58213e-314), 0.0, 1.0);
                            if (abs(decision_tt) < 0.5 && abs(decision_tt) > 0.1) {

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

        // Update Kalman filters with new detections
        for (size_t i = 0; i < picked.size(); ++i) {
            const cv::Rect &rect = picked[i];
            const double &score = scores[i]; // Retrieve the score corresponding to the rectangle

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
                                "ID: " + std::to_string(obj.first) + " Score: " + std::to_string(score),
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
                            "ID: " + std::to_string(nextObjectID) + " Score: " + std::to_string(score),
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
*/


// Created by Louis-Kaan Ay on 29.11.23.
//#include <algorithm>
//#include <opencv2/opencv.hpp>
//#include <opencv2/ml.hpp>
//#include <Eigen/Core>
//#include "../include/HOG.h"
//#include <iostream>
//#include <iterator>
//#include <random>
//#include <vector>
//#include <string>
//#include <filesystem>
//
//using namespace cv;
//using namespace cv::ml;
//namespace fs = std::filesystem;
//
//std::vector<std::string> getImagePaths(const std::string &directory, size_t maxImages = 10000000000000000) {
//    std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png"}; // Add more extensions if needed
//    std::vector<std::string> imagePaths;
//
//    size_t limit = 0;
//
//    // Check if the directory exists
//    if (fs::exists(directory) && fs::is_directory(directory)) {
//        for (const auto &entry: fs::recursive_directory_iterator(directory)) {
//            if (limit == maxImages) {
//                return imagePaths;
//            }
//
//            if (fs::is_regular_file(entry.path())) {
//                std::string fileExtension = entry.path().extension().string();
//                // Convert the extension to lowercase for comparison
//                std::transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(), ::tolower);
//
//                // Check if the file has an image extension
//                if (std::find(imageExtensions.begin(), imageExtensions.end(), fileExtension) != imageExtensions.end()) {
//                    imagePaths.push_back(entry.path().string());
//                    limit++;
//                }
//            }
//        }
//    } else {
//        std::cout << "Directory doesn't exist or is not a directory." << std::endl;
//    }
//
//    return imagePaths;
//}
//
//int main() {
//
//    Size sample_size = Size(96, 160);
//    int limit = 1000;
////    Size sample_size = Size(64,128);
//    // Load positive samples (e.g., pedestrian images) from INRIA dataset
//    std::vector<String> posFilesJPG, posFiles, tiktokPos, tiktokNeg;
////    cv::glob("/Users/louis/Downloads/INRIAPerson/Train/pos/*.png", posFilesPNG);
////    cv::glob("/Users/louis/PycharmProjects/cv_project/src/cv/svm/output_96_160/*.jpg", posFilesJPG);
//    tiktokPos = getImagePaths("/Users/louis/PycharmProjects/ki/cv/new_positive", limit);
//
//
////    posFiles.insert(posFiles.end(), posFilesPNG.begin(), posFilesPNG.end());
////    posFiles.insert(posFiles.end(), posFilesJPG.begin(), posFilesJPG.end());
//    posFiles.insert(posFiles.end(), tiktokPos.begin(), tiktokPos.end());
//
//
//    // Load negative samples (e.g., non-pedestrian images) from INRIA dataset
//    std::vector<String> negFilesPNG, negFilesJPG, negFiles;
////    glob("/Users/louis/Downloads/INRIAPerson/Train/neg/*.png", negFilesPNG); // PNG format
////    glob("/Users/louis/Downloads/INRIAPerson/Train/neg/*.jpg", negFilesJPG); // JPG format
//    tiktokNeg = getImagePaths("/Users/louis/PycharmProjects/cv_project/dataset/negative", limit);
////
////    negFiles.insert(negFiles.end(), negFilesPNG.begin(), negFilesPNG.end());
////    negFiles.insert(negFiles.end(), negFilesJPG.begin(), negFilesJPG.end());
//    negFiles.insert(negFiles.end(), tiktokNeg.begin(), tiktokNeg.end());
//
//    std::cout << posFiles.size() << std::endl;
//    std::cout << negFiles.size() << std::endl;
//
//
//    std::random_device rd;
//    std::mt19937 g(rd());
//    // Shuffle the file list
//    std::shuffle(posFiles.begin(), posFiles.end(), g);
//    std::shuffle(negFiles.begin(), negFiles.end(), g);
//
//    // Prepare data structures for training
//    Mat trainData;
//    std::vector<int> labels;
//
//    // Iterate through the combined list of files to extract HOG features
//    for (const auto &file: posFiles) {
//        Mat img = imread(file);
//        if (img.empty())
//            continue;
//
//        cvtColor(img, img, COLOR_BGR2GRAY);
//        resize(img, img, sample_size);
//
//        Mat flipped_image;
//        cv::flip(img, flipped_image, 1);
//
//        Eigen::MatrixXd eigen_image, eigen_image_flipped;
//        cv::cv2eigen(img, eigen_image);
//        cv::cv2eigen(flipped_image, eigen_image_flipped);
//
//        // Extract HOG features from the image
//        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hogFeatures_1 = HOG::compute(eigen_image, 9, std::make_pair(8, 8),
//                                                                                 std::make_pair(3, 3),
//                                                                                 cv::NORM_L2, false, false, true, true);
//        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hogFeatures_2 = HOG::compute(eigen_image_flipped, 9,
//                                                                                 std::make_pair(8, 8),
//                                                                                 std::make_pair(3, 3),
//                                                                                 cv::NORM_L2, false, false, true, true);
//
//        // Convert HOG features to OpenCV Mat
//        Mat hogMat, hogMatFlipped;
//        cv::eigen2cv(hogFeatures_1.first, hogMat);
//        cv::eigen2cv(hogFeatures_2.first, hogMatFlipped);
//
//        // Reshape the HOG Mat to a single row and append to trainData
//        hogMat = hogMat.reshape(1, 1); // Reshape to a single row matrix
//        hogMatFlipped = hogMatFlipped.reshape(1, 1); // Reshape to a single row matrix
//
//        trainData.push_back(hogMat);
//        trainData.push_back(hogMatFlipped);
//
//
//        labels.push_back(1);
//        labels.push_back(1);
//
//    }
//
//    for (const auto &file: negFiles) {
//        Mat img = imread(file);
//        if (img.empty())
//            continue;
//
//        cvtColor(img, img, COLOR_BGR2GRAY);
//        resize(img, img, sample_size);
//
//        Mat flipped_image;
//        cv::flip(img, flipped_image, 1);
//
//        Eigen::MatrixXd eigen_image, eigen_image_flipped;
//        cv::cv2eigen(img, eigen_image);
//        cv::cv2eigen(flipped_image, eigen_image_flipped);
//
//        // Extract HOG features from the image
//        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hogFeatures_1 = HOG::compute(eigen_image, 9, std::make_pair(8, 8),
//                                                                                 std::make_pair(3, 3),
//                                                                                 cv::NORM_L2, false, false, true, true);
//        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> hogFeatures_2 = HOG::compute(eigen_image_flipped, 9,
//                                                                                 std::make_pair(8, 8),
//                                                                                 std::make_pair(3, 3),
//                                                                                 cv::NORM_L2, false, false, true, true);
//
//        // Convert HOG features to OpenCV Mat
//        Mat hogMat, hogMatFlipped;
//        cv::eigen2cv(hogFeatures_1.first, hogMat);
//        cv::eigen2cv(hogFeatures_2.first, hogMatFlipped);
//
//        // Reshape the HOG Mat to a single row and append to trainData
//        hogMat = hogMat.reshape(1, 1); // Reshape to a single row matrix
//        hogMatFlipped = hogMatFlipped.reshape(1, 1); // Reshape to a single row matrix
//
//        trainData.push_back(hogMat);
//        trainData.push_back(hogMatFlipped);
//
//
//        labels.push_back(0);
//        labels.push_back(0);
//    }
//
//    // Convert labels to Mat
//    Mat labelsMat(labels, true);
//
//// Ensure the correct shape and type of trainData and labelsMat
//    trainData = trainData.reshape(1, trainData.rows); // Reshape trainData to have one row per sample
//    trainData.convertTo(trainData, CV_32F); // Convert to desired type (e.g., CV_32F)
//    labelsMat.convertTo(labelsMat, CV_32S); // Convert to desired type (e.g., CV_32S)
//
//
//    // Assuming trainData is your matrix of training data
//    int numSamples = trainData.rows; // Number of samples (rows)
//    int numFeatures = trainData.cols; // Number of features (columns)
//
//// Print the shape
//    std::cout << "Shape of trainData: " << numSamples << " rows x " << numFeatures << " columns" << std::endl;
//    std::cout << "Shape of labels: " << labelsMat.rows << " rows x " << labelsMat.cols << " columns" << std::endl;
//
//
//    if (trainData.empty() || labelsMat.empty()) {
//        std::cerr << "Empty data matrices!" << std::endl;
//        return -1; // Or handle appropriately
//    }
//
//    // Create an SVM instance
//    Ptr<SVM> svm = SVM::create();
//
//    // Set SVM parameters
//    svm->setType(SVM::C_SVC);
//    svm->setKernel(SVM::LINEAR);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//
//    // Train SVM using HOG features
//    svm->train(trainData, ROW_SAMPLE, labelsMat);
//
//    // Save trained SVM model to a file
//    svm->save("svm_model_tt_96_160_with_flipped_1000.xml");
//
//    return 0;
//}
