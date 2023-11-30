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

    cv::Mat frame, output_frame;

    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::load(
            "../models/svm_model_inria+tt_96_160_with_flipped.xml");
    cv::Size size(96, 160);
    cv::Size stepSize(10, 10);
    double downscale = 1.5;
    double scale_factor = 0.19;
    double detection_threshold = 0.9;
    float overlap_threshold = 0.3;

    std::vector<cv::Rect> detections;
    std::vector<double> scores;

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            break;
        }


        // compute if old react are still valid
//
//        for (int i = 0; i < detections.size(); ++i) {
//
//            const cv::Rect &rect = detections.at(i);
//
//            Eigen::MatrixXd eigen_rect;
//            cv::cv2eigen(frame(rect), eigen_rect);
//
//            Eigen::MatrixXd descriptors = computeHOG(eigen_rect);
//
//            cv::Mat descriptors_mat;
//            cv::eigen2cv(descriptors, descriptors_mat);
//
//            descriptors_mat.convertTo(descriptors_mat, CV_32F);
//
//            cv::Mat descriptorsMatReshaped = descriptors_mat.reshape(1, 1);
//
//            cv::Mat predMat;
//
//            model->predict(descriptorsMatReshaped, predMat);
//
//            float pred = predMat.at<float>(0, 0);
//
//            if (pred == 1) { // if maybe human
//                cv::Mat decisionMat;
//                model->predict(descriptorsMatReshaped, decisionMat, cv::ml::StatModel::RAW_OUTPUT);
//
//                double decision = decisionMat.at<double>(0, 0);
//
//                decision = clip(normalize(decision, 1.56338e-314, 1.58213e-314), 0.0, 1.0);
//
//
//                cout << decision << endl;
//                if (decision > detection_threshold) {
//                    detections.erase(detections.begin() + i);
//                }
//            }
//        }

        output_frame = frame.clone();

        cv::resize(frame, frame, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        double scale = 0;


        for (cv::Mat im_scaled = frame.clone(); im_scaled.rows >= size.height && im_scaled.cols >= size.width;
             cv::resize(im_scaled, im_scaled, cv::Size(), 1.0 / downscale, 1.0 / downscale)) {
            for (const auto &window: sliding_window(im_scaled, size, stepSize)) {

                Eigen::MatrixXd eigen_window;
                cv::cv2eigen(frame(window), eigen_window);

                Eigen::MatrixXd descriptors = computeHOG(eigen_window);

                cv::Mat descriptors_mat;
                cv::eigen2cv(descriptors, descriptors_mat);

                descriptors_mat.convertTo(descriptors_mat,
                                          CV_32F); // Todo change in HOG::compute that features are returned as CV_32F if possible

                cv::Mat descriptorsMatReshaped = descriptors_mat.reshape(1, 1);

                cv::Mat predMat;

                model->predict(descriptorsMatReshaped, predMat);

                float pred = predMat.at<float>(0, 0);
                if (pred == 1) {
                    cv::Mat decisionMat;
                    model->predict(descriptorsMatReshaped, decisionMat, cv::ml::StatModel::RAW_OUTPUT);

                    double decision = decisionMat.at<double>(0, 0);

                    decision = clip(normalize(decision, 1.56338e-314, 1.58213e-314), 0.0, 1.0);


                    cout << decision << endl;
                    if (decision > detection_threshold) {
                        double temp = pow(downscale, scale);
                        int x_orig = (int) (window.x * temp / scale_factor);
                        int y_orig = (int) (window.y * temp / scale_factor);
                        int w_orig = (int) (size.width * temp / scale_factor);
                        int h_orig = (int) (size.height * temp / scale_factor);

                        detections.push_back(cv::Rect(x_orig, y_orig, w_orig, h_orig));
                        scores.push_back(decision);
                    }
                }
            }
            scale++;
        }

        std::vector<cv::Rect> picked;
        picked = non_max_suppression(detections, scores, overlap_threshold);

        for (const auto &rect: picked) {
            cv::rectangle(output_frame, rect, cv::Scalar(0, 0, 255), 3);
            cv::putText(output_frame, "Person", cv::Point(rect.x - 2, rect.y - 2), cv::FONT_HERSHEY_SIMPLEX, 2,
                        cv::Scalar(0, 0, 255), 2);
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
//    Size sample_size = Size(64, 128);
//    int limit = 500;
////    Size sample_size = Size(64,128);
//    // Load positive samples (e.g., pedestrian images) from INRIA dataset
//    std::vector<String> posFilesJPG, posFiles, tiktokPos, tiktokNeg;
////    cv::glob("/Users/louis/Downloads/INRIAPerson/Train/pos/*.png", posFilesPNG);
//    cv::glob("/Users/louis/PycharmProjects/cv_project/src/cv/svm/output_96_160/*.jpg", posFilesJPG);
//    tiktokPos = getImagePaths("/Users/louis/PycharmProjects/ki/cv/new_positive", limit);
//
//
////    posFiles.insert(posFiles.end(), posFilesPNG.begin(), posFilesPNG.end());
//    posFiles.insert(posFiles.end(), posFilesJPG.begin(), posFilesJPG.end());
//    posFiles.insert(posFiles.end(), tiktokPos.begin(), tiktokPos.end());
//
//
//    // Load negative samples (e.g., non-pedestrian images) from INRIA dataset
//    std::vector<String> negFilesPNG, negFilesJPG, negFiles;
//    glob("/Users/louis/Downloads/INRIAPerson/Train/neg/*.png", negFilesPNG); // PNG format
//    glob("/Users/louis/Downloads/INRIAPerson/Train/neg/*.jpg", negFilesJPG); // JPG format
//    tiktokNeg = getImagePaths("/Users/louis/PycharmProjects/cv_project/dataset/negative", limit);
//
//    negFiles.insert(negFiles.end(), negFilesPNG.begin(), negFilesPNG.end());
//    negFiles.insert(negFiles.end(), negFilesJPG.begin(), negFilesJPG.end());
//    negFiles.insert(negFiles.end(), tiktokNeg.begin(), tiktokNeg.end());
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
//    svm->save("svm_model_inria+tt_64_128_with_flipped.xml");
//
//    return 0;
//}
