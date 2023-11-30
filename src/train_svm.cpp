////
//// Created by Louis-Kaan Ay on 30.11.23.
////
//
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
//    svm->save("svm_model_inria+tt_96_160_with_flipped.xml");
//
//    return 0;
//}