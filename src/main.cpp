#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <map>
#include <opencv2/bgsegm.hpp>
#include <opencv2/ximgproc/segmentation.hpp>


#include "../include/HOGDescriptor.h"
#include "../include/PersonDetector.h"
#include "../include/PersonTracker.h"


using std::cout;
using std::endl;
using std::cerr;
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::destroyAllWindows;


int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }


    HOGDescriptor hogDescriptor(9, std::make_pair(8, 8), std::make_pair(3, 3), cv::NORM_L2, false, false, true, true);


    const double scale_factor = 0.3; // 0.3 original we reduced to reduce features and so also contours


    const std::pair<int, int> size(96, 160);
    const double detection_threshold_1 = 0.4; // inria
    const double detection_threshold_2 = 0.4; // tt
    const float overlap_threshold = 0.3;


//    "../models/svm_model_tt_96_160_with_flipped_1000.xml"

    PersonDetector personDetector("/Users/louis/CLionProjects/Tracking/cmake-build-debug/svm_model_inria+tt_96_160_with_cropped_3pK_2nk_tt.xml",
                                  hogDescriptor, scale_factor,
                                  size, detection_threshold_1, detection_threshold_2, overlap_threshold);


    cv::Mat currentFrame, output_frame;


    int frameCount = 0, detectionCount = 0;
    while (true) {
        cap.read(currentFrame);
        if (currentFrame.empty()) {
            break;
        }


        output_frame = currentFrame.clone();

        cv::Mat gray;
        cv::cvtColor(currentFrame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, gray, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);

        Eigen::MatrixXd eigen_frame;
        cv::cv2eigen(gray, eigen_frame);


        std::pair<std::vector<cv::Rect>, std::vector<float>> detections = personDetector.detect(eigen_frame);

        std::vector<cv::Rect> rects = detections.first;
        std::vector<float> scores = detections.second;

        if(!rects.empty()) {
            detectionCount++;
        }

        frameCount++;

        for (int i = 0; i < rects.size(); ++i) {
            cv::Rect rect = rects[i];
            cv::putText(output_frame, "Conf: " + std::to_string(scores[i]), cv::Point (rect.x, rect.y -10), cv::FONT_HERSHEY_SIMPLEX,
                        2, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(output_frame, rect, cv::Scalar(0, 255, 0), 2);
        }


        cv::imshow("Person Detection", output_frame);


        if (cv::waitKey(1) == 27) {
            break;
        }
        cout << "Frame: " << frameCount << ", Detections: " << detectionCount << endl;

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

