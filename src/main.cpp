#include <opencv2/opencv.hpp>

using std::cerr;
using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::destroyAllWindows;

int main() {
    // Load an image from file
   Mat image = cv::imread("/Users/louis/CLionProjects/Tracking/images/frame_15344.jpg"); // Replace with your image path

    if (image.empty()) {
        cerr << "Error: Couldn't load the image!" << std::endl;
        return -1;
    }

    // Create a window to display the image
    namedWindow("Image", cv::WINDOW_NORMAL); // Create a resizable window
    imshow("Image", image);

    // Wait for a key press
    waitKey(0);

    // Close all OpenCV windows
    destroyAllWindows();

    return 0;
}
