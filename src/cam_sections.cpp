#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <chrono>
#include <thread>
using namespace cv;

int main() {
    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening camera" << std::endl;
        return -1;
    }

    // Check the camera properties and set the width and height of the capture frame
    cap.set(CAP_PROP_FRAME_WIDTH, 1600);
    cap.set(CAP_PROP_FRAME_HEIGHT, 900);

    // Create a window to display the output
    namedWindow("Split Frame", WINDOW_NORMAL);

    // Get the dimensions of the frame
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);



    while (true) {
        Mat frame;
        cap >> frame; // Capture frame from the camera

        if (frame.empty()) {
            std::cout << "No frame captured from camera" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
        int randomX = rand() % (width - (width / 3)); // Random x-value within the frame width
        Rect boundingBox(randomX, height / 4, width / 3, height / 2);

        // Split the frame into three equal sections
        Rect section1(0, 0, width / 3, height);
        Rect section2(width / 3, 0, width / 3, height);
        Rect section3((width / 3) * 2, 0, width / 3, height);

        // Calculate intersection areas
        double intersection1 = (section1 & boundingBox).area();
        double intersection2 = (section2 & boundingBox).area();
        double intersection3 = (section3 & boundingBox).area();

        // Calculate union areas
        double union1 = section1.area() + boundingBox.area() - intersection1;
        double union2 = section2.area() + boundingBox.area() - intersection2;
        double union3 = section3.area() + boundingBox.area() - intersection3;

        // Calculate IoU values
        double iou1 = intersection1 / union1;
        double iou2 = intersection2 / union2;
        double iou3 = intersection3 / union3;

        // Determine which section the bounding box belongs to based on the highest IoU
        int section = 0;
        double max_iou = std::max({iou1, iou2, iou3});
        if (max_iou == iou1) {
            section = 1;
        } else if (max_iou == iou2) {
            section = 2;
        } else if (max_iou == iou3) {
            section = 3;
        }

        // Draw the text indicating the section
        std::string section_text = "Section: " + std::to_string(section);
        putText(frame, section_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

        // Draw the rectangles for sections and bounding box
        rectangle(frame, section1, Scalar(255, 0, 0), 2); // Blue, thickness = 2
        rectangle(frame, section2, Scalar(0, 255, 0), 2); // Green, thickness = 2
        rectangle(frame, section3, Scalar(0, 0, 255), 2); // Red, thickness = 2
        rectangle(frame, boundingBox, Scalar(255, 255, 0), 2); // Yellow, thickness = 2

        // Display the frame
        imshow("Split Frame", frame);

        // Check for the 'ESC' key press to exit
        if (waitKey(1) == 27) {
            break;
        }
    }

    // Release the camera and destroy windows
    cap.release();
    destroyAllWindows();

    return 0;
}
