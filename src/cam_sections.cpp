#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <ctime>
using namespace cv;

int main() {
    // Open the default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening camera" << std::endl;
        return -1;
    }

    // Check the camera properties and set the width and height of the capture frame
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // Create a window to display the output
    namedWindow("Split Frame", WINDOW_NORMAL);

    // Get the dimensions of the frame
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);

    // Generate a random x-coordinate for the fourth bounding box
    srand(static_cast<unsigned int>(time(0)));
    int randomX = rand() % (width - (width / 3)); // Random x-value within the frame width

    // Fourth bounding box coordinates
    Rect boundingBox(randomX, height / 4, width / 3, height / 2);

    while (true) {
        Mat frame;
        cap >> frame; // Capture frame from the camera

        if (frame.empty()) {
            std::cout << "No frame captured from camera" << std::endl;
            break;
        }

        // Split the frame into three equal sections
        Rect section1(0, 0, width / 3, height);
        Rect section2(width / 3, 0, width / 3, height);
        Rect section3((width / 3) * 2, 0, width / 3, height);

        // Draw hollow rectangles with different colors in each section
        rectangle(frame, section1, Scalar(255, 0, 0), 2); // Blue, thickness = 2
        rectangle(frame, section2, Scalar(0, 255, 0), 2); // Green, thickness = 2
        rectangle(frame, section3, Scalar(0, 0, 255), 2); // Red, thickness = 2

        // Draw the fixed fourth bounding box
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
