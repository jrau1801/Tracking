//
// Created by Louis-Kaan Ay on 12.12.23.
//

#ifndef TRACKING_PERSONTRACKER_H
#define TRACKING_PERSONTRACKER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <map>

#include "../include/HOGDescriptor.h"
#include "../include/PersonDetector.h"
#include "../include/Track.h"

class PersonTracker {
private:
    PersonDetector personDetector;

    std::vector<Track> activeTracks;
    size_t trackCounter;

    int frameCount;
    double totalTime;


public:
    PersonTracker(PersonDetector &personDetector) : personDetector(personDetector), trackCounter(0), frameCount(0),
                                                    totalTime(0.0) {
    }

    std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<float>, double>
    track(const Eigen::MatrixXd &frame) {
        // Get detections from current frame
        std::pair<std::vector<cv::Rect>, std::vector<float>> res = personDetector.detect(frame);
        std::vector<cv::Rect> picked = res.first;
        std::vector<float> scores = res.second;
        std::vector<int> ids(picked.size(),1);


        // Todo Create a cost matrix for Hungarian algorithm

        // Todo Assign detections to tracks using Hungarian algorithm

        // Todo Update tracks based on assignments and unassigned detections

        // Todo Update states of active tracks and increment ages
//        for (auto &track : activeTracks) {
//            // Todo update velocity and acceleration
//            track.setAge(track.getAge() + 1);
//        }


        // Todo For unassigned detections, create new tracks for potential objects entering the scene.

        // Todo Remove inactive tracks that have been unassigned for too long or are to uncertain



        // Calculate FPS
        frameCount++;
        double current_time = static_cast<double>(cv::getTickCount());
        double elapsedTime = (current_time - totalTime) / cv::getTickFrequency();
        totalTime = current_time;
        double fps = 1.0 / elapsedTime;

        // Pybind11 will throw an error if we return std::vector<cv::Rects> so we convert it std::vector<std::vector<int>> here
        std::vector<std::vector<int>> castedRects;

        for (auto &r: picked) {
            std::vector<int> rectValues;
            rectValues.push_back(r.x);
            rectValues.push_back(r.y);
            rectValues.push_back(r.width);
            rectValues.push_back(r.height);
            castedRects.push_back(rectValues);

        }

        return std::make_tuple(castedRects, ids, scores, fps);
    }
};


#endif //TRACKING_PERSONTRACKER_H
