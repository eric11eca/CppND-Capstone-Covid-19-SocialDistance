#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <string>

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class VideoCapture {
    public:
        VideoCapture(std::string &video_path);
        bool ReadFrame(cv::Mat &frame);
        int GetFrameWidth();
        int GetFrameHeight();
        void Release();
        ~VideoCapture(){};
  
    private:
        cv::VideoCapture cap_;
        bool empty_frame_ = false;
};

#endif