#include "VideoCapture.h"

VideoCapture::VideoCapture(std::string &video_path) {
    if (video_path == "stream") {
        cap_ = cv::VideoCapture(0);
    } else {
        cap_ = cv::VideoCapture(video_path);
    }
    
}

bool VideoCapture::ReadFrame(cv::Mat &frame) {
    if (!empty_frame_) {
        cap_ >> frame;
    }

    if (frame.empty()) {
        empty_frame_ = true;
    }

    return !empty_frame_;
}

int VideoCapture::GetFrameWidth() {
    return cap_.get(cv::CAP_PROP_FRAME_WIDTH);
}

int VideoCapture::GetFrameHeight() {
    return cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
}

void VideoCapture::Release() {
    cap_.release();
}