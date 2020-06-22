#include "Render.h"         
                                     
Render &Render::operator=(const Render &other) {
    if (this == &other) {
        return *this;
    }

    return *this;
}                   
                                   
Render &Render::operator=(Render &&other) {
    if (this == &other) {
        return *this;
    }

    return *this;
}   

cv::Mat Render::RenderResult(cv::Mat &frame, const vector<cv::Rect> &boxes, 
                const vector<int> &indices, const vector<int> &status, const vector<int> &report, 
                const vector<vector<cv::Point>> &close_pair, const vector<vector<cv::Point>> &s_close_pair) {
    int kk = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        if (status[kk] == 1) {
            cv::rectangle(frame, cv::Point(box.x, box.y), 
                cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0,0,150), 2);
        } else if (status[kk] == 0) {
            cv::rectangle(frame, cv::Point(box.x, box.y), 
                cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0,255,0), 2);
        } else {
            cv::rectangle(frame, cv::Point(box.x, box.y), 
                cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0,120,255), 2);
        }
        kk += 1;
    }

    for (vector<cv::Point> pair : close_pair) {
        line(frame, pair[0], pair[1], cv::Scalar(0,0,255), 2);
    }

    for (vector<cv::Point> pair : s_close_pair) {
        line(frame, pair[0], pair[1], cv::Scalar(0,255,255), 2);
    }

    int W = frame.cols, H = frame.rows;
    cv::Mat FR = AddText(report, H, W);

    cv::Mat finalFrame;
    try {
        finalFrame = FR(cv::Rect(0, 0, frame.cols, frame.rows));
    } catch (...) {
        cout << "Trying to create roi out of image boundaries" << std::endl;
    }
    frame.copyTo(finalFrame);
    frame = FR;

    cv::Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    return std::move(frame);
}

cv::Mat Render::AddText(const vector<int> &report, const int &H, const int &W) {
    int FW;
    if (W < 1075) { 
        FW = 1075;
    } else {
        FW = W;
    }

    cv::Mat FR(H+210, FW, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::line(FR, cv::Point(0, H+1), cv::Point(FW, H+1), cv::Scalar(0,0,0), 2);
    cv::putText(FR, "COVID-19 Social Distancing Monitor", cv::Point(210, H+60),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);

    cv::rectangle(FR, cv::Point(20, H+80), cv::Point(510, H+180), cv::Scalar(100,100,100), 2);
    cv::putText(FR, "Connecting lines shows closeness among people. ", cv::Point(30, H+100),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100,100,0), 2);

    cv::putText(FR, "-- YELLOW: CLOSE", cv::Point(50, H+90+40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,170,170), 2);
    cv::putText(FR, "-- RED: VERY CLOSE", cv::Point(50, H+40+110),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 2);

    cv::rectangle(FR, cv::Point(535, H+80), cv::Point(1060, H+140+40), cv::Scalar(100,100,100), 2);
    cv::putText(FR, "Bounding box shows the level of risk to the person.", cv::Point(545, H+100),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(100,100,0), 2);

    cv::putText(FR, "-- RED: HIGH RISK", cv::Point(565, H+90+40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,150), 2);
    cv::putText(FR, "-- ORANGE: LOW RISK", cv::Point(565, H+150),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,120,255), 2);
    cv::putText(FR, "-- GREEN: SAFE", cv::Point(565, H+170),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,150,0), 2);
    
    string tot_str = "TOTAL COUNT: " + std::to_string(report[0]);
    string safe_str = "SAFE COUNT: " + std::to_string(report[1]);
    string low_str = "LOW RISK COUNT: " + std::to_string(report[2]);
    string high_str = "HIGH RISK COUNT: " + std::to_string(report[3]);

    cv::putText(FR, tot_str, cv::Point(10, H +25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    cv::putText(FR, safe_str, cv::Point(200, H +25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 170, 0), 2);
    cv::putText(FR, low_str, cv::Point(380, H +25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 120, 255), 2);
    cv::putText(FR, high_str, cv::Point(630, H +25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 150), 2);
    return FR;
}