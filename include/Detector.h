#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv::dnn;

class Detector {
public:
    Detector() = delete;
    Detector(string &model, string &weights, string &coco_names);
    Detector(const Detector &other);                                                      //copy constructor
    Detector &operator=(const Detector &other);                                           //copy asignment operator
    Detector(Detector &&other);                                                           //move constructor
    Detector &operator=(Detector &&other);                                                //move assignment operator
    ~Detector() {}

    void DetectObjects(cv::Mat &frame);

    vector<cv::Rect> getBoundingBoxes() {return boxes_;}
    vector<int> GetStatus() {return std::move(status_);}
    vector<int> GetReport() {return std::move(report_);}
    vector<int> GetIndicies() {return std::move(indices_);}
    vector<vector<cv::Point>> GetRiskPair() {return std::move(close_pair_);}
    vector<vector<cv::Point>> GetHighRiskPair() {return std::move(s_close_pair_);}

private:
    void GetClassNames();
    void ExtractPeople(cv::Mat &frame, const vector<cv::Mat> &outs, const vector<string>& classes);
    void DistanceAnalyze(cv::Mat &frame);
    
    int Distance(cv::Point c1, cv::Point c2);
    int T2S(int T);
    int T2C(int T);
    int IsClose(cv::Rect box1, cv::Rect box2, cv::Point cen1, cv::Point cen2);

    static vector<string> classes;
    vector<cv::Rect> boxes_;
    vector<int> indices_;
    
    vector<int> status_;
    vector<int> report_;
    vector<vector<cv::Point>> close_pair_;
    vector<vector<cv::Point>> s_close_pair_;

    const int angle_factor = 0.8;
    const int H_zoom_factor = 1.2;
    const int width = 416;        
    const int height = 416;       
    const float threshold = 0.5; 
    const float nms = 0.4; 

    cv::dnn::Net net;
    string model_{""};
    string weights_{""};
    string coco_names_{""};
};

#endif