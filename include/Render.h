#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

class Render {
    public:
        Render() {}
        Render(const Render &other) {}                                                    //copy constructor
        Render &operator=(const Render &other);                                           //copy asignment operator
        Render(Render &&other) {}                                                       //move constructor
        Render &operator=(Render &&other);                                                //move assignment operator
        ~Render() {}

        cv::Mat RenderResult(cv::Mat &frame, const vector<cv::Rect> &boxes, 
                const vector<int> &indices, const vector<int> &status, const vector<int> &report, 
                const vector<vector<cv::Point>> &close_pair, const vector<vector<cv::Point>> &s_close_pair);
    private:
        cv::Mat AddText(const vector<int> &report, const int &H, const int &W);
};

#endif