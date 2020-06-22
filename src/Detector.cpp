#include "Detector.h"

std::vector<std::string> Detector::classes;

Detector::Detector(std::string &model, std::string &weights, std::string &coco_names) {
    model_ = model;
    weights_ = weights;
    coco_names_= coco_names;

    net = readNetFromDarknet(model_, weights_);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    GetClassNames();
}

Detector::Detector(const Detector &other) {
    model_ = other.model_;
    weights_ = other.weights_;
    coco_names_ = other.coco_names_;
    net = other.net;
}

Detector &Detector::operator=(const Detector &other) {
    if (this == &other) {
        return *this;
    }

    model_ = other.model_;
    weights_ = other.weights_;
    coco_names_ = other.coco_names_;
    net = other.net;

    return *this;
}

Detector::Detector(Detector &&other) {
    model_ = std::move(other.model_);
    weights_ = std::move(other.weights_);
    coco_names_ = std::move(other.coco_names_);
    net = std::move(other.net);
}

Detector &Detector::operator=(Detector &&other) {
    if (this == &other) {
        return *this;
    }

    model_ = std::move(other.model_);
    weights_ = std::move(other.weights_);
    coco_names_ = std::move(other.coco_names_);
    net = std::move(other.net);

    return *this;
}

void Detector::GetClassNames() {
    ifstream ifs(coco_names_.c_str());
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
}

void Detector::DetectObjects(cv::Mat &frame) {
    cv::Mat blob;
    blobFromImage(frame, blob, 1 / 300.0, cv::Size(width, height), 
                    cv::Scalar(0, 0, 0), true, false);

    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    ExtractPeople(frame, outs, classes);
    DistanceAnalyze(frame);
}

void Detector::ExtractPeople(cv::Mat &frame, const std::vector<cv::Mat> &outs, const vector<string>& classes) {
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    for (cv::Mat out : outs) {
        float *data = (float *)out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            cv::Mat location = out.row(j).colRange(0, 4);
            cv::Mat scores = out.row(j).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;

            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (classes[classIdPoint.x] == "person") {
                if (confidence > threshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }   
            }
        }
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, threshold, nms, indices);

    boxes_ = boxes;
    indices_ = indices;
}

void Detector::DistanceAnalyze(cv::Mat &frame) {
    vector<cv::Point> center;
    close_pair_.clear();
    s_close_pair_.clear();
    status_.clear();

    for (size_t i = 0; i < indices_.size(); ++i) {
        int idx = indices_[i];
        cv::Rect box = boxes_[idx];
        cv::Point cen;
        cen.x = box.x + box.width / 2;
        cen.y = box.y + box.height / 2;
        center.push_back(cen);
        circle(frame, cen, 1, cv::Scalar(0,0,0), 1);
        status_.push_back(0);
    }

    for (int i = 0; i < center.size(); i++) {
        for (int j = 0; j < center.size(); j++) {
            int score = IsClose(boxes_[i], boxes_[j], center[i], center[j]);
            if (score == 1) {
                vector<cv::Point> pair; 
                pair.push_back(center[i]);
                pair.push_back(center[j]);

                close_pair_.push_back(pair);
                status_[i] = 1;
                status_[j] = 1;
            } else if (score == 2) {
                vector<cv::Point> pair; 
                pair.push_back(center[i]); 
                pair.push_back(center[j]);

                s_close_pair_.push_back(pair);
                if (status_[i] != 1) status_[i] = 2;
                if (status_[j] != 1) status_[j] = 2;
            }
        }
    }

    int total_p = center.size();
    int safe_p = std::count(status_.begin(), status_.end(), 0);
    int low_risk_p = std::count(status_.begin(), status_.end(), 2);
    int high_risk_p = std::count(status_.begin(), status_.end(), 1);
    
    report_.clear();
    report_.push_back(total_p);
    report_.push_back(safe_p);
    report_.push_back(low_risk_p);
    report_.push_back(high_risk_p);
}

int Detector::Distance(cv::Point c1, cv::Point c2) {
    int p1 = pow((c1.x - c2.x),2);
    int p2 = pow((c1.y - c2.y),2);
    return (int) sqrt(p1+p2);
}

int Detector::T2S(int T) {
    int S = abs(T/pow(pow(1+T, 2), 0.5));
    return S;
}

int Detector::T2C(int T) {
    int C = abs(1/pow(pow(1+T, 2), 0.5));
    return C;
}

int Detector::IsClose(cv::Rect box1, cv::Rect box2, cv::Point cen1, cv::Point cen2) {
    int close_dist = Distance(cen1, cen2);
    int a_w, a_h;
    
    /*if (box1.height < box2.height) {
        a_w = box1.width;
        a_h = box1.height;
    } else {
        a_w = box2.width;
        a_h = box2.height;
    }

    int T;
    int dinominator = cen1.x - cen2.x;
    if (dinominator > 0) {
        T = (cen1.y - cen2.y) / dinominator;
    } else {
        T = 1.633123935319537e+10;
    }

    int S = T2S(T);
    int C = T2C(T);

    int d_hor = C*close_dist;
    int d_ver = S*close_dist;
    int vc_calib_hor = a_w*1.3;
    int vc_calib_ver = a_h*0.4*angle_factor;
    int c_calib_hor = a_w *1.7;
    int c_calib_ver = a_h*0.2*angle_factor;

    if (close_dist < 100) {
        cout << "distance: " << close_dist << endl;
        cout << "d_hor: " << d_hor << " d_ver: " << d_ver << endl;
        cout << "vc_calib_hor: " << vc_calib_hor << " vc_calib_ver: " << vc_calib_ver << endl;
        cout << "c_calib_hor: " << c_calib_hor << " c_calib_ver: " << c_calib_ver << endl;
    }

    if (d_hor > 0 && d_ver > 0) {
        if (d_hor < vc_calib_hor && d_ver < vc_calib_ver) {
            return 1;
        } else if (d_hor < c_calib_hor && d_ver < c_calib_ver) {
            return 2;
        }
    }*/

    if (close_dist <= 100 && close_dist > 50) {
        return 2;
    } else if (close_dist <= 50 && close_dist > 10) {
        return 1;
    }

    return 0;
}

