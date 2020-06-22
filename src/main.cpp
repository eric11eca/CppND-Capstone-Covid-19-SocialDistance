#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <memory>

#include "Render.h"
#include "VideoCapture.h"
#include "Detector.h"

const char *keys =
    "{help  h  | |examples: --video=../data/town.avi}"
    "{video v  |<none>| path to input video   }";

std::string video_out = "output.avi";

std::string config_path {"../model/yolov3.cfg"};
std::string weights_path {"../model/yolov3.weights"};
std::string classes_path {"../model/coco.names"};

int main(int argc, char **argv) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection in OpenCV.");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    } else if (parser.has("video")) {
        std::string input = parser.get<std::string>("video");

        if (input == "") {
            input = "stream";
        } else { 
            std::ifstream video_stream(input);
            if (!video_stream) {
                std::cout << "The given path does not exit. \n";
                return 0;
            }
        }

        std::unique_ptr<VideoCapture> capture = std::make_unique<VideoCapture>(input);
        std::unique_ptr<Detector> detector = std::make_unique<Detector>(config_path, weights_path, classes_path);
        std::unique_ptr<Render> render = std::make_unique<Render>();

        cv::VideoWriter video(video_out, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 20, 
                cv::Size(capture->GetFrameWidth(), capture->GetFrameHeight()));

        while(1) {
            cv::Mat frame, blob;
            capture->ReadFrame(frame);
       
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, cv::Size(1280,720), cv::INTER_CUBIC);
            detector.get()->DetectObjects(frame);

            vector<cv::Rect> boxes = detector.get()->getBoundingBoxes();
            vector<int> indices = detector.get()->GetIndicies();
            vector<int> status = detector.get()->GetStatus();
            vector<int> report = detector.get()->GetReport(); 
            vector<vector<cv::Point>> close_pair = detector.get()->GetRiskPair();
            vector<vector<cv::Point>> s_close_pair = detector.get()->GetHighRiskPair();

            cv::Mat detectedFrame = render.get()->RenderResult(
                frame, boxes, indices, status, report, close_pair, s_close_pair);
            imshow("Covid19-Social-Distance", detectedFrame);

            video.write(detectedFrame);

            char c = (char)cv::waitKey(25);
            if(c==27) break;
        }

        capture->Release();
        video.release();
        cv::destroyAllWindows();

        return 0;
    } else {
        std::cout << "Invalid arguments. \n";
    }
}