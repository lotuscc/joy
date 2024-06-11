#ifndef GLOBALCOMM_H
#define GLOBALCOMM_H
#include "NvInfer.h"
#include <g3log/g3log.hpp>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace nvinfer1;

// extern std::map<std::string, ICudaEngine*> g_EngineList;
struct yoloCls {
    int clsIndex = -1;
    float conf = 0.1f;
};

struct yoloDetection {
    int cls = -1;
    float conf = 0.1f;
    cv::Rect box;
};

struct yoloPose {
    cv::Rect box;
    float conf = 0.1f;
    std::vector<cv::Point> keyPoints;
};

struct scrfdFace {
    cv::Rect box;
    cv::Rect matchBox;
    float conf = 0.1f;
    std::vector<cv::Point> landMarks;
};

struct ArcfaceFeature {
    std::vector<float> feature;
};

struct yoloResult {
    std::vector<scrfdFace> faceVec;
    std::vector<yoloPose> poseVec;
    std::vector<yoloDetection> detectVec;
    std::vector<cv::Point> pointVec;
    yoloCls cls;
    ArcfaceFeature faceFeature;
    void clear()
    {
        faceVec.clear();
        poseVec.clear();
        detectVec.clear();
        pointVec.clear();
        faceFeature.feature.clear();
    }
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        // if (severity <= Severity::kWARNING) {
        std::cout << msg << '\n';
        //}
    }
};

#endif // GLOBALCOMM_H
