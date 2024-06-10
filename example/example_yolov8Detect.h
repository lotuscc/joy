#ifndef EXAMPLE_YOLOV8DETECT_H
#define EXAMPLE_YOLOV8DETECT_H
#include "baseModel.h"
#include "contextDoer_yolov8Detect.h"
#include "globalComm.h"
#include "model_yolov8Detect.h"
#include "templateModel.h"

void example_yoloDetect_mainVer2()
{
    std::string pngPath = "/home/lotuscc/work/bus.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    Yolov8Detect yolov8n("yolov8n", "/home/lotuscc/work/yolov8n.engine");
    yoloResult out;
    auto infer = yolov8n.getInfer();
    for (int i = 0; i < 1; ++i) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }
    for (int i = 0; i < out.detectVec.size(); ++i) {
        cv::rectangle(inMat, out.detectVec[i].box, cv::Scalar(255, 155, 0));
    }
    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
}

void example_yoloDetect_main()
{
    std::string pngPath = "/home/lotuscc/work/bus.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    TemplateModel<ContextYolov8Detect> yolov8n("yolov8n", "/home/lotuscc/work/yolov8n.engine");
    yoloResult out;
    auto infer = yolov8n.getContextInfer();
    for (int i = 0; i < 1; ++i) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }
    for (int i = 0; i < out.detectVec.size(); ++i) {
        cv::rectangle(inMat, out.detectVec[i].box, cv::Scalar(255, 155, 0));
    }
    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
}

#endif // EXAMPLE_YOLOV8DETECT_H
