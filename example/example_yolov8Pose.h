#ifndef EXAMPLE_YOLOV8POSE_H
#define EXAMPLE_YOLOV8POSE_H
#include "baseModel.h"
#include "contextDoer_yolov8Detect.h"
#include "contextDoer_yolov8Pose.h"
#include "globalComm.h"
#include "templateModel.h"

void example_yolov8Pose_main()
{

    // TemplateModel<ContextYolov8Pose>::saveOnnxModel("/home/lotuscc/work/yolov8n-pose.onnx", "/home/lotuscc/work/yolov8n-pose.engine");

    std::string pngPath = "/home/lotuscc/work/bus.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    TemplateModel<ContextYolov8Pose> yolov8nPose("yolov8nPose", "/home/lotuscc/work/yolov8n-pose.engine");
    yoloResult out;
    auto infer = yolov8nPose.getContextInfer();
    for (int i = 0; i < 1; ++i) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }
    for (int i = 0; i < out.poseVec.size(); ++i) {
        cv::rectangle(inMat, out.poseVec[i].box, cv::Scalar(255, 155, 0));
        for (cv::Point& p : out.poseVec[i].keyPoints) {
            cv::circle(inMat, p, 2, cv::Scalar(255, 155, 0), 1);
        }
    }
    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
}

#endif // EXAMPLE_YOLOV8POSE_H
