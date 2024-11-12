#ifndef EXAMPLE_YOLOV8SEG_H
#define EXAMPLE_YOLOV8SEG_H
#include "baseModel.h"
#include "contextDoer_yolov8Seg.h"
#include "globalComm.h"
#include "templateModel.h"

void example_yolov8Seg_main()
{

    // TemplateModel<ContextYolov8Seg>::saveOnnxModel("/home/lotuscc/work/yolov8n-seg.onnx", "/home/lotuscc/work/yolov8n-seg.engine");

    std::string pngPath = "/home/lotuscc/work/bus.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    TemplateModel<ContextYolov8Seg> yolov8nSeg("yolov8nSeg", "/home/lotuscc/work/yolov8n-seg.engine");
    yoloResult out;
    auto infer = yolov8nSeg.getContextInfer();
    for (int i = 0; i < 1000; ++i) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }
    for (int i = 0; i < out.segVec.size(); ++i) {
        cv::rectangle(inMat, out.segVec[i].box, cv::Scalar(255, 155, 0));
        inMat.setTo(cv::Scalar(125 * i % 255, 34 * i % 255, 74 * i % 255), out.segVec[i].mask);
    }
    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
}
#endif // EXAMPLE_YOLOV8SEG_H
