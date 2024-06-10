#ifndef EXAMPLE_YOLOV8CLS_H
#define EXAMPLE_YOLOV8CLS_H
#include "baseModel.h"
#include "contextDoer_yolov8Cls.h"
#include "globalComm.h"
#include "templateModel.h"

void example_yolov8Cls_main()
{

    // TemplateModel<ContextYolov8Cls>::saveOnnxModel("/home/lotuscc/work/yolov8n-cls.onnx", "/home/lotuscc/work/yolov8n-cls.engine");

    std::string pngPath = "/home/lotuscc/work/basketball.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    TemplateModel<ContextYolov8Cls> yolov8nCls("yolov8nCls", "/home/lotuscc/work/yolov8n-cls.engine");
    auto infer = yolov8nCls.getContextInfer();
    yoloResult out;
    for (int j = 0; j < 1000; ++j) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }

    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
}
#endif // EXAMPLE_YOLOV8CLS_H
