#ifndef EXAMPLE_HANDPOSEX_H
#define EXAMPLE_HANDPOSEX_H
#include "baseModel.h"
#include "contextDoer_handPoseX.h"
#include "globalComm.h"
#include "templateModel.h"

void example_handPoseX_main()
{
    std::string pngPath = "/home/lotuscc/gitProject/opencv/sample/img1.jpg";
    cv::Mat inMat = cv::imread(pngPath);
    cv::resize(inMat, inMat, cv::Size(256, 256));
    TemplateModel<ContextHandPoseX> handposeX("handposeX", "/home/lotuscc/gitProject/opencv/resnet_50_size-256-handposeX.engine");
    auto infer = handposeX.getContextInfer();
    yoloResult out;
    for (int i = 0; i < 1000; ++i) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }

    for (int i = 0; i < out.pointVec.size(); ++i) {
        cv::circle(inMat, out.pointVec[i], 4, cv::Scalar(255, 155, 0), 1);
    }
    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);
} ///////*/

#endif // EXAMPLE_HANDPOSEX_H
