#ifndef EXAMPLE_ARCFACE_H
#define EXAMPLE_ARCFACE_H
#include "contextDoer_arcFace.h"
#include "globalComm.h"
#include "templateModel.h"

void example_arcFace_main()
{

    // TemplateModel<ContextArcFace>::saveOnnxModel("/home/lotuscc/work/ms1mv3_arcface_r100.onnx", "/home/lotuscc/work/ms1mv3_arcface_r100.engine");

    std::string pngPath1 = "/home/lotuscc/gitlib/lite.ai.toolkit/examples/lite/resources/test_lite_arcface_resnet_1.png";
    std::string pngPath2 = "/home/lotuscc/gitlib/lite.ai.toolkit/examples/lite/resources/test_lite_arcface_resnet_0.png";
    cv::Mat inMat1 = cv::imread(pngPath1);
    cv::Mat inMat2 = cv::imread(pngPath2);
    // cv::resize(inMat, inMat, cv::Size(640, 640));

    TemplateModel<ContextArcFace> arcFace("arcFace", "/home/lotuscc/work/ms1mv3_arcface_r100.engine");
    auto infer = arcFace.getContextInfer();
    yoloResult out1;
    yoloResult out2;
    for (int j = 0; j < 1; ++j) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat1, out1);
        infer->Inference(inMat2, out2);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }

    cv::Mat feature1 = cv::Mat(1, 512, CV_32FC1, out1.faceFeature.feature.data());
    cv::Mat feature2 = cv::Mat(1, 512, CV_32FC1, out2.faceFeature.feature.data());
    cv::Mat x = feature1 * feature2.t();

    cv::FileStorage fsWriter("/home/lotuscc/work/m.xml", cv::FileStorage::WRITE);
    fsWriter << "feature" << x;
    fsWriter.release();

    float cos = x.at<float>(0, 0);
    float cost = 0.0f;
    for (int i = 0; i < out1.faceFeature.feature.size(); ++i) {
        cost += out1.faceFeature.feature[i] * out2.faceFeature.feature[i];
    }

    std::cout << cos << " exit \n";
}
#endif // EXAMPLE_ARCFACE_H
