#ifndef EXAMPLE_SCRFD_H
#define EXAMPLE_SCRFD_H
#include "contextDoer_scrfd.h"
#include "face_align.h"
#include "globalComm.h"
#include "templateModel.h"

void example_scrfdFace_main()
{

    // TemplateModel<ContextScrfd>::saveOnnxModel("/home/lotuscc/work/scrfd_2.5g_bnkps_shape640x640.onnx", "/home/lotuscc/work/scrfd_2.5g_bnkps_shape640x640.engine");

    std::string pngPath = "/home/lotuscc/gitlib/lite.ai.toolkit/examples/lite/resources/test_lite_face_detector_3.jpg";
    cv::Mat inMat = cv::imread(pngPath);

    // cv::resize(inMat, inMat, cv::Size(640, 640));

    TemplateModel<ContextScrfd> Scrfd("scrfdFace", "/home/lotuscc/work/scrfd_2.5g_bnkps_shape640x640.engine");
    auto infer = Scrfd.getContextInfer();
    yoloResult out;
    for (int j = 0; j < 1; ++j) {
        struct timeval start;
        struct timeval end;
        gettimeofday(&start, NULL);

        infer->Inference(inMat, out);

        gettimeofday(&end, NULL);
        float timeUse = (end.tv_sec - start.tv_sec) * 10000 + (end.tv_usec - start.tv_usec) / 100;
        LOG(INFO) << timeUse << " [0.1ms]";
    }
    for (int i = 0; i < out.faceVec.size(); ++i) {
        cv::Mat alignMat = FacePreprocess::align_face(inMat, out.faceVec[i].matchBox, out.faceVec[i].landMarks);
        static int x = 0;
        std::string path = "/home/lotuscc/work/face/";
        path = path + std::to_string(x++) + ".png";
        cv::imwrite(path, alignMat);
    }
    for (int i = 0; i < out.faceVec.size(); ++i) {
        cv::rectangle(inMat, out.faceVec[i].matchBox, cv::Scalar(255, 155, 0));
        for (cv::Point& p : out.faceVec[i].landMarks) {
            cv::circle(inMat, p, 2, cv::Scalar(255, 155, 0), 1);
        }
    }

    // 在窗口中显示图片
    cv::namedWindow("keypoint", 0);
    cv::imshow("keypoint", inMat);
    cv::waitKey(0);

    std::cout << "exit \n";
}
#endif // EXAMPLE_SCRFD_H
