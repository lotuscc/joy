// main.cpp
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <memory>

#include "CustomSink.h"
#include "ffmpeg.h"

#include "preProcess.h"

#include "baseContext.h"
#include "contextDoer_handPoseX.h"
#include "contextDoer_yolov8Detect.h"

#include "baseModel.h"
#include "model_handPoseX.h"
#include "model_yolov8Detect.h"
#include "templateModel.h"
#include <sys/time.h>

#include "example/example_arcFace.h"
#include "example/example_ffmpeg.h"
#include "example/example_handPoseX.h"
#include "example/example_scrfd.h"
#include "example/example_yolov8Cls.h"
#include "example/example_yolov8Detect.h"
#include "example/example_yolov8Pose.h"
#include "ffmpgeDecoderWithCPU.h"
#include <boost/lockfree/queue.hpp>
#include <ffmpgeDecoderWithGPU.h>
struct DataPipe {
    cv::Mat mat;
    int x;
};

int main(int argc, char** argv)
{
    using namespace g3;
    std::unique_ptr<LogWorker> logworker { LogWorker::createLogWorker() };
    auto sinkHandle = logworker->addSink(std::make_unique<CustomSink>(),
        &CustomSink::ReceiveLogMessage);

    auto defaultHandler = logworker->addDefaultLogger(argv[0],
        "/home/lotuscc/Desktop/log");

    // initialize the logger before it can receive LOG calls
    initializeLogging(logworker.get());
    LOG(WARNING) << "This log call, may or may not happend before"
                 << "the sinkHandle->call below";

    ffmpge_main(argc, argv);

    ffmpegDecoderWithCPU* decoder = new ffmpegDecoderWithCPU();

    std::string x = "/home/lotuscc/Desktop/out.mov";

    // std::vector<cv::Mat> yuvFrames;

    // decoder->init(x);

    // decoder->setOnFrameCallback([&yuvFrames](const cv::Mat& yuvMat) {
    //     yuvFrames.push_back(yuvMat.clone());
    // });
    // decoder->start();

    // while (!decoder->isEnd())
    //     ;

    // demo_main(argc, argv);
    // demo_GPU_main();
    // example_yoloDetect_main();
    // example_handPoseX_main();
    // example_yolov8Cls_main();
    // example_scrfdFace_main();

    // example_arcFace_main();

    // boost::lockfree::queue<int> queue(128);

    // cv::Mat x;
    // queue.push(x);

    std::cout << "end! \n";
}
