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
#include "example/example_handPoseX.h"
#include "example/example_scrfd.h"
#include "example/example_yolov8Cls.h"
#include "example/example_yolov8Detect.h"
#include "example/example_yolov8Pose.h"

int main(int argc, char** argv)
{
    using namespace g3;
    std::unique_ptr<LogWorker> logworker { LogWorker::createLogWorker() };
    auto sinkHandle = logworker->addSink(std::make_unique<CustomSink>(),
        &CustomSink::ReceiveLogMessage);

    // initialize the logger before it can receive LOG calls
    initializeLogging(logworker.get());
    LOG(WARNING) << "This log call, may or may not happend before"
                 << "the sinkHandle->call below";

    // demo_main(argc, argv);
    demo_GPU_main();
    // example_yoloDetect_main();
    // example_handPoseX_main();
    // example_yolov8Cls_main();
    example_scrfdFace_main();

    // example_arcFace_main();

    std::cout << "end! \n";
}
