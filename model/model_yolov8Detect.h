#ifndef MODEL_YOLOV8DETECT_H
#define MODEL_YOLOV8DETECT_H

#include "baseModel.h"
#include "contextDoer_yolov8Detect.h"
#include "globalComm.h"

class Yolov8Detect : baseModel {
public:
    Yolov8Detect(const std::string& name, const std::string& enginePath)
        : baseModel(name, enginePath)
    {
    }

    virtual std::unique_ptr<BaseContext> getInfer()
    {
        if (engine_ != nullptr) {
            return std::unique_ptr<BaseContext>(new ContextYolov8Detect(engine_->createExecutionContext()));
        }
        return nullptr;
    }
};

#endif // MODEL_YOLOV8DETECT_H
