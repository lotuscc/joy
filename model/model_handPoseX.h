#ifndef MODEL_HANDPOSEX_H
#define MODEL_HANDPOSEX_H
#include "baseModel.h"
#include "contextDoer_handPoseX.h"
#include "globalComm.h"

class HandPoseX : baseModel {
public:
    HandPoseX(const std::string& name, const std::string& enginePath)
        : baseModel(name, enginePath)
    {
    }
    virtual std::unique_ptr<BaseContext> getInfer()
    {
        if (engine_ != nullptr) {
            return std::unique_ptr<BaseContext>(new ContextHandPoseX(engine_->createExecutionContext()));
        }
        return nullptr;
    }
};

#endif // MODEL_HANDPOSEX_H
