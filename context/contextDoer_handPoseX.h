#ifndef CONTEXTDOER_HANDPOSEX_H
#define CONTEXTDOER_HANDPOSEX_H
#include "baseContext.h"
#include "preProcess.h"

class ContextHandPoseX : public BaseContext {
public:
    ContextHandPoseX(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    void preProcess(cv::Mat& inferData)
    {
        cv::Mat outData;
        preProcessHandposeX(inferData, outData);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    void postProcess(cv::Mat& inferData, yoloResult& out)
    {
        postProcessHandposeX(cpuOutput_[0], out.pointVec, inputH_, inputW_);
    }
};

#endif // CONTEXTDOER_HANDPOSEX_H
