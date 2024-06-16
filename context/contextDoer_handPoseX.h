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
        cv::dnn::blobFromImage(inferData, outData, 1.0 / 255.0, cv::Size(inputH_, inputW_), cv::Scalar(128, 128, 128), false, false);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    void postProcess(cv::Mat& inferData, yoloResult& out)
    {
        x_factor_ = (float)inferData.cols / (float)inputW_;
        y_factor_ = (float)inferData.rows / (float)inputH_;

        // postProcessHandposeX(cpuOutput_[0], out.pointVec, inputH_, inputW_);
        float* baseData = cpuOutput_[0];
        for (int i = 0; i < 21; ++i) {
            int x = baseData[i * 2] * inputW_ * x_factor_;
            int y = baseData[i * 2 + 1] * inputH_ * y_factor_;
            out.pointVec.emplace_back(cv::Point(x, y));
        }
    }
};

#endif // CONTEXTDOER_HANDPOSEX_H
