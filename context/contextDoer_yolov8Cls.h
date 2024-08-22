#ifndef CONTEXTDOER_YOLOV8CLS_H
#define CONTEXTDOER_YOLOV8CLS_H
#include "baseContext.h"

class ContextYolov8Cls : public BaseContext {
public:
    ContextYolov8Cls(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    virtual void preProcess(cv::Mat& inferData) override
    {

        cv::Mat outData;
        cv::dnn::blobFromImage(inferData, outData, 1.0 / 255.0, cv::Size(inputW_, inputH_), cv::Scalar(0, 0, 0), true, false);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    virtual void postProcess(cv::Mat& inferData, yoloResult& out) override
    {
        float x_factor = (float)inferData.cols / (float)inputW_;
        float y_factor = (float)inferData.rows / (float)inputH_;
        int tolCls = outputDims_[0].d[1];
        float* base = cpuOutput_[0];
        float maxConf = 0.5f;
        int maxCls = -1;
        for (int i = 0; i < tolCls; ++i) {
            if (base[i] > maxConf) {
                maxConf = base[i];
                maxCls = i;
            }
        }
        out.cls.clsIndex = maxCls;
        out.cls.conf = maxConf;
    }
};
#endif // CONTEXTDOER_YOLOV8CLS_H
