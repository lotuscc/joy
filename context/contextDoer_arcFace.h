#ifndef CONTEXTDOER_ARCFACE_H
#define CONTEXTDOER_ARCFACE_H
#include "baseContext.h"

class ContextArcFace : public BaseContext {
public:
    ContextArcFace(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    void preProcess(cv::Mat& inferData)
    {
        cv::Mat outData;
        cv::dnn::blobFromImage(inferData, outData, 1.0 / 127.5, cv::Size(inputW_, inputH_), cv::Scalar(127.5, 127.5, 127.5), true, false);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    void postProcess(cv::Mat& inferData, yoloResult& out)
    {
        int featureSize = outputDims_[0].d[1];
        std::vector<float> embedding_norm;
        float* base = cpuOutput_[0];
        for (int i = 0; i < featureSize; ++i) {
            embedding_norm.push_back(base[i]);
        }
        cv::normalize(embedding_norm, embedding_norm); // l2 normalize

        out.faceFeature.feature = embedding_norm;
    }
};

#endif // CONTEXTDOER_ARCFACE_H
