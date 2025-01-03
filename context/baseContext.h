#ifndef BASECONTEXT_H
#define BASECONTEXT_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "globalComm.h"
#include "preProcess.h"
#include <opencv2/opencv.hpp>
#include <vector>

#include "baseTensor.h"

using namespace nvinfer1;

class BaseContext {
public:
    BaseContext(IExecutionContext* context)
        : context_(context)
    {
        Malloc();
    }
    ~BaseContext()
    {
        FreeMalloc();
    }

    virtual void preProcess(cv::Mat& inferData)
    {
        LOG(INFO) << "preProcess is empty !";
    }
    virtual void postProcess(cv::Mat& inferData, yoloResult& out)
    {
        LOG(INFO) << "postProcess is empty !";
    }

    bool Inference(cv::Mat& inferData, yoloResult& out)
    {
        preProcess(inferData);

        cudaStreamSynchronize(stream_);
        cudaMemcpyAsync(inputTensor.device(), inputTensor.host(), inputTensor.bytes(), cudaMemcpyHostToDevice, stream_);

        context_->setTensorAddress(inputTensor.name().c_str(), inputTensor.device());
        for (int i = 0; i < outputTensor.size(); ++i) {
            context_->setTensorAddress(outputTensor[i].name().c_str(), outputTensor[i].device());
        }
        context_->enqueueV3(stream_);
        for (int i = 0; i < outputTensor.size(); ++i) {
            cudaMemcpyAsync(outputTensor[i].host(), outputTensor[i].device(), outputTensor[i].bytes(), cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);

        postProcess(inferData, out);

        return true;
    }

    void Malloc()
    {
        cudaStreamCreate(&stream_);

        const ICudaEngine& engine = context_->getEngine();

        int num = engine.getNbIOTensors();
        assert(num >= 2);

        std::string inputName = engine.getIOTensorName(0);
        Dims dimIn = engine.getTensorShape(inputName.c_str());
        DataType dataTypeIn = engine.getTensorDataType(inputName.c_str());

        inputTensor.init(inputName, dimIn, dataTypeIn);

        batSize_ = dimIn.d[0];
        inputC_ = dimIn.d[1];
        inputW_ = dimIn.d[2];
        inputH_ = dimIn.d[3];

        for (int i = 1; i < num; ++i) {
            std::string name = engine.getIOTensorName(i);
            Dims dim = engine.getTensorShape(name.c_str());
            DataType dataTypeOut = engine.getTensorDataType(name.c_str());

            Tensor tensor(name, dim, dataTypeOut);
            outputTensor.push_back(std::move(tensor));

            std::cout << name << "\n";
        }
    }

    void FreeMalloc()
    {
    }

    float x_factor_ = 1.0f;
    float y_factor_ = 1.0f;

    int batSize_; // 模型输入尺寸
    int inputC_;
    int inputW_;
    int inputH_;

    Tensor inputTensor;
    std::vector<Tensor> outputTensor;

    IExecutionContext* context_;
    cudaStream_t stream_;
};

#endif // BASECONTEXT_H
