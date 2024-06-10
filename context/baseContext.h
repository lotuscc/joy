#ifndef BASECONTEXT_H
#define BASECONTEXT_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "globalComm.h"
#include "preProcess.h"
#include <opencv2/opencv.hpp>
#include <vector>

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

    static int getSize(Dims& dim)
    {
        int size = 1;
        for (int i = 0; i < dim.nbDims; ++i) {
            size *= dim.d[i];
        }
        return size;
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
        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        cudaMemcpyAsync(gpuInput_, cpuInput_, inputSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
        // context->enqueue(batchSize, buffers, stream, nullptr);

        context_->setTensorAddress(inputTensorName_.c_str(), gpuInput_);
        for (int i = 0; i < outputTensorNameVec_.size(); ++i) {
            context_->setTensorAddress(outputTensorNameVec_[i].c_str(), gpuOutput_[i]);
        }
        context_->enqueueV3(stream_);
        for (int i = 0; i < outputTensorNameVec_.size(); ++i) {
            cudaMemcpyAsync(cpuOutput_[i], gpuOutput_[i], outputSize_[i] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
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

        inputTensorName_ = engine.getIOTensorName(0);
        Dims dimIn = engine.getTensorShape(inputTensorName_.c_str());
        DataType dataTypeIn = engine.getTensorDataType(inputTensorName_.c_str());
        batSize_ = dimIn.d[0];
        inputC_ = dimIn.d[1];
        inputW_ = dimIn.d[2];
        inputH_ = dimIn.d[3];
        inputSize_ = batSize_ * inputC_ * inputW_ * inputH_;
        cpuInput_ = (float*)malloc(inputSize_ * sizeof(float));
        cudaMalloc(&gpuInput_, inputSize_ * sizeof(float));

        for (int i = 1; i < num; ++i) {
            std::string x = engine.getIOTensorName(i);
            Dims dim = engine.getTensorShape(x.c_str());
            DataType dataTypeOut = engine.getTensorDataType(x.c_str());
            int size = getSize(dim);
            outputSize_.push_back(size);
            outputTensorNameVec_.push_back(x);
            outputDims_.push_back(dim);
            float* pcpu = (float*)malloc(size * sizeof(float));
            cpuOutput_.push_back(pcpu);
            float* pgpu;
            cudaMalloc(&pgpu, size * sizeof(float));
            gpuOutput_.push_back(pgpu);
            std::cout << x << "\n";
        }
    }

    void FreeMalloc()
    {
        if (cpuInput_ != nullptr) {
            free(cpuInput_);
            cpuInput_ = nullptr;
        }
        if (gpuInput_ != nullptr) {
            cudaFree(gpuInput_);
            gpuInput_ = nullptr;
        }
        for (int i = 0; i < cpuOutput_.size(); ++i) {
            if (cpuOutput_[i] != nullptr) {
                free(cpuOutput_[i]);
                cpuOutput_[i] = nullptr;
            }
        }
        for (int i = 0; i < gpuOutput_.size(); ++i) {
            if (gpuOutput_[i] != nullptr) {
                cudaFree(gpuOutput_[i]);
                gpuOutput_[i] = nullptr;
            }
        }
    }

    float x_factor_ = 1.0f;
    float y_factor_ = 1.0f;

    int batSize_; // 模型输入尺寸
    int inputC_;
    int inputW_;
    int inputH_;

    int inputSize_;
    float* cpuInput_ = nullptr;
    float* gpuInput_ = nullptr;
    std::string inputTensorName_ = "keypoint_input";

    std::vector<int> outputSize_;
    std::vector<std::string> outputTensorNameVec_;
    std::vector<float*> cpuOutput_;
    std::vector<float*> gpuOutput_;
    std::vector<Dims> outputDims_;

    IExecutionContext* context_;
    cudaStream_t stream_;
};

#endif // BASECONTEXT_H
