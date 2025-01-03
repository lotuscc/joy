#ifndef GPUMANAGER_H
#define GPUMANAGER_H
#include "gpuThread.h"

class GPUManager {
private:
    std::vector<std::unique_ptr<GPUThread>> threads_;
    int numGpus_;

    GPUManager()
    {
        // 获取可用的 GPU 数量
        CUDA_CHECK(cudaGetDeviceCount(&numGpus_));
        for (int i = 0; i < numGpus_; ++i) {
            threads_.emplace_back(std::make_unique<GPUThread>(i));
        }
    }

    ~GPUManager() = default;

    // 禁止复制
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;

    // 禁止移动
    GPUManager(GPUManager&&) = delete;
    GPUManager& operator=(GPUManager&&) = delete;

public:
    static GPUManager& getInstance()
    {
        // 使用函数局部静态变量来实现单例模式
        static GPUManager instance;
        return instance;
    }

    //
    void enqueueBy()
    {
    }

    void enqueueTask(int gpuId, std::function<void()> task)
    {
        if (gpuId >= 0 && gpuId < numGpus_) {
            threads_[gpuId]->enqueue(task);
        } else {
            std::cerr << "Invalid GPU ID: " << gpuId << std::endl;
        }
    }

    int getNumGpus() const
    {
        return numGpus_;
    }
};
#endif // GPUMANAGER_H
