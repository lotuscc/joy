#ifndef GPUTHREAD_H
#define GPUTHREAD_H

#include "NvInfer.h" // 包含 nvinfer1 头文件
#include "macro.h"
#include <condition_variable>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

class GPUThread {
private:
    int gpuId_;
    std::thread thread_;
    std::queue<std::function<void()>> taskQueue_;
    std::mutex queueMutex_;
    std::condition_variable condition_;
    bool stop_;

public:
    GPUThread(int gpuId)
        : gpuId_(gpuId)
        , stop_(false)
    {
        // 设置当前线程使用的 GPU
        CUDA_CHECK(cudaSetDevice(gpuId_));
        // 启动线程
        thread_ = std::thread(&GPUThread::run, this);
    }

    ~GPUThread()
    {
        // 停止线程
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            stop_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void enqueue(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            taskQueue_.push(task);
        }
        condition_.notify_one();
    }

    void run()
    {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                condition_.wait(lock, [this] { return !taskQueue_.empty() || stop_; });
                if (stop_ && taskQueue_.empty()) {
                    return;
                }
                task = std::move(taskQueue_.front());
                taskQueue_.pop();
            }
            task();
        }
    }
};

#endif
