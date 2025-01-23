#ifndef CUDABANDWIDTHTEST_HPP
#define CUDABANDWIDTHTEST_HPP
#include <boost/lockfree/queue.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
struct SpeedMetrics {
    SpeedMetrics(double _ms, double _band)
        : ms(_ms)
        , band(_band)
    {
    }
    SpeedMetrics() { }
    double ms;
    double band;
};

class cudaBandWidthTest {
private:
    cudaBandWidthTest() { }
    boost::lockfree::queue<SpeedMetrics, boost::lockfree::capacity<1000>> speedMetricsQueue_;

    void workPrint()
    {
        while (1) {
            if (speedMetricsQueue_.empty()) {
                usleep(100);
            }
            SpeedMetrics m;
            if (speedMetricsQueue_.pop(m)) {
                printf("cost %.5f ms, speed: %.5f GB/S \n", m.ms, m.band);
            } else {
                usleep(100);
            }
        }
    }

    bool pushData(const SpeedMetrics& m)
    {
        if (speedMetricsQueue_.push(m)) {
            return true;
        }
        return false;
    }

    static void workPrintThread()
    {
        cudaBandWidthTest::GetInstance()->workPrint();
    }

    void runBandTest()
    {
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
            return;
        }

        size_t memsize = 10 * 1024 * 1024;

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);

            std::cout << "Device " << i << " name: " << deviceProp.name << std::endl;
            std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

            std::thread doThread = std::thread(cudaBandWidthTest::bandTestThread, i, memsize);
            doThread.detach();
        }
    }

    static void bandTestThread(int device_id, size_t bytes)
    {
        cudaSetDevice(device_id);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        void* hostRawPtr = nullptr;
        // cudaMallocHost(&hostRawPtr, bytes);
        hostRawPtr = malloc(bytes);
        memset(hostRawPtr, 'x', bytes);

        void* deviceRawPtr = nullptr;
        cudaMalloc(&deviceRawPtr, bytes);
        cudaMemset(deviceRawPtr, 'x', bytes);

        while (1) {
            struct timeval start, end;
            gettimeofday(&start, NULL);
            cudaMemcpy(deviceRawPtr, hostRawPtr, bytes, cudaMemcpyHostToDevice);
            gettimeofday(&end, NULL);

            double duration = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
            duration /= 1000;
            double transfer_speed = static_cast<double>(bytes) * 1000 / 1024.0 / 1024.0 / 1024.0 / duration; // GS/S

            // printf("cost %.5f ms, speed: %.5f GB/S \n", duration, transfer_speed);
            cudaBandWidthTest::GetInstance()->pushData(SpeedMetrics { duration, transfer_speed });
        }

        cudaStreamDestroy(stream);
    }

public:
    static cudaBandWidthTest* GetInstance()
    {
        static cudaBandWidthTest* pInstance = new cudaBandWidthTest();
        if (nullptr != pInstance) {
            return pInstance;
        }
        std::cout << "error ! \n";
        return pInstance;
    }

    void Run()
    {
        std::thread doThreadPrint = std::thread(cudaBandWidthTest::workPrintThread);
        doThreadPrint.detach();

        runBandTest();

        while (1) {
        }
    }
};

#endif // CUDABANDWIDTHTEST_HPP
