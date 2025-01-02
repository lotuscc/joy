#ifndef BASETENSOR_H
#define BASETENSOR_H
#include <NvInferRuntime.h>

#include "macro.h"
#include <cstddef>
#include <cstdint>
#include <string>

class Tensor {
private:
    struct HostDeleter {
        void operator()(void* ptr) const
        {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFreeHost(ptr));
            }
        }
    };

    struct DeviceDeleter {
        void operator()(void* ptr) const
        {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    };
    std::unique_ptr<void, HostDeleter> hostPtr_;
    std::unique_ptr<void, DeviceDeleter> devicePtr_;

    std::string name_ {};
    nvinfer1::Dims dims_ {};
    int64_t bytes_ {};
    nvinfer1::DataType dataType_;

    inline int getSize(const nvinfer1::Dims& dim) const
    {
        int size = 1;
        for (int i = 0; i < dim.nbDims; ++i) {
            size *= dim.d[i];
        }
        return size;
    }
    size_t getDataTypeSize(nvinfer1::DataType dataType) const
    {
        switch (dataType) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4U;
        case nvinfer1::DataType::kHALF:
            return 2U;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return 1U;
        default:
            throw std::runtime_error("Unsupported data type");
        }
        return 0;
    }

public:
    Tensor()
        : hostPtr_(nullptr)
        , devicePtr_(nullptr)
        , bytes_(0)
    {
    }

    Tensor(const std::string& name, const nvinfer1::Dims& dims, nvinfer1::DataType dataType)
        : name_(name)
        , dims_(dims)
        , dataType_(dataType)
        , bytes_(getSize(dims_) * getDataTypeSize(dataType))
    {
        Malloc();
    }

    ~Tensor() = default;

    // 禁止复制
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 允许移动
    Tensor(Tensor&& other) noexcept
        : hostPtr_(std::move(other.hostPtr_))
        , devicePtr_(std::move(other.devicePtr_))
        , name_(std::move(other.name_))
        , dims_(other.dims_)
        , bytes_(other.bytes_)
        , dataType_(other.dataType_)
    {
        other.bytes_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept
    {
        if (this != &other) {
            hostPtr_ = std::move(other.hostPtr_);
            devicePtr_ = std::move(other.devicePtr_);
            name_ = std::move(other.name_);
            dims_ = other.dims_;
            bytes_ = other.bytes_;
            dataType_ = other.dataType_;
            other.bytes_ = 0;
        }
        return *this;
    }

    void init(const std::string& name, const nvinfer1::Dims& dims, nvinfer1::DataType dataType)
    {
        name_ = name;
        dims_ = dims;
        dataType_ = dataType;
        bytes_ = getSize(dims_) * getDataTypeSize(dataType_);

        Free();
        Malloc();
    }

    void Malloc()
    {
        if (bytes_ > 0) {
            hostPtr_.reset();
            devicePtr_.reset();
            // 分配主机内存
            void* hostRawPtr = nullptr;
            CUDA_CHECK(cudaMallocHost(&hostRawPtr, bytes_));
            hostPtr_.reset(hostRawPtr);

            // 分配设备内存
            void* deviceRawPtr = nullptr;
            CUDA_CHECK(cudaMalloc(&deviceRawPtr, bytes_));
            devicePtr_.reset(deviceRawPtr);
        }
    }

    void Free()
    {
        hostPtr_.reset();
        devicePtr_.reset();
    }

    void* host()
    {
        return hostPtr_.get();
    }

    void* device()
    {
        return devicePtr_.get();
    }

    const std::string& name() const
    {
        return name_;
    }

    const nvinfer1::Dims& dims() const
    {
        return dims_;
    }

    int64_t bytes() const
    {
        return bytes_;
    }

    nvinfer1::DataType dataType() const
    {
        return dataType_;
    }
};
#endif // BASETENSOR_H
