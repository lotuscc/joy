#ifndef TEMPLATEMODEL_H
#define TEMPLATEMODEL_H
#include "baseContext.h"
#include "globalComm.h"
#include <NvOnnxParser.h>
#include <fstream>

template <typename DrierModel>
class TemplateModel {
public:
    TemplateModel(const std::string& name, const std::string& enginePath)
        : modelName_(name)
        , engineName_(enginePath)
    {
        loadModel(enginePath);
    }
    ~TemplateModel() { }

    bool loadModel(const std::string& engineFile)
    {

        if (engine_ != nullptr) {
            LOG(INFO) << engineName_ << "is exist !";
            return true;
        }

        IRuntime* runtime = nvinfer1::createInferRuntime(logger_);
        std::ifstream infile(engineFile, std::ios::binary);
        if (infile.good()) {
            infile.seekg(0, infile.end);
            size_t size = infile.tellg();
            infile.seekg(0, infile.beg);
            std::vector<char> trtModelStream(size);

            infile.read(trtModelStream.data(), size);
            infile.close();

            engine_ = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        } else {
            LOG(WARNING) << engineFile << "error";
            return false;
        }
        return true;
    }

    ICudaEngine* getEngine()
    {
        return engine_;
    }
    std::unique_ptr<BaseContext> getContextInfer()
    {
        if (engine_ != nullptr) {
            return std::unique_ptr<BaseContext>(new DrierModel(engine_->createExecutionContext()));
        }
        return nullptr;
    }

    static void saveOnnxModel(const std::string& onnxPath, const std::string& savePath)
    {
        Logger logger;
        // std::string onnxPath = "/home/lotuscc/gitProject/onnx_run/resnet_50_size-256-handposeX.onnx";

        IBuilder* builder = createInferBuilder(logger);

        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        INetworkDefinition* network = builder->createNetworkV2(flag);

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        parser->parseFromFile(onnxPath.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING));

        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cout << parser->getError(i)->desc() << std::endl;
        }

        IBuilderConfig* config = builder->createBuilderConfig();

        if (builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
        }

        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
        IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

        std::ofstream ofs(savePath, std::ios::out | std::ios::binary);
        ofs.write((char*)(serializedModel->data()), serializedModel->size());
        ofs.close();
        // IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        // ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
    }

    std::string modelName_;
    std::string engineName_;
    Logger logger_;

    ICudaEngine* engine_ = nullptr;

    // std::vector<BaseContext*> contextInferList_;
};
#endif // TEMPLATEMODEL_H
