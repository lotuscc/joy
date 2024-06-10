#ifndef CONTEXTDOER_YOLOV8DETECT_H
#define CONTEXTDOER_YOLOV8DETECT_H
#include "baseContext.h"
#include "preProcess.h"

class ContextYolov8Detect : public BaseContext {
public:
    ContextYolov8Detect(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    virtual void preProcess(cv::Mat& inferData) override
    {
        cv::Mat outData;
        preProcessYolov8Detect(inferData, inputH_, inputW_, outData);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    virtual void postProcess(cv::Mat& inferData, yoloResult& out) override
    {
        x_factor_ = (float)inferData.cols / (float)inputW_;
        y_factor_ = (float)inferData.rows / (float)inputH_;

        int h = outputDims_[0].d[1];
        int grid = outputDims_[0].d[2];

        cv::Mat data(h, grid, CV_32FC1, cpuOutput_[0]);
        cv::Mat dataOut;
        cv::transpose(data, dataOut);

        std::vector<cv::Rect> boxVec;
        std::vector<float> confVec;
        std::vector<int> clsVec;

        for (size_t i = 0; i < dataOut.rows; ++i) {
            float maxConf = 0.5f;
            int maxCls = -1;
            const float* pRow = dataOut.ptr<float>(i);
            for (int j = 4; j < dataOut.cols; ++j) {
                if (pRow[j] > maxConf) {
                    maxConf = pRow[j];
                    maxCls = j;
                }
            }
            if (maxConf > 0.5f) {
                float x = pRow[0];
                float y = pRow[1];
                float w = pRow[2];
                float h = pRow[3];
                int left = int((x - 0.5 * w) * x_factor_);
                int top = int((y - 0.5 * h) * y_factor_);
                int width = int(w * x_factor_);
                int height = int(h * y_factor_);

                boxVec.push_back(cv::Rect(left, top, width, height));
                confVec.push_back(maxConf);
                clsVec.push_back(maxCls - 4);
            }
        }

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxVec, confVec, 0.5f, 0.5f, nmsResult);

        for (size_t i = 0; i < nmsResult.size(); ++i) {
            int idx = nmsResult[i];
            yoloDetection det;
            det.box = boxVec[idx];
            det.cls = clsVec[idx];
            det.conf = confVec[idx];
            out.detectVec.emplace_back(det);
        }
    }
};
#endif // CONTEXTDOER_YOLOV8DETECT_H
