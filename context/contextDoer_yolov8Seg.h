#ifndef CONTEXTDOER_YOLOV8SEG_H
#define CONTEXTDOER_YOLOV8SEG_H
#include "baseContext.h"

class ContextYolov8Seg : public BaseContext {
public:
    ContextYolov8Seg(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    virtual void preProcess(cv::Mat& inferData) override
    {
        cv::Mat outData;
        cv::dnn::blobFromImage(inferData, outData, 1.0 / 255.0, cv::Size(inputW_, inputH_), cv::Scalar(0, 0, 0), true, false);
        memcpy(inputTensor.host(), outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }
    virtual void postProcess(cv::Mat& inferData, yoloResult& out) override
    {
        x_factor_ = (float)inferData.cols / (float)inputW_;
        y_factor_ = (float)inferData.rows / (float)inputH_;

        int h = outputTensor[1].dims().d[1];
        int grid = outputTensor[1].dims().d[2];

        cv::Mat data(h, grid, CV_32FC1, outputTensor[1].host());
        cv::Mat dataOut;
        cv::transpose(data, dataOut);

        int cls = h - 4 - 32;

        std::vector<cv::Rect> boxVec;
        std::vector<float> confVec;
        std::vector<int> clsVec;
        std::vector<cv::Mat> maskVec;

        for (size_t i = 0; i < dataOut.rows; ++i) {
            float maxConf = 0.5f;
            int maxCls = -1;
            float* pRow = dataOut.ptr<float>(i);

            for (int k = 0; k < cls; ++k) {
                if (pRow[4 + k] > maxConf) {
                    maxConf = pRow[4 + k];
                    maxCls = k;
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

                cv::Mat padMask(1, 32, CV_32FC1, pRow + 4 + cls);

                boxVec.push_back(cv::Rect(left, top, width, height));
                confVec.push_back(maxConf);
                clsVec.push_back(maxCls);
                maskVec.push_back(padMask.clone());
            }
        }

        int padW = outputTensor[0].dims().d[2];
        int padH = outputTensor[0].dims().d[3];

        cv::Mat padMask(32, padW * padH, CV_32FC1, outputTensor[0].host());

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxVec, confVec, 0.5f, 0.5f, nmsResult);

        for (size_t i = 0; i < nmsResult.size(); ++i) {
            int idx = nmsResult[i];
            yoloSeg det;
            det.box = boxVec[idx];
            det.conf = confVec[idx];

            cv::Mat proMask = maskVec[idx] * padMask;
            cv::Mat maskMat = proMask > 0.00001f;
            cv::Mat mask = maskMat.reshape(1, { padW, padH });

            cv::Mat back(inferData.rows, inferData.cols, CV_8UC1);
            back.setTo(0);
            std::vector<cv::Point> pts;
            pts.push_back(cv::Point(det.box.x, det.box.y));
            pts.push_back(cv::Point(det.box.x + det.box.width, det.box.y));
            pts.push_back(cv::Point(det.box.x + det.box.width, det.box.y + det.box.height));
            pts.push_back(cv::Point(det.box.x, det.box.y + det.box.height));
            cv::fillPoly(back, pts, 255);

            cv::resize(mask, mask, cv::Size(back.cols, back.rows));
            det.mask = mask & back;
            out.segVec.emplace_back(det);
        }
    }
};
#endif // CONTEXTDOER_YOLOV8SEG_H
