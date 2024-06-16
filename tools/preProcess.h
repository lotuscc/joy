#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

static void preProcessHandposeX(const cv::Mat& bgrMat, cv::Mat& outMat)
{
    cv::Mat modelInput = bgrMat;
    // cv::Mat blob;
    cv::Size modelShape(256, 256);
    cv::dnn::blobFromImage(modelInput, outMat, 1.0 / 255.0, cv::Size(256, 256), cv::Scalar(128, 128, 128), false, false);
}
static void postProcessHandposeX(float* outData, std::vector<cv::Point>& pts, int inputH, int inputW)
{
    for (int i = 0; i < 21; ++i) {
        int x = outData[i * 2] * inputW;
        int y = outData[i * 2 + 1] * inputH;
        pts.emplace_back(cv::Point(x, y));
    }
}
// static void preProcessYolov8Detect(const cv::Mat& bgrMat, int outH, int outW, cv::Mat& outMat)
// {
//     cv::Mat modelInput = bgrMat;
//     cv::dnn::blobFromImage(modelInput, outMat, 1.0 / 255.0, cv::Size(outH, outW), cv::Scalar(0, 0, 0), true, false);
// }

// static void postProcessYolov8Detect(const cv::Mat& bgrMat, int outH, int outW, cv::Mat& outMat)
// {
//     cv::Mat modelInput = bgrMat;
//     cv::dnn::blobFromImage(modelInput, outMat, 1.0 / 255.0, cv::Size(outH, outW), cv::Scalar(0, 0, 0), true, false);
// }

static void inputDataPreproce(cv::Mat& mat, std::vector<float>& inData, int inputH, int inputW)
{
    auto* ptrB = inData.data();
    auto* ptrG = ptrB + inputH * inputW;
    auto* ptrR = ptrG + inputH * inputW;

    int cnt = 0;

    for (int i = 0; i < mat.rows; ++i) {
        auto* ptr = mat.ptr(i);
        for (int j = 0; j < mat.cols; ++j) {
            ptrB[cnt] = (ptr[3 * j] - 128.0) / 256.0;
            ptrG[cnt] = (ptr[3 * j + 1] - 128.0) / 256.0;
            ptrR[cnt] = (ptr[3 * j + 2] - 128.0) / 256.0;
            ++cnt;
        }
    }
}

static void inputDataPreproce(cv::Mat& mat, float* inData, int inputH, int inputW)
{
    auto* ptrB = inData;
    auto* ptrG = ptrB + inputH * inputW;
    auto* ptrR = ptrG + inputH * inputW;

    int cnt = 0;

    for (int i = 0; i < mat.rows; ++i) {
        auto* ptr = mat.ptr(i);
        for (int j = 0; j < mat.cols; ++j) {
            ptrB[cnt] = (ptr[3 * j] - 128.0) / 256.0;
            ptrG[cnt] = (ptr[3 * j + 1] - 128.0) / 256.0;
            ptrR[cnt] = (ptr[3 * j + 2] - 128.0) / 256.0;
            ++cnt;
        }
    }
}

#endif // PREPROCESS_H
