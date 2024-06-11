#ifndef CONTEXTDOER_SCRFD_H
#define CONTEXTDOER_SCRFD_H
#include "baseContext.h"

class ContextScrfd : public BaseContext {
public:
    ContextScrfd(IExecutionContext* context)
        : BaseContext(context)
    {
    }
    virtual void preProcess(cv::Mat& inferData) override
    {
        cv::Mat outData;
        cv::dnn::blobFromImage(inferData, outData, 1.0 / 128.0, cv::Size(inputW_, inputH_), cv::Scalar(127.5, 127.5, 127.5), true, false);
        memcpy(cpuInput_, outData.data, inputC_ * inputW_ * inputH_ * sizeof(float));
    }

    void generate_bboxes_kps(std::vector<cv::Rect>& boxVec, std::vector<float>& confVec, std::vector<std::vector<cv::Point>>& landMarksVec, float* baseConf, float* baseBox, float* baselandMarks, int stride, int grid)
    {

        const int nw = inputW_ / stride;
        const int nh = inputH_ / stride;
        std::vector<cv::Point> kps;
        for (int j = 0; j < nh; ++j) {
            for (int i = 0; i < nw; ++i) {
                for (int k = 0; k < 2; ++k) {
                    kps.emplace_back(i, j);
                }
            }
        }
        assert(kps.size() == grid);
        for (int i = 0; i < grid; ++i) {
            const float conf = baseConf[i];
            if (conf > 0.5f) {
                const float x1 = (kps[i].x - baseBox[i * 4 + 0]) * stride * x_factor_;
                const float y1 = (kps[i].y - baseBox[i * 4 + 1]) * stride * y_factor_;
                const float x2 = (kps[i].x + baseBox[i * 4 + 2]) * stride * x_factor_;
                const float y2 = (kps[i].y + baseBox[i * 4 + 3]) * stride * y_factor_;
                boxVec.emplace_back(x1, y1, x2 - x1, y2 - y1);
                confVec.push_back(conf);
                std::vector<cv::Point> landMarks;
                for (int k = 0; k < 10; k += 2) {
                    float x = (kps[i].x + baselandMarks[i * 10 + k + 0]) * stride * x_factor_;
                    float y = (kps[i].y + baselandMarks[i * 10 + k + 1]) * stride * y_factor_;
                    landMarks.emplace_back(x, y);
                }
                landMarksVec.push_back(landMarks);
            }
        }
    }

    virtual void postProcess(cv::Mat& inferData, yoloResult& out) override
    {
        x_factor_ = (float)inferData.cols / (float)inputW_;
        y_factor_ = (float)inferData.rows / (float)inputH_;

        std::vector<cv::Rect> boxVec;
        std::vector<float> confVec;
        std::vector<std::vector<cv::Point>> landMarksVec;

        int grid8 = outputDims_[0].d[1];
        generate_bboxes_kps(boxVec, confVec, landMarksVec, cpuOutput_[0], cpuOutput_[1], cpuOutput_[2], 8, grid8);

        int grid16 = outputDims_[3].d[1];
        generate_bboxes_kps(boxVec, confVec, landMarksVec, cpuOutput_[3], cpuOutput_[4], cpuOutput_[5], 16, grid16);

        int grid32 = outputDims_[6].d[1];
        generate_bboxes_kps(boxVec, confVec, landMarksVec, cpuOutput_[6], cpuOutput_[7], cpuOutput_[8], 32, grid32);
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxVec, confVec, 0.5f, 0.5f, nmsResult);

        for (size_t i = 0; i < nmsResult.size(); ++i) {
            int idx = nmsResult[i];

            scrfdFace det;
            det.box = boxVec[idx];
            det.matchBox = getMatchFaceRect(boxVec[idx], inferData.cols, inferData.rows);
            det.conf = confVec[idx];
            det.landMarks = landMarksVec[idx];

            out.faceVec.emplace_back(det);
        }

        std::cout << "exit \n";
    }

    cv::Rect getMatchFaceRect(const cv::Rect& box, int maxX, int maxY)
    {
        int x0 = std::max(0, box.x - box.width / 2);
        int y0 = std::max(0, box.y - box.height / 2);
        int x1 = std::min(box.x + box.width * 3 / 2, maxX);
        int y1 = std::min(box.y + box.height * 3 / 2, maxY);

        cv::Rect retRect(x0, y0, std::abs(x1 - x0), std::abs(y1 - y0));

        return retRect;
    }
};
#endif // CONTEXTDOER_SCRFD_H
