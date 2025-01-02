#ifndef FFMPGEDECODERWITHCPU_H
#define FFMPGEDECODERWITHCPU_H
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}

#include "globalComm.h"
#include <atomic>
#include <condition_variable>
#include <functional> // For std::function
#include <g3log/g3log.hpp>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

class ffmpegDecoderWithCPU {
public:
    ffmpegDecoderWithCPU() = default;
    ~ffmpegDecoderWithCPU()
    {
        av_frame_free(&frame_);
        avcodec_free_context(&codecCtx_);
        avformat_close_input(&formatCtx_);
    }
    bool init(const std::string& path)
    {

	AVDictionary *opts = 0;
	av_dict_set(&opts, "rtsp_transport", "tcp", 0);

        if (avformat_open_input(&formatCtx_, path.c_str(), NULL, &opts) < 0) {
            LOG(WARNING) << "can not open file!";
            return false;
        }

        if (avformat_find_stream_info(formatCtx_, NULL) < 0) {
            LOG(WARNING) << "can not get stream infos! \n";
            return false;
        }
        for (int i = 0; i < formatCtx_->nb_streams; ++i) {
            AVStream* stream = formatCtx_->streams[i];
            if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            } else if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            } else {
            }
            LOG(INFO) << ("duration: %lld ms \n", stream->duration * 1000 / AV_TIME_BASE);
        }

        streamIndex_ = av_find_best_stream(formatCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (streamIndex_ < 0) {
            return false;
        }

        AVCodecParameters* codecParam = formatCtx_->streams[streamIndex_]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParam->codec_id);
        if (!codec) {
            LOG(WARNING) << "can not find codec!\n";
            return -1;
        }
        codecCtx_ = avcodec_alloc_context3(codec);
        if (!codecCtx_) {
            LOG(WARNING) << "can not alloc context!\n";
            return -1;
        }
        avcodec_parameters_to_context(codecCtx_,
            formatCtx_->streams[streamIndex_]->codecpar);
        if (avcodec_open2(codecCtx_, codec, NULL) < 0) {
            LOG(WARNING) << "can not open codec!\n";
            return -1;
        }
        av_init_packet(&packet_);
        frame_ = av_frame_alloc();

        send_thread_ = std::thread(&ffmpegDecoderWithCPU::send_packet_thread, this);
        receive_thread_ = std::thread(&ffmpegDecoderWithCPU::receive_frame_thread, this);

        send_thread_.detach();
        receive_thread_.detach();

        return true;
    }

    void start()
    {
        isStart_ = 1;
    }

    void stop()
    {
        isStart_ = 0;
    }

    bool isEnd()
    {
        if (isStart_ == 1) {
            return false;
        }
        return isEnd_ == true;
    }

    void setOnFrameCallback(std::function<void(const cv::Mat&)> callback)
    {
        onFrameCallback_ = callback;
    }

private:
    void send_packet_thread()
    {
        while (true) {
            if (isStart_ == 0 || isEnd_ == true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                LOG(INFO) << "sleep in send_packet_thread";
                continue;
            }

            const int ret = av_read_frame(formatCtx_, &packet_);
            if (ret < 0) {
                LOG(WARNING) << "av_read_frame error! end of file! \n";
                isEnd_ = true;
                continue;
            }
            if (packet_.stream_index != streamIndex_) {
                continue;
            }
            std::unique_lock<std::mutex> lock(codecMutex_);
            while (true) {
                int ret = avcodec_send_packet(codecCtx_, &packet_);
                if (ret == AVERROR(EAGAIN)) {
                    LOG(INFO) << "avcodec_send_packet EAGAIN \n";
                    codecCV_.wait(lock); // 等待接收线程完成
                } else if (ret == AVERROR_EOF || ret == AVERROR(EINVAL) || ret == AVERROR(ENOMEM)) {
                    LOG(WARNING) << "avcodec_send_packet ERROR \n";
                    break;
                } else if (ret == 0) {
                    LOG(WARNING) << "avcodec_send_packet success \n";
                    break;
                }
            }
            av_packet_unref(&packet_);
            codecCV_.notify_one(); // 通知接收线程可以继续
        }
    }

    void receive_frame_thread()
    {
        while (1) {
            if (isStart_ == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                LOG(INFO) << "sleep in receive_frame_thread";
                continue;
            }
            std::unique_lock<std::mutex> lock(codecMutex_);
            int ret = avcodec_receive_frame(codecCtx_, frame_);
            if (ret == AVERROR(EAGAIN)) {
                if (isEnd_ == true) {
                    isStart_ = 0;
                    continue;
                }
                codecCV_.wait(lock); // 等待发送线程发送新的包
            } else if (ret == AVERROR_EOF || ret == AVERROR(EINVAL)) {
                LOG(INFO) << "End of file reached during decoding.";
                if (isEnd_ == true) {
                    isStart_ = 0;
                }
                continue;
            } else if (ret < 0) {
                LOG(WARNING) << "avcodec_receive_frame error \n";
                break;
            } else if (ret == 0) {
                AVPixelFormat pix_fmt = (AVPixelFormat)frame_->format;
                // 检查像素格式
                if (pix_fmt == AV_PIX_FMT_YUV420P) {
                    LOG(INFO) << "The decoded frame is in YUV 4:2:0 format: " << pix_fmt;
                } else if (pix_fmt == AV_PIX_FMT_NV12) {
                    LOG(INFO) << "The decoded frame is in YUV 4:2:2 (UYVY) format: " << pix_fmt;
                } else {
                    LOG(INFO) << "The decoded frame is in another pixel format: " << pix_fmt;
                }
                const int width = frame_->width;
                const int height = frame_->height;
                cv::Mat yuvMat = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
                memcpy(yuvMat.data, frame_->data[0], width * height);
                memcpy(yuvMat.data + width * height, frame_->data[1],
                    width * height / 4);
                memcpy(yuvMat.data + width * height * 5 / 4, frame_->data[2],
                    width * height / 4);
                // 调用回调函数，传递cv::Mat
                if (onFrameCallback_) {
                    onFrameCallback_(yuvMat);
                }

                /*cv::Mat bgrMat;
                cv::cvtColor(yuvMat, bgrMat, cv::COLOR_YUV2BGR_I420);
                cv::imshow("win", bgrMat);
                cv::waitKey(10);*/

                LOG(INFO) << "codec a frame data, width: " << codecCtx_->width << " , height: " << codecCtx_->height;
                codecCV_.notify_one(); // 通知发送线程可以继续
            }
        }
    }

    AVFormatContext* formatCtx_ = NULL;
    AVCodecContext* codecCtx_ = NULL;
    AVPacket packet_;
    int streamIndex_ = 0;
    AVFrame* frame_;
    std::atomic<int> isStart_ = 0;
    std::atomic<bool> isEnd_ = false; // 新增的解码完成标志
    std::thread send_thread_;
    std::thread receive_thread_;

    std::mutex codecMutex_;
    std::condition_variable codecCV_;
    std::function<void(const cv::Mat&)> onFrameCallback_; // 新增的回调函数成员
};

#endif // FFMPGEDECODERWITHCPU_H
