#ifndef FFMPGEDECODERWITHGPU_H
#define FFMPGEDECODERWITHGPU_H
/*

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

// static AVBufferRef* hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;

static enum AVPixelFormat get_hw_format(AVCodecContext* ctx,
    const enum AVPixelFormat* pix_fmts)
{
    const enum AVPixelFormat* p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

class ffmpegDecoderWithGPU {
public:
    ffmpegDecoderWithGPU() = default;
    ~ffmpegDecoderWithGPU()
    {
        av_packet_free(&packet_);
        av_frame_free(&frame_);
        av_frame_free(&sw_frame_);
        avcodec_free_context(&codecCtx_);
        avformat_close_input(&formatCtx_);
        av_buffer_unref(&hwDeviceCtx_);
    }
    bool init(const std::string& path)
    {

        if (avformat_open_input(&formatCtx_, path.c_str(), NULL, NULL) < 0) {
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

        streamIndex_ = av_find_best_stream(formatCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder_, 0);
        if (streamIndex_ < 0) {
            return false;
        }

        std::string codeName = "cuda";

        AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
        if (type == AV_HWDEVICE_TYPE_NONE) {
            fprintf(stderr, "Device type %s is not supported.\n", codeName);
            fprintf(stderr, "Available device types:");
            while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
                fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
            fprintf(stderr, "\n");
            return -1;
        }
        for (int i = 0;; i++) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(decoder_, i);
            if (!config) {
                fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    decoder_->name, av_hwdevice_get_type_name(type));
                return -1;
            }
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == type) {
                hw_pix_fmt = config->pix_fmt;
                break;
            }
        }

        codecCtx_ = avcodec_alloc_context3(decoder_);
        if (!codecCtx_) {
            LOG(WARNING) << "can not alloc context!\n";
            return -1;
        }

        videoStream_ = formatCtx_->streams[streamIndex_];

        if (avcodec_parameters_to_context(codecCtx_,
                formatCtx_->streams[streamIndex_]->codecpar)
            < 0) {
            return -1;
        }

        codecCtx_->get_format = get_hw_format;

        int err = 0;
        if ((err = av_hwdevice_ctx_create(&hwDeviceCtx_, type,
                 NULL, NULL, 0))
            < 0) {
            fprintf(stderr, "Failed to create specified HW device.\n");
            return err;
        }
        codecCtx_->hw_device_ctx = av_buffer_ref(hwDeviceCtx_);

        if (avcodec_open2(codecCtx_, decoder_, NULL) < 0) {
            LOG(WARNING) << "can not open codec!\n";
            return -1;
        }
        packet_ = av_packet_alloc();
        if (!packet_) {
            LOG(WARNING) << "Failed to allocate AVPacket\n";
            return -1;
        }
        sw_frame_ = av_frame_alloc();
        frame_ = av_frame_alloc();
        send_thread_ = std::thread(&ffmpegDecoderWithGPU::send_packet_thread, this);
        receive_thread_ = std::thread(&ffmpegDecoderWithGPU::receive_frame_thread, this);
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

            const int ret = av_read_frame(formatCtx_, packet_);
            if (ret < 0) {
                LOG(WARNING) << "av_read_frame error! end of file! \n";
                isEnd_ = true;
                continue;
            }
            if (packet_->stream_index != streamIndex_) {
                continue;
            }
            std::unique_lock<std::mutex> lock(codecMutex_);
            while (true) {
                int ret = avcodec_send_packet(codecCtx_, packet_);
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
            av_packet_unref(packet_);
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
                } else if (pix_fmt == AV_PIX_FMT_UYVY422) {
                    LOG(INFO) << "The decoded frame is in YUV 4:2:2 (UYVY) format: " << pix_fmt;
                } else {
                    LOG(INFO) << "The decoded frame is in another pixel format: " << pix_fmt;
                }
                // retrieve data from GPU to CPU
if ((ret = av_hwframe_transfer_data(sw_frame_, frame_, 0)) < 0) {
    fprintf(stderr, "Error transferring the data to system memory\n");
    break;
}

const int width = sw_frame_->width;
const int height = sw_frame_->height;
cv::Mat yuvMat = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
memcpy(yuvMat.data, sw_frame_->data[0], width* height);
memcpy(yuvMat.data + width * height, sw_frame_->data[1],
    width* height / 4);
memcpy(yuvMat.data + width * height * 5 / 4, sw_frame_->data[2],
    width* height / 4);
// 调用回调函数，传递cv::Mat
if (onFrameCallback_) {
    onFrameCallback_(yuvMat);
}

// cv::Mat bgrMat;
//  cv::cvtColor(yuvMat, bgrMat, cv::COLOR_YUV2BGR_I420);
//  cv::imshow("win", bgrMat);
//  cv::waitKey(10);
LOG(INFO) << "codec a frame data, width: " << codecCtx_->width << " , height: " << codecCtx_->height;
codecCV_.notify_one(); // 通知发送线程可以继续
}
}
}

AVPixelFormat hw_pix_fmt_;
enum AVPixelFormat hwPixFmt_ = AV_PIX_FMT_NONE;
AVBufferRef* hwDeviceCtx_ = NULL;
AVFormatContext* formatCtx_ = NULL;
AVCodecContext* codecCtx_ = NULL;
const AVCodec* decoder_ = NULL;
AVPacket* packet_;

AVStream* videoStream_ = NULL;

int streamIndex_ = 0;
AVFrame* frame_;
AVFrame* sw_frame_;
std::atomic<int> isStart_ = 0;
std::atomic<bool> isEnd_ = false; // 新增的解码完成标志
std::thread send_thread_;
std::thread receive_thread_;

std::mutex codecMutex_;
std::condition_variable codecCV_;
std::function<void(const cv::Mat&)> onFrameCallback_; // 新增的回调函数成员
}
;
*/
#endif // FFMPGEDECODERWITHGPU_H
