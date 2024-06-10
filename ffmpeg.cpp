#include "ffmpeg.h"

#include <assert.h>

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILENAME "rtsp://127.0.0.1:8554/123"

int demo_main(int argc, char** argv)
{
    // av_register_all();

    AVFormatContext* formatCtx = NULL;
    // const char* fileName = "/home/lotuscc/Desktop/out.mov";
    const char* fileName = "rtsp://127.0.0.1:8554/123";

    // avformat_close_input()

    if (avformat_open_input(&formatCtx, fileName, NULL, NULL) < 0) {
        printf("can not open file! \n");
        return -1;
    }
    if (avformat_find_stream_info(formatCtx, NULL) < 0) {
        printf("can not get stream infos! \n");
        return -1;
    }
    for (int i = 0; i < formatCtx->nb_streams; ++i) {
        AVStream* stream = formatCtx->streams[i];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
        } else if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        } else {
        }
        printf("duration: %lld ms \n", stream->duration * 1000 / AV_TIME_BASE);
    }
    AVCodecContext* codecCtx = NULL;
    int streamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (streamIndex >= 0) {
        AVCodecParameters* codecParam = formatCtx->streams[streamIndex]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParam->codec_id);
        if (!codec) {
            printf("can not find codec!\n");
            return -1;
        }
        codecCtx = avcodec_alloc_context3(codec);
        if (!codecCtx) {
            printf("can not alloc context!\n");
            return -1;
        }
        avcodec_parameters_to_context(codecCtx,
            formatCtx->streams[streamIndex]->codecpar);
        if (avcodec_open2(codecCtx, codec, NULL) < 0) {
            printf("can not open codec!\n");
            return -1;
        }
    }
    AVPacket packet;
    av_init_packet(&packet);

    AVFrame* frame = av_frame_alloc();

    while (av_read_frame(formatCtx, &packet) >= 0) {
        if (packet.stream_index == streamIndex) {
            int ret = avcodec_send_packet(codecCtx, &packet);
            if (ret < 0) {
                printf("avcodec_send_packet error \n");
                break;
            }
            while (ret >= 0) {
                ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    printf("avcodec_receive_frame error \n");
                    break;
                }
                AVPixelFormat pix_fmt = (AVPixelFormat)frame->format;
                // 检查像素格式
                if (pix_fmt == AV_PIX_FMT_YUV420P) {
                    printf("The decoded frame is in YUV 4:2:0 format.\n");
                } else if (pix_fmt == AV_PIX_FMT_UYVY422) {
                    printf("The decoded frame is in YUV 4:2:2 (UYVY) format.\n");
                } else {
                    printf("The decoded frame is in another pixel format: %d.\n", pix_fmt);
                }
                const int width = frame->width;
                const int height = frame->height;
                cv::Mat yuvMat = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
                memcpy(yuvMat.data, frame->data[0], width * height);
                memcpy(yuvMat.data + width * height, frame->data[1],
                    width * height / 4);
                memcpy(yuvMat.data + width * height * 5 / 4, frame->data[2],
                    width * height / 4);
                cv::Mat bgrMat;
                // cv::cvtColor(yuvMat, bgrMat, cv::COLOR_YUV2RGB_I420);
                cv::cvtColor(yuvMat, bgrMat, cv::COLOR_YUV2BGR_I420);
                cv::imshow("win", bgrMat);
                cv::waitKey(10);
                printf("codec a frame data, width: %d, height: %d \n", codecCtx->width,
                    codecCtx->height);
            }
        }
        av_packet_unref(&packet);
    }
    av_frame_free(&frame);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);

    return 0;
}

int demo_GPU_main()
{
    AVFormatContext* formatCtx = NULL;
    AVCodecContext* codecCtx = NULL;
    const AVCodec* codec = NULL;
    AVPacket packet;
    AVFrame* frame = NULL;
    AVBufferRef* device_ref = NULL;
    int streamIndex;
    int ret;

    if (avformat_open_input(&formatCtx, FILENAME, NULL, NULL) < 0) {
        printf("Can't open file.\n");
        return -1;
    }
    if (avformat_find_stream_info(formatCtx, NULL) < 0) {
        printf("Can't find stream information.\n");
        return -1;
    }

    // Find the best video stream.
    streamIndex = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (streamIndex < 0) {
        printf("No suitable video stream found.\n");
        return -1;
    }

    // Get codec parameters from stream.
    AVCodecParameters* codecParam = formatCtx->streams[streamIndex]->codecpar;
    codec = avcodec_find_decoder_by_name("h264_cuvid");
    if (!codec) {
        printf("Failed to find hardware accelerated decoder.\n");
        return -1;
    }

    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        printf("Failed to allocate the codec context.\n");
        return -1;
    }

    // Set up hardware device context.
    av_hwdevice_ctx_create(&device_ref, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);
    if (!device_ref) {
        printf("Failed to create CUDA device context.\n");
        return -1;
    }

    // codecCtx->get_format = avcodec_hw_get_format;
    codecCtx->get_format = avcodec_default_get_format;
    codecCtx->hw_device_ctx = av_buffer_ref(device_ref);

    // Copy codec parameters to codec context and open it.
    ret = avcodec_parameters_to_context(codecCtx, codecParam);
    if (ret < 0) {
        printf("Failed to copy codec parameters to codec context.\n");
        return -1;
    }

    ret = avcodec_open2(codecCtx, codec, NULL);
    if (ret < 0) {
        printf("Failed to open hardware accelerated decoder.\n");
        return -1;
    }

    // Allocate frame.
    frame = av_frame_alloc();
    if (!frame) {
        printf("Failed to allocate AVFrame.\n");
        return -1;
    }

    // Decode video frames.
    while (av_read_frame(formatCtx, &packet) >= 0) {
        if (packet.stream_index == streamIndex) {
            ret = avcodec_send_packet(codecCtx, &packet);
            if (ret < 0) {
                printf("Error sending a packet for decoding.\n");
                break;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(codecCtx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    printf("Error during decoding.\n");
                    break;
                }

                // Convert GPU frame to CPU frame.
                AVFrame* cpu_frame = av_frame_alloc();

                // av_image_fill_arrays(cpu_frame->data, cpu_frame->linesize, frame->data[0], codecCtx->pix_fmt, codecCtx->width, codecCtx->height, 1);

                av_hwframe_transfer_data(cpu_frame, frame, 0);

                av_frame_copy_props(cpu_frame, frame);

                // av_image_copy_to_buffer

                cpu_frame->format = codecCtx->pix_fmt;
                cpu_frame->width = codecCtx->width;
                cpu_frame->height = codecCtx->height;

                // Display or process the frame here.
                cv::Mat yuvMat = cv::Mat::zeros(cpu_frame->height * 3 / 2, cpu_frame->width, CV_8UC1);
                memcpy(yuvMat.data, cpu_frame->data[0], cpu_frame->width * cpu_frame->height);
                memcpy(yuvMat.data + cpu_frame->width * cpu_frame->height, cpu_frame->data[1], cpu_frame->width * cpu_frame->height / 4);
                memcpy(yuvMat.data + cpu_frame->width * cpu_frame->height * 5 / 4, cpu_frame->data[2], cpu_frame->width * cpu_frame->height / 4);
                cv::Mat bgrMat;
                cv::cvtColor(yuvMat, bgrMat, cv::COLOR_YUV2BGR_I420);
                cv::imshow("win", bgrMat);
                cv::waitKey(10);

                av_frame_free(&cpu_frame);
            }
        }
        av_packet_unref(&packet);
    }

    // Clean up resources.
    av_frame_free(&frame);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&formatCtx);
    av_buffer_unref(&device_ref);

    return 0;
}
