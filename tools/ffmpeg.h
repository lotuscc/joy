#ifndef FFMPEG_H
#define FFMPEG_H
#include <cstdio>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int demo_main(int argc, char** argv);
int demo_main_new(int argc, char** argv);
int demo_GPU_main();

#endif // FFMPEG_H
