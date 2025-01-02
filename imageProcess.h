#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

void interpolate(const cv::Mat& srcImg, cv::Mat& dstImg, const int dstHeight, const int dstWidth);

int example_interpolate();

#endif // IMAGEPROCESS_H
