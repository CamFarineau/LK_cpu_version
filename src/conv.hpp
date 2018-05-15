#ifndef CONV_H
#define CONV_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

cv::Mat conv_ver(cv::Mat& img, cv::Mat& kernel_ver);
cv::Mat conv_hor(cv::Mat& img, cv::Mat& kernel_hor);
cv::Mat conv(cv::Mat& img, cv::Mat& kernel_hor, cv::Mat& kernel_ver);


#endif