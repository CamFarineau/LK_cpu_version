#ifndef PYRAMID_H
#define PYRAMID_H

#include <opencv2/opencv.hpp>
#include "conv.hpp"

using namespace std;
using namespace cv;

class Pyramid
{
    public:
    Pyramid();
    void create_pyramid(cv::Mat& img, int level, int win_size);
    void create_pyramid(cv::Mat& img, int level, int win_size, cv::Point2f feature);
    int n_levels;
    vector<cv::Mat> img_levels;

    private:
    cv::Mat gaussian_kernel_hor_;
    cv::Mat gaussian_kernel_ver_;
    float kernel_size;
};

#endif