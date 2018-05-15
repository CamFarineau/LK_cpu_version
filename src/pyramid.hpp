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
    void create_pyramid_gradx(Pyramid& img_pyr, int level);
    void create_pyramid_grady(Pyramid& img_pyr, int level);    
    int n_levels;
    vector<cv::Mat> img_levels;

    private:
    cv::Mat gaussian_kernel_hor_;
    cv::Mat gaussian_kernel_ver_;
    cv::Mat gaussian_kernel_gradx_hor_;
    cv::Mat gaussian_kernel_gradx_ver_;
    cv::Mat gaussian_kernel_grady_hor_;
    cv::Mat gaussian_kernel_grady_ver_;
 
};

#endif