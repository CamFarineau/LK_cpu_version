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
    int n_levels;
    vector<cv::Mat> img_levels;

    private:

    void subsampling(cv::Mat& upper_level, cv::Mat& new_level, int nrows, int ncols);
    void separable_conv(cv::Mat &src,cv::Mat &dst);
    void separable_conv_with_subsampling(cv::Mat &src, cv::Mat &dst, int n_rows, int n_cols);
    cv::Mat gaussian_kernel_hor_;
    cv::Mat gaussian_kernel_ver_;
    cv::Mat gaussian_kernel_;
 
};

#endif