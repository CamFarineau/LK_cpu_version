#include "conv.hpp"

cv::Mat conv_hor(cv::Mat& img, cv::Mat& kernel_hor)
{
    cv::Mat result = cv::Mat::zeros(img.rows,img.cols,img.type());
    int radius = kernel_hor.size().width / 2;
    int ncols = img.cols, nrows = img.rows;
    int i,j,k;
    float ppp, sum;
    for (j = 0; j < nrows; j++ )
    {
        for (i = 0; i < radius; i++ )
        {
            result.at<float>(Point(i,j)) = 0.0f;
        }
        for ( ; i < ncols - radius ; i++){
            ppp = img.at<float>(Point(i - radius,j));
            sum = 0.0f;
            int cpt = 1;
            for (k = kernel_hor.size().width-1 ; k >= 0 ; k--)
            {
                sum += ppp * kernel_hor.at<float>(Point(k,0));
                ppp = img.at<float>(Point(i - radius + cpt,j));
                cpt++;
            }
                
            result.at<float>(Point(i,j)) = sum;
            int test = 0;
        }
    }

    return result;
}

cv::Mat conv_ver(cv::Mat& img, cv::Mat& kernel_ver)
{
    int test = img.type();
    cv::Mat result = cv::Mat::zeros(img.rows,img.cols,img.type());
    int radius = kernel_ver.size().height / 2;
    int ncols = img.cols, nrows = img.rows;
    int i,j,k;
    float ppp, sum;
    for (i = 0; i < ncols; i++ )
    {
        for (j = 0; j < radius; j++ )
        {
            result.at<float>(Point(i,j)) = 0.0f;
        }
        for ( ; j < nrows- radius ; j++){
            ppp = img.at<float>(Point(i,j - radius));
            sum = 0.0f;
            int cpt = 1;
            for (k = kernel_ver.size().height-1 ; k >= 0 ; k--)
            {
                sum += ppp * kernel_ver.at<float>(Point(0,k));
                ppp = img.at<float>(Point(i,j - radius + cpt));
                cpt++;
            }
            result.at<float>(Point(i,j)) = sum;
        }
    }

    return result;
}

cv::Mat conv(cv::Mat& img, cv::Mat& kernel_hor, cv::Mat& kernel_ver)
{
    cv::Mat result = conv_hor(img,kernel_hor);
    result = conv_ver(result, kernel_ver);
    return result;
}