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
        for (i = radius ; i < ncols - radius ; i++){
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
        }
    }

    return result;
}

cv::Mat conv_ver(cv::Mat& img, cv::Mat& kernel_ver)
{
    cv::Mat result = cv::Mat::zeros(img.rows,img.cols,img.type());
    int radius = kernel_ver.size().height / 2;
    int ncols = img.cols, nrows = img.rows;
    int i,j,k;
    float ppp, sum;
    for (i = 0; i < ncols; i++ )
    {
        for (j = radius ; j < nrows - radius ; j++){
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
    namedWindow( "pyramid", 1 );
    imshow("pyramid", result);
    waitKey(0);
    result = conv_ver(result, kernel_ver);
    namedWindow( "pyramid", 1 );
    imshow("pyramid", result);
    waitKey(0);
    return result;
}


// Overloading with convolution only in window
cv::Mat conv_hor(cv::Mat& img, int index_col_start, int index_col_end, int index_row_start, int index_row_end, cv::Mat& kernel_hor)
{
    cv::Mat result;
    img.copyTo(result);
    int radius = kernel_hor.size().width / 2;
    int ncols = img.cols, nrows = img.rows;
    int i,j,k;
    float ppp, sum;

    index_row_start -= radius;
    if(index_row_start < 0)
        index_row_start = 0;
    
    index_row_end += radius;
    if(index_row_end > nrows - 1)
        index_row_end = nrows - 1;

    for (j = index_row_start; j <= index_row_end; j++ )
    {
        for (i = index_col_start; i <= index_col_end ; i++){
            
            if(i - radius < 0 || i + radius > ncols - 1)
            {
                result.at<float>(Point(i,j)) = 0.0f;
            }
            else
            {
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
            }

        }
    }

    return result;
}

cv::Mat conv_ver(cv::Mat& img, int index_col_start, int index_col_end, int index_row_start, int index_row_end, cv::Mat& kernel_ver)
{
    cv::Mat result;
    img.copyTo(result);
    int radius = kernel_ver.size().height / 2;
    int ncols = img.cols, nrows = img.rows;
    int i,j,k;
    float ppp, sum;

    // index_col_start -= radius;
    // if(index_col_start < 0)
    //     index_col_start = 0;
    
    // index_col_end += radius;
    // if(index_col_end > ncols - 1)
    //     index_col_end = ncols - 1;


    for (i = index_col_start; i <= index_col_end; i++ )
    {
        for (j = index_row_start ; j <= index_row_end ; j++){

            if(j - radius < 0 || j + radius > nrows - 1)
            {
                result.at<float>(Point(i,j)) = 0.0f;
            }
            else
            {
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
    }

    return result;
}

cv::Mat conv(cv::Mat& img, int index_col_start, int index_col_end, int index_row_start, int index_row_end, cv::Mat& kernel_hor, cv::Mat& kernel_ver)
{
    cv::Mat result = conv_hor(img, index_col_start, index_col_end, index_row_start, index_row_end, kernel_hor);
    result = conv_ver(result, index_col_start, index_col_end, index_row_start, index_row_end, kernel_ver);
    return result;
}