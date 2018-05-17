#include "pyramid.hpp"

Pyramid::Pyramid(){
    this->n_levels = 0;
    this->gaussian_kernel_ver_ = (Mat_<float>(5,1) <<   1.0f, 4.0f, 6.0f, 4.0f, 1.0f);
    this->gaussian_kernel_hor_ = (Mat_<float>(1,5) <<   1.0f, 4.0f, 6.0f, 4.0f, 1.0f);
    this->gaussian_kernel_ver_ *= 1/256.0f;

    this->gaussian_kernel_ = (Mat_<float>(5,5) <<   1.0f, 4.0f, 6.0f, 4.0f, 1.0f,
                                                    4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
                                                    6.0f, 24.0f, 36.0f, 24.0f, 6.0f,
                                                    4.0f, 16.0f, 24.0f, 16.0f, 4.0f,
                                                    1.0f, 4.0f, 6.0f, 4.0f, 1.0f);
    this->gaussian_kernel_ *= 1/256.0f;
}

void Pyramid::subsampling(cv::Mat& upper_level, cv::Mat& new_level, int nrows, int ncols)
{
    int radius = this->gaussian_kernel_.size().height / 2;
    int kernel_size = this->gaussian_kernel_.size().height;
    int ii = 0, jj = 0, kk = 0, ll = 0;
    // Classic convolution
    for (int i = radius ; i < ncols - radius; i++)
    {
        for (int j = radius ; j < nrows - radius; j++)
        {
            float sum = 0.0f;
            for(int k = -radius ; k <= radius; k++)
            {
                for(int l = -radius ; l < radius ; l++)
                {
                    sum += upper_level.at<float>(Point(i*2 + k,j*2 + l)) * this->gaussian_kernel_.at<float>(Point(k + radius,l + radius));
                }
            }
            new_level.at<float>(Point(i,j)) = sum;
        }
    }
}


void Pyramid::separable_conv(cv::Mat &src,cv::Mat &dst)
{
    // Radius of the kernel
    int radius = this->gaussian_kernel_ver_.size().height / 2;

    // Width and height of the source image
    int width = src.size().width;
    int height = src.size().height;

    // Create the destination matrix at the correct size
    dst.create(height,width,src.type());

    // Buffer to stock the temp convoluted row
    AutoBuffer<float> _row(width + radius*2);
    AutoBuffer<float> _res(1);

    // Pointer of the buffer
    float *res=(float*)_res;
    float *row = (float*)_row + radius;

    // Loop on the rows
    for(int y = 0 ; y < height ; y++)
    {
        // Pointer to the source and destination: fast access to data
        float *srow0 = (float*)(src.data + src.step*y), *srow1=0;
        float *drow = (float*)(dst.data + dst.step*y);

        // Vertical convolution
        for(int x = 0; x < width; x++ )
        {
            // Pixel of interest
            row[x] = srow0[x] * this->gaussian_kernel_ver_.at<float>(Point(0,radius));
        }

        // Get the value of pixel above and below
        // Accessing -2 rows at a time (symetric kernel)
        for(int k = 1 ; k <= radius ; k++)
        {
            // Accessing row below and above
            srow0 = (float*)(src.data + src.step*std::max(y - k,0));
            srow1 = (float*)(src.data + src.step*std::min(y + k,height - 1));
            for(int x = 0 ; x < width ; x++)
            {        
                // Compute convolution (same kernel for both pixel)
                float p = srow0[x] + srow1[x];
                row[x] += this->gaussian_kernel_ver_.at<float>(Point(0,radius-k)) * p;
            }
        }

        // Horizontal convolution
        for(int x = 0 ; x < width ; x++)
        {
            // Compute kernel * pixel of interest
            res[0] = row[x] * this->gaussian_kernel_hor_.at<float>(Point(radius,0));
            // Accessing left and right pixel and compute conv (same value of weight for both pixel because symetric kernel)
            for(int k = 1 ; k <= radius ; k++)
            {
                int index1 = std::max(x - k,0);
                int index2 = std::min(x + k, width - 1);
                float p = (row[index1] + row[index2]) * this->gaussian_kernel_hor_.at<float>(Point(radius-k,0));
                res[0] = res[0] + p;
            }
            // Storing result of the convolution
            drow[x] = res[0];
        }

    }
}

void Pyramid::separable_conv_with_subsampling(cv::Mat &src, cv::Mat &dst, int n_rows, int n_cols)
{
    // Radius of the kernel
    int radius = this->gaussian_kernel_ver_.size().height / 2;

    // Width and height of the source image
    int width_src = src.size().width;
    int height_src = src.size().height;

    // Width and height of the destination image
    int width_dst = n_cols;
    int height_dst = n_rows;

    // Create the destination matrix at the correct size
    dst.create(n_rows,n_cols,src.type());

    // Buffer to stock the temp convoluted row
    AutoBuffer<float> _row(width_src + radius*2);
    AutoBuffer<float> _res(1);

    // Pointer of the buffer
    float *res=(float*)_res;
    float *row = (float*)_row + radius;

    // Loop on the rows
    for(int y = 0 ; y < height_dst ; y++)
    {
        // Pointer to the source and destination: fast access to data
        // Source: accessing one row of two (subsampling) but every col for the vertical convolution
        // because updated value are needed for the horizontal convolution
        float *srow0 = (float*)(src.data + src.step*y*2), *srow1=0;
        float *drow = (float*)(dst.data + dst.step*y);

        // Vertical convolution
        for(int x = 0; x < width_src; x++ )
        {
            // Pixel of interest
            row[x] = srow0[x] * this->gaussian_kernel_ver_.at<float>(Point(0,radius));
        }

        // Get the value of pixel above and below
        // Accessing -2 rows at a time (symetric kernel)
        for(int k = 1 ; k <= radius ; k++)
        {
            // Accessing row below and above (one row out of two for source: subsampling)
            srow0 = (float*)(src.data + src.step*std::max(y*2 - k,0));
            srow1 = (float*)(src.data + src.step*std::min(y*2 + k,height_src - 1));
            for(int x = 0 ; x < width_src ; x++)
            {   
                // Compute convolution (same kernel for both pixel)    
                float p = srow0[x] + srow1[x];
                row[x] += this->gaussian_kernel_ver_.at<float>(Point(0,radius-k)) * p;
            }
        }

        // Horizontal convolution
        for(int x = 0 ; x < width_dst ; x++)
        {
            // Compute kernel * pixel of interest
            // Only one cols out of two (subsampling)
            res[0] = row[x*2] * this->gaussian_kernel_hor_.at<float>(Point(radius,0));
            // Accessing left and right pixel and compute conv (same value of weight for both pixel because symetric kernel)
            for(int k = 1 ; k <= radius ; k++)
            {
                // Computing for only one cols out of two (subsampling)
                int index1 = std::max(x*2 - k,0);
                int index2 = std::min(x*2 + k, width_src - 1);
                float p = (row[index1] + row[index2]) * this->gaussian_kernel_hor_.at<float>(Point(radius-k,0));
                res[0] = res[0] + p;
            }
            // Storing result of the convolution
            drow[x] = res[0];
        }

    }
}



void Pyramid::create_pyramid(cv::Mat& img, int level, int win_size)
{
    if(img.size().height < win_size*2 + 1 || img.size().width < win_size*2 + 1)
    {
        std::cout<<"Problem with the image: image too small"<<std::endl;
        return;
    }
    cv::Mat temp_img;
    //temp_img = conv(img,this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);
    img.copyTo(temp_img);

    this->img_levels.push_back(temp_img);
    int n_rows = 0, n_colums = 0;
    this->n_levels = 1;

    for(int l = 0; l < level; l++)
    {
        //std::cout<<"create pyr"<<std::endl;
        if(this->img_levels.at(l).size().height < win_size*2 + 1 || this->img_levels.at(l).size().width < win_size*2 + 1)
        {
            std::cout<<"Img of Pyramid too small, stopped at level: "<<l<<std::endl;
            this->n_levels = l + 1;
            break;
        }
        else if(this->img_levels.at(l).size().height % 2 || this->img_levels.at(l).size().width % 2)
        {
            n_rows = floor(this->img_levels.at(l).rows / 2);
            n_colums = floor(this->img_levels.at(l).cols / 2);
        }
        else
        {
            n_rows = this->img_levels.at(l).rows / 2;
            n_colums = this->img_levels.at(l).cols / 2;
        }
        //std::cout<<"rows: "<<n_rows<<" , colums: "<<n_colums<<std::endl;
        

        cv::Mat new_level = cv::Mat::zeros(n_rows,n_colums,this->img_levels.at(l).type());
        //std::cout<<new_level.size<<std::endl;

        // OLD VERSION
        //temp_img = conv(this->img_levels.at(i),this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);
        //temp_img = cv::Mat::zeros(this->img_levels.at(i).rows,this->img_levels.at(i).cols,this->img_levels.at(i).type());
        // separable_conv(this->img_levels.at(l),temp_img);

        // // Subsample
        // for (int i = 0 ; i < n_colums ; i++)
        // {
        //     for (int j = 0 ; j < n_rows ; j++)
        //     {
        //         new_level.at<float>(Point(i,j)) = temp_img.at<float>(Point(i*2,j*2));
        //     }
        // }

        // FAST SEPARABLE CONV + SUBSAMPLING
        separable_conv_with_subsampling(this->img_levels.at(l),new_level,n_rows,n_colums);

        // NEW_VERSION
        //subsampling(this->img_levels.at(l),new_level,n_rows,n_colums);

        this->img_levels.push_back(new_level);
        this->n_levels++;
        //temp_img.deallocate();
    }
}