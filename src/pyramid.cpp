#include "pyramid.hpp"

Pyramid::Pyramid(){
    this->n_levels = 0;
    this->gaussian_kernel_ver_ = (Mat_<float>(5,1) <<   1.0f, 4.0f, 6.0f, 4.0f, 1.0f);
    this->gaussian_kernel_hor_ = (Mat_<float>(1,5) <<   1.0f, 4.0f, 6.0f, 4.0f, 1.0f);
    this->gaussian_kernel_hor_ *= 1/256.0f;
    this->kernel_size = 5.0f;
}

void Pyramid::create_pyramid(cv::Mat& img, int level, int win_size)
{
    if(img.size().height < win_size*2 + 1 || img.size().width < win_size*2 + 1)
    {
        std::cout<<"Problem with the image: image too small"<<std::endl;
        return;
    }
    cv::Mat temp_img;
    temp_img = conv(img,this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);

    this->img_levels.push_back(temp_img);
    int n_rows = 0, n_colums = 0;
    this->n_levels = 1;

    for(int i = 0; i < level; i++)
    {
        //std::cout<<"create pyr"<<std::endl;
        if(this->img_levels.at(i).size().height < win_size*2 + 1 || this->img_levels.at(i).size().width < win_size*2 + 1)
        {
            std::cout<<"Img of Pyramid too small, stopped at level: "<<i<<std::endl;
            this->n_levels = i + 1;
            break;
        }
        else if(this->img_levels.at(i).size().height % 2 || this->img_levels.at(i).size().width % 2)
        {
            n_rows = floor(this->img_levels.at(i).rows / 2);
            n_colums = floor(this->img_levels.at(i).cols / 2);
        }
        else
        {
            n_rows = this->img_levels.at(i).rows / 2;
            n_colums = this->img_levels.at(i).cols / 2;
        }
        //std::cout<<"rows: "<<n_rows<<" , colums: "<<n_colums<<std::endl;
        

        cv::Mat new_level = cv::Mat::zeros(n_rows,n_colums,this->img_levels.at(i).type());
        //std::cout<<new_level.size<<std::endl;

        temp_img = conv(this->img_levels.at(i),this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);

        // Subsample
        for (int i = 0 ; i < n_colums ; i++)
        {
            for (int j = 0 ; j < n_rows ; j++)
            {
                new_level.at<float>(Point(i,j)) = temp_img.at<float>(Point(i*2,j*2));
            }
        }

        this->img_levels.push_back(new_level);
        this->n_levels++;
        //temp_img.deallocate();
    }
}

void Pyramid::create_pyramid(cv::Mat& img, int level, int win_size, cv::Point2f feature)
{
    if(img.size().height < win_size*2 + 1 || img.size().width < win_size*2 + 1)
    {
        std::cout<<"Problem with the image: image too small"<<std::endl;
        return;
    }
            
    // Define the area corresponding to the window
    int index_col_start = int(std::floor(feature.x)) - 3 - win_size * pow(2,level);
    if(index_col_start < 0)
        index_col_start = 0;
    int index_row_start = int(std::floor(feature.y)) - 3 - win_size * pow(2,level);
    if(index_row_start < 0)
        index_row_start = 0;
    int index_col_end = int(std::floor(feature.x)) + 3 + win_size * pow(2,level);
    if(index_col_end >= img.cols)
        index_col_end = img.cols - 1;
    int index_row_end = int(std::floor(feature.y)) + 3 + win_size * pow(2,level);
    if(index_row_end >= img.rows)
        index_row_end = img.rows - 1;



    cv::Mat temp_img;
    temp_img = conv(img,index_col_start,index_col_end,index_row_start,index_row_end,this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);

    this->img_levels.push_back(temp_img);
    int n_rows = 0, n_colums = 0;
    this->n_levels = 1;

    for(int k = 0; k < level; k++)
    {
        //std::cout<<"create pyr"<<std::endl;
        if(this->img_levels.at(k).size().height < win_size*2 + 1 || this->img_levels.at(k).size().width < win_size*2 + 1)
        {
            std::cout<<"Img of Pyramid too small, stopped at level: "<<k<<std::endl;
            break;
        }
        else if(this->img_levels.at(k).size().height % 2 || this->img_levels.at(k).size().width % 2)
        {
            n_rows = floor(this->img_levels.at(k).rows / 2);
            n_colums = floor(this->img_levels.at(k).cols / 2);
        }
        else
        {
            n_rows = this->img_levels.at(k).rows / 2;
            n_colums = this->img_levels.at(k).cols / 2;
        }
        //std::cout<<"rows: "<<n_rows<<" , colums: "<<n_colums<<std::endl;

        cv::Mat new_level = cv::Mat::zeros(n_rows,n_colums,this->img_levels.at(k).type());
        //std::cout<<new_level.size<<std::endl;

        temp_img = conv(this->img_levels.at(0),index_col_start,index_col_end,index_row_start,index_row_end,this->gaussian_kernel_hor_,this->gaussian_kernel_ver_);

        // Subsample
        for (int i = 0 ; i < n_colums ; i++)
        {
            for (int j = 0 ; j < n_rows ; j++)
            {
                new_level.at<float>(Point(i,j)) = temp_img.at<float>(Point(i*pow(2,k+1),j*pow(2,k+1)));
            }
        }

        this->img_levels.push_back(new_level);
        this->n_levels++;
        //temp_img.deallocate();
    }
}