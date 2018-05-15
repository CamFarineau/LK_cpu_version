#include "opt_flow_lk.hpp"

OptFlowLK::OptFlowLK(){}
float get_value (cv::Mat const &image, cv::Point2f index)
{
    if (index.x >= image.cols)
        index.x = image.cols - 1.0f;
    else if (index.x < 0.0f)
        index.x = 0.0f;

    if (index.y >= image.rows)
        index.y = image.rows - 1.0f;
    else if (index.y < 0.0f)
        index.y = 0.0f;
    
    float test = image.at<float>(Point(index.x, index.y));
    // std::cout<<"val en "<<index<<" : "<<test<<std::endl;
    return test;
}
float get_subpixel_value (cv::Mat const &image, cv::Point2f index)
{
    float floor_col = (float) floor (index.x);
    float floor_row = (float) floor (index.y);

    float fract_col = index.x - floor_col;
    float fract_row = index.y - floor_row;

    return        ((1.0f - fract_col) * (1.0f - fract_row) * get_value(image, cv::Point2f(floor_col, floor_row))
                + (fract_col * (1.0f - fract_row) * get_value(image, cv::Point2f(floor_col + 1.0f, floor_row)))
                + ((1.0f - fract_col) * fract_row * get_value(image, cv::Point2f(floor_col, floor_row + 1.0f)))
                + (fract_col * fract_row * get_value(image, cv::Point2f(floor_col + 1.0f, floor_row + 1.0f))));
}

void OptFlowLK::compute_lk(Mat& frame1, Mat& frame2, vector<Point2f>& features, vector<Point2f>& new_features, int win_size, int level, float min_eigen_threshold, int max_iterations, float eps_criteria)
{
    new_features.erase(new_features.begin(),new_features.end());
    new_features.clear();

    if(features.empty())
    {
        std::cout<<"No feature to track"<<std::endl;
        return;
    }

    //std::cout<<features.size()<<" features to track"<<std::endl;
    vector<Point2f> features_copy(features);

    int nb_feature_erased = 0;

    this->frame1_pyr_.img_levels.erase(this->frame1_pyr_.img_levels.begin(),this->frame1_pyr_.img_levels.end());
    this->frame1_pyr_.img_levels.clear();

    this->frame2_pyr_.img_levels.erase(this->frame2_pyr_.img_levels.begin(),this->frame2_pyr_.img_levels.end());
    this->frame2_pyr_.img_levels.clear();

    TicToc time_exec;
    time_exec.tic();
    this->frame1_pyr_.create_pyramid(frame1,level,win_size);
    this->frame2_pyr_.create_pyramid(frame2,level,win_size);
    std::cout<<"Time create pyr: ";
    time_exec.toc();


    std::cout<<"levels: "<<this->frame1_pyr_.img_levels.size()<<std::endl;

    // namedWindow( "pyramid", 1 );

    // for(int i = 0; i < this->frame1_pyr_.n_levels ; i++)
    // {
    //     imshow("pyramid", this->frame1_pyr_.img_levels.at(i));
    //     waitKey(0);
    //     imshow("pyramid", this->frame2_pyr_.img_levels.at(i));
    //     waitKey(0);
    // }

    //std::cout<<this->frame1_pyr_.n_levels<<std::endl;

    time_exec.tic();

    // For all features
    for(int f = 0; f<features_copy.size(); f++)
    {
        if(features_copy.at(f).x < 0.0f || features_copy.at(f).x > frame2.cols || features_copy.at(f).y < 0.0f || features_copy.at(f).y >= frame2.rows)
        {
            // Don't track feature that are outside the image
            //std::cout<<"Features out"<<std::endl;
            features_copy.erase(features_copy.begin() + f);
            continue;
        }
        //std::cout<<"feature "<<f<<" : "<<features_copy.at(f)<<std::endl;
        
        cv::Mat d_position_final = cv::Mat::zeros(2,1,frame1.type());
        cv::Mat pyramid_position = cv::Mat::zeros(2,1,frame1.type());

        Status status = Status::Tracked;

        for(int temp_level = this->frame1_pyr_.img_levels.size() - 1; temp_level >= 0; temp_level-- )
        {
            Point2f current_point = Point2f(features_copy.at(f).x / pow(2,temp_level),features_copy.at(f).y / pow(2,temp_level));
            
            //std::cout<<"Feature: "<<current_point<<std::endl;
            // Define the area corresponding to the window
            float index_col_start = current_point.x - (float)win_size;
            if(index_col_start < 0.0f)
                index_col_start = 0.0f;
            float index_row_start = current_point.y - (float)win_size;
            if(index_row_start < 0.0f)
                index_row_start = 0.0f;
            float index_col_end = current_point.x + (float)win_size;
            if(index_col_end > (float)this->frame1_pyr_.img_levels.at(temp_level).cols)
                index_col_end = (float)this->frame1_pyr_.img_levels.at(temp_level).cols;
            float index_row_end = current_point.y + (float)win_size;
            if(index_row_end > (float)this->frame1_pyr_.img_levels.at(temp_level).rows)
                index_row_end = (float)this->frame1_pyr_.img_levels.at(temp_level).rows;

            //std::cout<<"indices: "<<index_col_start<<", "<<index_col_end<<" // "<<index_row_start<<", "<<index_row_end<<std::endl;
            
            cv::Mat gradient_mat = cv::Mat::zeros(2,2,frame1.type());

            std::vector<Point2f> derivatives;

            float g1 = 0.0f, g2 = 0.0f;

            for (float i = index_col_start; i <= index_col_end; i += 1.0f)
            {
                for (float j = index_row_start; j <= index_row_end; j += 1.0f)
                { 
                    //std::cout<<"pixel en "<<i<<", "<<j<<" : "<<this->frame1_pyr_.img_levels.at(0).at<float>(Point2f(i,j))<<std::endl;
                    
                    g1 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i + 1.0f,j));
                    g2 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i - 1.0f,j));

                    float gradx = (g1 - g2) / 2.0f;

                    g1 = 0.0f;
                    g2 = 0.0f;

                    g1 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j + 1.0f));
                    g2 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j - 1.0f));

                    float grady = (g1 - g2) / 2.0f;

                    Point2f test = Point2f(gradx, grady);
                    //std::cout<<"derivative: "<<test<<std::endl;
                    derivatives.push_back(test);

                    gradient_mat.at<float>(Point(0, 0)) += gradx * gradx;
                    gradient_mat.at<float>(Point(0, 1)) += gradx * grady;
                    gradient_mat.at<float>(Point(1, 0)) += gradx * grady;
                    gradient_mat.at<float>(Point(1, 1)) += grady * grady;
                }
            }
            float det = gradient_mat.at<float>(Point(0, 0))*gradient_mat.at<float>(Point(1, 1)) - gradient_mat.at<float>(Point(0, 1))*gradient_mat.at<float>(Point(1, 0));

            if(det < min_eigen_threshold)
            {
                status = Status::SmallDet;
                break;
            }
            //std::cout<<"grad mat: "<<gradient_mat<<std::endl<<std::endl;
            gradient_mat = gradient_mat.inv();
            //std::cout<<"grad mat inv: "<<gradient_mat<<std::endl<<std::endl;

            cv::Mat d_position = cv::Mat::zeros(2,1,frame1.type());

            int iteration = 0;
            float x_abs,y_abs, norm_temp;

            do{
                cv::Mat image_mismatch = cv::Mat::zeros(2,1,frame1.type());

                int cpt = 0;
                x_abs = 0.0f;
                y_abs = 0.0f;

                for (float i = index_col_start; i <= index_col_end; i += 1.0f)
                {
                    for (float j = index_row_start; j <= index_row_end; j += 1.0f)
                    {
                        float next_index_i = i + pyramid_position.at<float>(0,0) + d_position.at<float>(0, 0);
                        float next_index_j = j + pyramid_position.at<float>(0,1) + d_position.at<float>(0, 1);

                        float img_difference = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j)) - get_subpixel_value(this->frame2_pyr_.img_levels.at(temp_level),Point2f(next_index_i,next_index_j));
                        image_mismatch.at<float>(Point(0,0)) += img_difference * derivatives.at(cpt).x;
                        image_mismatch.at<float>(Point(0,1)) += img_difference * derivatives.at(cpt).y;

                        cpt++;
                    }
                }
                //std::cout<<"image mismatch"<<image_mismatch<<std::endl;
                cv::Mat temp = cv::Mat::zeros(2,1,frame1.type());
                temp = gradient_mat * image_mismatch;
                //std::cout<<"temp float pos: "<<temp<<std::endl;

                x_abs = std::abs(temp.at<float>(Point(0,0)));
                y_abs = std::abs(temp.at<float>(Point(0,1)));
                //std::cout<<"norm: "<<norm_temp<<std::endl;

                d_position.at<float>(Point(0,0)) += temp.at<float>(Point(0,0));
                d_position.at<float>(Point(0,1)) += temp.at<float>(Point(0,1));

                //std::cout<<"d_position: "<<d_position<<std::endl;
                iteration++;
                //std::cout<<"iteration: "<<iteration<<std::endl;

                image_mismatch.release();
                temp.release();

            }while((x_abs >= eps_criteria || y_abs >= eps_criteria) && iteration < max_iterations);
            //std::cout<<"iteration: "<<iteration<<std::endl;
            if(iteration >= max_iterations)
            {
                status = Status::NotConverge;
                break;
            }
            
            // // Check for Large Residue
            // float sum_window = 0.0f;
            // for (float i = index_col_start; i <= index_col_end; i += 1.0f)
            // {
            //     for (float j = index_row_start; j <= index_row_end; j += 1.0f)
            //     {
            //         float next_index_i = i + pyramid_position.at<float>(0,0) + d_position.at<float>(0, 0);
            //         float next_index_j = j + pyramid_position.at<float>(0,1) + d_position.at<float>(0, 1);

            //         sum_window += std::abs(get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j)) - get_subpixel_value(this->frame2_pyr_.img_levels.at(temp_level),Point2f(next_index_i,next_index_j)));
            //     }
            // }
            // std::cout<<"Residue: "<<sum_window / (float)((win_size*2 + 1)*(win_size*2 + 1))<<std::endl;
            // if(sum_window / (float)((win_size*2 + 1)*(win_size*2 + 1)) > 10.0f)
            // {
            //     status = Status::LargeResidue;
            //     break;
            // }

            if(temp_level == 0)
            {
                d_position_final = d_position;
            }
            else
            {
                pyramid_position = 2 * (pyramid_position + d_position);
            }


            gradient_mat.release();
            d_position.release();
        }

        d_position_final += pyramid_position;
        //std::cout<<"d_position finale: "<<d_position_final<<std::endl;
        //std::cout<<"Pos init: "<<features_copy.at(f)<<std::endl;
        //std::cout<<"Status: "<<status<<std::endl;

        Point2f new_pos(features_copy.at(f).x + d_position_final.at<float>(Point(0, 0)),features_copy.at(f).y + d_position_final.at<float>(Point(0, 1)));
        //std::cout<<"Pos finale: "<<new_pos<<std::endl;
        if(status != Status::Tracked || new_pos.x - (float)win_size < 0.0f || new_pos.x + (float)win_size > frame2.cols || new_pos.y - (float)win_size < 0.0f || new_pos.y + (float)win_size >= frame2.rows)
        {
            //std::cout<<"feature "<<f<<" erased"<<std::endl;
            features.erase(features.begin() + f - nb_feature_erased);
            nb_feature_erased++;
            continue;
        }
        else
        {
            new_features.push_back(new_pos); 
            //std::cout<<"add feature "<<f<<std::endl; 
        }
    }
    std::cout<<"Time track features: ";
    time_exec.toc();
}