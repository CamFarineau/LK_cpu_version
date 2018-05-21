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
    // NEW VERSION WIP
    int floor_col = static_cast<int>(std::floor(index.x));
    int floor_col1 = floor_col + 1;
    int floor_row = static_cast<int>(std::floor(index.y));

    float fract_col = index.x - std::floor(index.x);
    float fract_row = index.y - std::floor(index.y);

    if (floor_col >= image.cols - 1)
    {
        floor_col = image.cols - 1;
        floor_col1 = floor_col;
    }
    else if (floor_col < 0)
    {
        floor_col = 0;
        floor_col1 = 1;
    }

    if (floor_row >= image.rows)
        floor_row = image.rows - 1;
    else if (floor_row < 0)
        floor_row = 0;

    float *srow0 = (float*)(image.data + image.step*floor_row);
    float *srow1 = (float*)(image.data + image.step*std::min(floor_row + 1,image.rows - 1));

    return        ((1.0f - fract_col) * (1.0f - fract_row) * srow0[floor_col])
                + (fract_col * (1.0f - fract_row) * srow0[floor_col1])
                + ((1.0f - fract_col) * fract_row * srow1[floor_col])
                + (fract_col * fract_row * srow1[floor_col1]);

    ///////////////////////////
    
    
    // float floor_col = (float) std::floor(index.x);
    // float floor_row = (float) std::floor(index.y);

    // float fract_col = index.x - floor_col;
    // float fract_row = index.y - floor_row;

    // if (floor_col >= (float)image.cols)
    //     floor_col = (float)image.cols - 1.0f;
    // else if (floor_col < 0.0f)
    //     floor_col = 0.0f;

    // if (floor_row >= (float)image.rows)
    //     floor_row = (float)image.rows - 1.0f;
    // else if (floor_row < 0.0f)
    //     floor_row = 0.0f;

    // return        ((1.0f - fract_col) * (1.0f - fract_row) * get_value(image, cv::Point2f(floor_col, floor_row))
    //             + (fract_col * (1.0f - fract_row) * get_value(image, cv::Point2f(floor_col + 1.0f, floor_row)))
    //             + ((1.0f - fract_col) * fract_row * get_value(image, cv::Point2f(floor_col, floor_row + 1.0f)))
    //             + (fract_col * fract_row * get_value(image, cv::Point2f(floor_col + 1.0f, floor_row + 1.0f))));
}

void OptFlowLK::compute_lk(Mat& frame1, Mat& frame2, vector<Point2f>& features, vector<Point2f>& new_features, vector<uchar>& status, int win_size, int level, float min_eigen_threshold, int max_iterations, float eps_criteria)
{
    new_features.erase(new_features.begin(),new_features.end());
    new_features.clear();

    if(features.empty())
    {
        //std::cout<<"No feature to track"<<std::endl;
        return;
    }

    //std::cout<<features.size()<<" features to track"<<std::endl;
    status.reserve(features.size());

    TicToc time_exec;
    time_exec.tic();

    Mat frame1_float, frame2_float;
    frame1.convertTo(frame1_float, CV_32FC3,1/255.0);
    frame2.convertTo(frame2_float, CV_32FC3,1/255.0);

    if(!this->frame2_pyr_.img_levels.empty())
    {
        std::swap(this->frame1_pyr_.img_levels,this->frame2_pyr_.img_levels);
        this->frame1_pyr_.n_levels = this->frame2_pyr_.n_levels;
        //std::cout<<"swap"<<std::endl;
    }
    else
    {
        this->frame1_pyr_.img_levels.erase(this->frame1_pyr_.img_levels.begin(),this->frame1_pyr_.img_levels.end());
        this->frame1_pyr_.img_levels.clear();
        //std::cout<<"new pyr frame1"<<std::endl;
    }


    this->frame2_pyr_.img_levels.erase(this->frame2_pyr_.img_levels.begin(),this->frame2_pyr_.img_levels.end());
    this->frame2_pyr_.img_levels.clear();

    
    if(this->frame1_pyr_.img_levels.empty())
    {
        //std::cout<<"reuse pyr frame1"<<std::endl;
        this->frame1_pyr_.create_pyramid(frame1_float,level,win_size);
    }
    this->frame2_pyr_.create_pyramid(frame2_float,level,win_size);
    // std::cout<<"Time create pyr: ";
    // time_exec.toc();


    //std::cout<<"levels: "<<this->frame1_pyr_.img_levels.size()<<std::endl;

    // namedWindow( "pyramid", 1 );

    // for(int i = 0; i < this->frame1_pyr_.n_levels ; i++)
    // {
    //     imshow("pyramid", this->frame1_pyr_.img_levels.at(i));
    //     waitKey(0);
    //     imshow("pyramid", this->frame2_pyr_.img_levels.at(i));
    //     waitKey(0);
    // }

    //std::cout<<this->frame1_pyr_.n_levels<<std::endl;

    // time_exec.tic();

    // For all features
    for(int f = 0; f < features.size(); f++)
    {
        status[f] = true;
        if(features.at(f).x < 0.0f || features.at(f).x > frame2.cols || features.at(f).y < 0.0f || features.at(f).y >= frame2.rows)
        {
            // Don't track feature that are outside the image
            new_features.push_back(Point2f(-1.0f,-1.0f));
            status[f] = false;
            continue;
        }
        //std::cout<<"feature "<<f<<" : "<<features.at(f)<<std::endl;
        
        Point2f d_position_final(0.0f,0.0f), pyramid_position(0.0f,0.0f);

        Status status_description = Status::Tracked;

        for(int temp_level = this->frame1_pyr_.img_levels.size() - 1; temp_level >= 0; temp_level-- )
        {
            Point2f current_point = Point2f(features.at(f).x / pow(2,temp_level),features.at(f).y / pow(2,temp_level));

            //std::cout<<"Feature "<<f<<": "<<current_point<<std::endl;
            // Define the area corresponding to the window
            float index_col_start = current_point.x - (float)win_size;
            if(index_col_start < 0.0f)
                index_col_start = 0.0f;
            float index_row_start = current_point.y - (float)win_size;
            if(index_row_start < 0.0f)
                index_row_start = 0.0f;
            float index_col_end = current_point.x + (float)win_size;
            if(index_col_end >= (float)this->frame1_pyr_.img_levels.at(temp_level).cols)
                index_col_end = (float)this->frame1_pyr_.img_levels.at(temp_level).cols - 1.0f;
            float index_row_end = current_point.y + (float)win_size;
            if(index_row_end >= (float)this->frame1_pyr_.img_levels.at(temp_level).rows)
                index_row_end = (float)this->frame1_pyr_.img_levels.at(temp_level).rows - 1.0f;

            //std::cout<<"indices: "<<index_col_start<<", "<<index_col_end<<" // "<<index_row_start<<", "<<index_row_end<<std::endl;

            int nb_pix_win = (win_size * 2 + 1)*(win_size * 2 + 1), cpt = 0;
            Point2f derivatives[nb_pix_win];

            float g1 = 0.0f, g2 = 0.0f;
            float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f, gradx = 0.0f, grady = 0.0f;

            for (float i = index_col_start; i <= index_col_end; i += 1.0f)
            {
                for (float j = index_row_start; j <= index_row_end; j += 1.0f)
                { 
                    //std::cout<<"pixel en "<<i<<", "<<j<<" : "<<this->frame1_pyr_.img_levels.at(0).at<float>(Point2f(i,j))<<std::endl;
                    
                    g1 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i + 1.0f,j));
                    g2 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i - 1.0f,j));

                    gradx = (g1 - g2) / 2.0f;

                    g1 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j + 1.0f));
                    g2 = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j - 1.0f));

                    grady = (g1 - g2) / 2.0f;

                    derivatives[cpt] = Point2f(gradx, grady);
                    cpt++;

                    gxx += gradx * gradx;
                    gxy += gradx * grady;
                    gyy += grady * grady;
                }
            }

            float det = gxx * gyy - gxy * gxy;
            //float minEig = (gyy + gxx - std::sqrt((gxx-gyy)*(gxx-gyy) + 4.f*gxy*gxy))/(2*((win_size*2+1)*(win_size*2+1)));
            
            if(det < min_eigen_threshold)
            {
                status_description = Status::SmallDet;
                status[f] = false;
                break;
            }

            det = 1.0f/det;
            //std::cout<<"grad mat: "<<gradient_mat<<std::endl<<std::endl;

            // time_exec.tic();
            Point2f d_position(0.0f,0.0f), delta(0.0f,0.0f), delta_prev(0.0f,0.0f), image_mismatch(0.0f,0.0f);
            float next_index_j,next_index_i,img_difference;

            for(int iteration = 0; iteration < max_iterations; iteration++)
            {
                image_mismatch = Point2f(0.0f,0.0f);
                cpt = 0;

                for (float i = index_col_start; i <= index_col_end; i += 1.0f)
                {
                    for (float j = index_row_start; j <= index_row_end; j += 1.0f)
                    {
                        next_index_i = i + pyramid_position.x + d_position.x;
                        next_index_j = j + pyramid_position.y + d_position.y;

                        //std::cout<<"next index i: "<<next_index_i<<"/ j: "<<next_index_j<<std::endl;

                        img_difference = get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j)) - get_subpixel_value(this->frame2_pyr_.img_levels.at(temp_level),Point2f(next_index_i,next_index_j));
                        image_mismatch.x += img_difference * derivatives[cpt].x;
                        image_mismatch.y += img_difference * derivatives[cpt].y;

                        cpt++;
                    }
                }

                //std::cout<<"image mismatch"<<image_mismatch<<std::endl;
                delta.x = det * (gyy * image_mismatch.x - gxy * image_mismatch.y);
                delta.y = det * (-gxy * image_mismatch.x + gxx * image_mismatch.y);

                d_position.x += delta.x;
                d_position.y += delta.y;
                //std::cout<<"d_position: "<<d_position<<std::endl;

                iteration++;
                //std::cout<<"iteration: "<<iteration<<std::endl;
                

                if(std::abs(delta.x) <= eps_criteria && std::abs(delta.y) <= eps_criteria)
                    break;

                if(iteration > 0 && std::abs(delta_prev.x + delta.x) < 0.001f && std::abs(delta_prev.y + delta.y) < 0.001f)
                {
                    d_position.x -= 0.5f * delta.x;
                    d_position.y -= 0.5f * delta.y;
                    break;
                }

                delta_prev = delta;
            }
            // std::cout<<"descente gradient: ";
            // time_exec.toc();
            //std::cout<<"iteration: "<<iteration<<std::endl;

            
            // // Check for Large Residue
            // float sum_window = 0.0f;
            // for (float i = index_col_start; i <= index_col_end; i += 1.0f)
            // {
            //     for (float j = index_row_start; j <= index_row_end; j += 1.0f)
            //     {
            //         float next_index_i = i + pyramid_position.x + d_position.x;
            //         float next_index_j = j + pyramid_position.y + d_position.y;

            //         sum_window += std::abs(get_subpixel_value(this->frame1_pyr_.img_levels.at(temp_level),Point2f(i,j)) - get_subpixel_value(this->frame2_pyr_.img_levels.at(temp_level),Point2f(next_index_i,next_index_j)));
            //     }
            // }
            // //std::cout<<"Residue: "<<sum_window / (float)((win_size*2 + 1)*(win_size*2 + 1))<<std::endl;
            // if(sum_window / (float)((win_size*2 + 1)*(win_size*2 + 1)) > 10.0f)
            // {
            //     status_description = Status::LargeResidue;
            //     status[f] = false;
            //     break;
            // }

            if(temp_level == 0)
            {
                d_position_final = d_position;
            }
            else
            {
                pyramid_position = 2.0f * (pyramid_position + d_position);
            }
        }

        d_position_final += pyramid_position;
        //std::cout<<"d_position finale: "<<d_position_final<<std::endl;
        //std::cout<<"Pos init: "<<features.at(f)<<std::endl;
        //std::cout<<"Status: "<<status_description<<std::endl;

        Point2f new_pos = features.at(f) + d_position_final;

        //std::cout<<"Pos finale: "<<new_pos<<std::endl;
        if(status[f] == false || new_pos.x < 0.0f || new_pos.x >= frame2.cols || new_pos.y < 0.0f || new_pos.y >= frame2.rows)
        {
            //std::cout<<"feature "<<f<<" erased. Status: "<<status_description<<std::endl;
            new_features.push_back(Point2f(-1.0f,-1.0f));
            status[f] = false;
            continue;
        }
        else
        {
            new_features.push_back(new_pos); 
            //std::cout<<"add feature "<<f<<std::endl; 
        }
    }
    // std::cout<<"Time track features: ";
    // time_exec.toc();
}

void OptFlowLK::release_pyr()
{
    this->frame1_pyr_.img_levels.erase(this->frame1_pyr_.img_levels.begin(),this->frame1_pyr_.img_levels.end());
    this->frame1_pyr_.img_levels.clear();

    this->frame2_pyr_.img_levels.erase(this->frame2_pyr_.img_levels.begin(),this->frame2_pyr_.img_levels.end());
    this->frame2_pyr_.img_levels.clear();
}