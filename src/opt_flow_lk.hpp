#ifndef OPTFLOWLK_H
#define OPTFLOWLK_H

#include <opencv2/opencv.hpp>
#include "pyramid.hpp"
#include "conv.hpp"
#include "tictoc.hpp"
#include <cmath> 

using namespace std;
using namespace cv;

enum Status {Tracked, NotConverge, LargeResidue, OutOfBounds, SmallDet};

class OptFlowLK
{
    public:

    OptFlowLK();
    void compute_lk(Mat& frame1, Mat& frame2, vector<Point2f>& features, vector<Point2f>& new_features, int win_size, int level, float min_eigen_threshold, int max_iterations, float eps_criteria);

    private:
    
    Pyramid frame1_pyr_;
    Pyramid frame1_gradx_pyr;
    Pyramid frame1_grady_pyr;
    Pyramid frame2_pyr_;
    Pyramid frame2_gradx_pyr;
    Pyramid frame2_grady_pyr;
};

#endif