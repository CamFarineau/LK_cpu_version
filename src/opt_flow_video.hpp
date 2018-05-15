#ifndef OPTFLOWVIDEO_H
#define OPTFLOWVIDEO_H

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <string> 
#include "tictoc.hpp"
#include "opt_flow_lk.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

/* Class in order to get the optical flow of a video
*  A single video ca be processed with this class. Give it the filename (path to the video) and parameters and call the needed functions
*  If you want to process a lot of videos, put them together in the same folder and use the OptFlowProcessDirectoryVideo class
*/
class OptFlowVideo{
    public:

        // Default constructor
        OptFlowVideo();
        // Constructor with the filename as arguments can be useful
        OptFlowVideo(string filepath_temp);

        // Function to get (write in a file) the first frame of the video with the important features that will be tracked
        void write_image_with_init_features(bool show_output = false);
        // Function to get (write in a file) the first frame of the video with the optical flow of each of the features
        void write_image_with_optical_flow(bool show_output = false);
        // Function to get (write in a file) the same video with vectors representing the optical flow of each features at each frame
        void write_vector_video(bool write_json_vector = true, bool show_output = false, int fps = 25);
        // Function to release the video (deallocate memory)
        void release_video();

        // {get, set} for filepath
        void set_filepath(string filepath_temp);
        string get_filepath(){return filepath_;};

        // {get, set} for max_features
        void set_max_features(int max_features_temp){max_features_ = max_features_temp;};
        int get_max_features(){return max_features_;};

        // {get, set} for quality_level
        void set_quality_level(double quality_level_temp){quality_level_ = quality_level_temp;};
        double get_quality_level(){return quality_level_;};

        // {get, set} for min_distance
        void set_min_distance(double min_distance_temp){min_distance_ = min_distance_temp;};
        double get_min_distance(){return min_distance_;};

        // {get, set} for block_size
        void set_block_size(int block_size_temp){block_size_ = block_size_temp;};
        int get_block_size(){return block_size_;};

        // {get, set} for use_harris_detector
        void set_use_harris_detector(bool _use_harris_detector_temp){use_harris_detector_ = _use_harris_detector_temp;};
        bool get_use_harris_detector(){return use_harris_detector_;};
        
        // {get, set} for win_size_
        void set_win_size(Size win_size_temp){win_size_ = win_size_temp;};
        Size get_win_size(){return win_size_;};

        // {get, set} for max_level_pyramids_
        void set_max_level_pyramids(int max_level_pyramids_temp){max_level_pyramids_ = max_level_pyramids_temp;};
        int get_max_level_pyramids(){return max_level_pyramids_;};

        // {get, set} for max_iterations_
        void set_max_iterations(int max_iterations_temp){max_iterations_ = max_iterations_temp;};
        int get_max_iterations(){return max_iterations_;};

        // {get, set} for min_eigen_threshold_
        void set_min_eigen_threshold(float min_eigen_threshold_temp){min_eigen_threshold_ = min_eigen_threshold_temp;};
        float get_min_eigen_threshold(){return min_eigen_threshold_;};

        // {get, set} for epsilon_criteria_
        void set_epsilon_criteria(float epsilon_criteria_temp){epsilon_criteria_ = epsilon_criteria_temp;};
        float get_epsilon_criteria(){return epsilon_criteria_;};

        // {get, set} for term_criteria_
        void set_term_criteria(TermCriteria term_criteria_temp){term_crit_ = term_criteria_temp;};
        TermCriteria get_term_criteria(){return term_crit_;};

    private:

        //Video attributs:
        string filepath_;
        string filename_;
        // Boolean to know if the video change (in case of true the new video will be loaded in order to proces it)
        bool change_name_;
        // The video itself
        VideoCapture video_;
        // first frame of the video
        Mat first_frame_;

        string path_to_data_folder_;

        //Features: vector of Point2f
        vector<Point2f> init_features_;

        //goodFeaturesToTrack parameters
        int max_features_;
        double quality_level_;
        double min_distance_;
        int block_size_;
        bool use_harris_detector_;

        //calcOpticalFlowPyrLK parameters
        TermCriteria term_crit_;
        Size win_size_;
        int max_level_pyramids_;

        //Own LK OpticalFlow calc
        int max_iterations_;
        float min_eigen_threshold_;
        float epsilon_criteria_;

        //LK class to estimate flow
        OptFlowLK estim_flow;

        // Get the initial features
        // All the parameters need to be initialized
        int search_init_features();
        
        // Function to load the video
        // Return 0 if successful, 1 if not
        int load_video();
};

#endif