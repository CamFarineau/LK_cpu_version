#ifndef OPTFLOWPROCESSDIRECTORYVIDEO_H
#define OPTFLOWPROCESSDIRECTORYVIDEO_H

#include <boost/filesystem.hpp>
#include "opt_flow_video.hpp"
#include "opt_flow_frame.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace boost::filesystem;

/* Class to process videos (get opt flow) that are all in the same folder
*  Inconvenient: all videos will be processed with the same parameters for the optical flow.
*  It is possible to implement a way to read parameter out of json file for example.
*/ 
class OptFlowProcessDirectoryVideo{
    
    public:
    OptFlowProcessDirectoryVideo();
    // Function to process all the video that are contain in the same folder
    void process_all_videos_folder(std::string path, bool write_json_vector = true, bool show_output = false);
    // Create an OptFlowVideo
    OptFlowVideo video_to_process;
    // Create an OptFlowFrames
    OptFlowFrame directory_frame_to_process;
};

#endif