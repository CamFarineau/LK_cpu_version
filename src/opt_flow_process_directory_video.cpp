#include "opt_flow_process_directory_video.hpp"

OptFlowProcessDirectoryVideo::OptFlowProcessDirectoryVideo(){
    this->video_to_process = OptFlowVideo();
    this->directory_frame_to_process = OptFlowFrame();
}

// Function to process all the videos that are all in the same folder
// Parameters: path -> path to the folder, show_output (optional): show the results while running
void OptFlowProcessDirectoryVideo::process_all_videos_folder(std::string path, bool write_json_vector, bool show_output)
{
    // Get the path to the dir
    boost::filesystem::path video_dir(path);


    if(exists(video_dir))
    {
        std::cout<<"**********************************************************************"<<std::endl;
        std::cout<<"Processing: "<<video_dir.string()<<std::endl;
        std::cout<<"**********************************************************************"<<std::endl;

        boost::filesystem::directory_iterator end_iter;

        // Vector of path that will contains all the videos
        vector<boost::filesystem::path> videos_set;
        // Vector of path that will contains all the directory of frames
        vector<boost::filesystem::path> frames_set;
        // Go through all the files in the folder
        for( boost::filesystem::directory_iterator dir_iter(video_dir) ; dir_iter != end_iter ; ++dir_iter)
        {
            if(boost::filesystem::is_directory(dir_iter->path()))
            {
                frames_set.push_back(dir_iter->path());
            }
            else
            {
                // Insert the path of each video into the vector of path
                videos_set.push_back(dir_iter->path());
            }
        }

        if(videos_set.empty() && frames_set.empty())
        {
            std::cout<<"**********************************************************************"<<std::endl;
            std::cout<<"Directory is empty: "<<video_dir.string()<<std::endl;
            std::cout<<"**********************************************************************"<<std::endl;
            return;
        }

        // For each video in the folder, process everything
        for(int i = 0; i < videos_set.size(); i++)
        {
            std::cout<<"Processing video: "<<videos_set.at(i).string()<<std::endl<<std::endl;
            // Set the correct filename of the video in question
            this->video_to_process.set_filepath(videos_set.at(i).string());
            // Get the image with the features
            this->video_to_process.write_image_with_init_features(show_output);
            // Get the image with the optical flow
            this->video_to_process.write_image_with_optical_flow(show_output);
            // Get the video with the vector representing the optical flow
            this->video_to_process.write_vector_video(write_json_vector,show_output);
            // Release the video at the end of the processing
            this->video_to_process.release_video();
            std::cout<<"**********************************************************************"<<std::endl;
        }

        // For each directory contening frames in the folder, process everything
        for(int j = 0; j < frames_set.size(); j++)
        {
            std::cout<<"Processing directory of frames: "<<frames_set.at(j).string()<<std::endl<<std::endl;
            // Set the correct directory path
            this->directory_frame_to_process.set_directory_path(frames_set.at(j).string());
            // Get the first frame with the initial features
            this->directory_frame_to_process.write_first_frame_with_init_features(show_output);
            // Get the first frame with the optical flow
            this->directory_frame_to_process.write_first_frame_with_optical_flow(show_output);
            // Get the video with the vector representing the optical flow
            this->directory_frame_to_process.write_vector_video(write_json_vector,show_output);
            // Release the frames from the vector at the end of the processing
            this->directory_frame_to_process.release_vector_frames();
            std::cout<<"**********************************************************************"<<std::endl;
            
        }
        
        std::cout<<"**********************************************************************"<<std::endl;
        std::cout<<"All videos / frames of: "<<video_dir.string()<<" have been processed !"<<std::endl;
        std::cout<<"**********************************************************************"<<std::endl;

        cv::destroyAllWindows();
    }
    else
    {
        std::cout<<"**********************************************************************"<<std::endl;
        std::cout<<"Directory: "<<video_dir.string()<<" does not exist!"<<std::endl;
        std::cout<<"**********************************************************************"<<std::endl;
    }


}