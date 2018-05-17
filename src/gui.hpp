#ifndef GUI_H
#define GUI_H

#include <nana/gui/wvl.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/widgets/label.hpp>
#include <nana/gui/widgets/checkbox.hpp>
#include <opencv2/opencv.hpp>
#include "opt_flow_process_directory_video.hpp"


using namespace nana;
using namespace std;

class Gui{

    public:

    Gui();
    void init_gui();
    void execute();


    private:
    void set_parameters();

    form fm;

    label path_label{fm};
    label max_features_label{fm};
    label quality_level_label{fm};
    label min_distance_label{fm};
    label block_size_label{fm};
    label use_harris_detector_label{fm};
    label win_size_height_label{fm};
    label win_size_width_label{fm};
    label max_level_pyramids_label{fm};
    label max_iterations_label{fm};
    label min_eigen_threshold_label{fm};
    label epsilon_criteria_label{fm};
    label use_opencv_lk_label_{fm};

    textbox path_box{fm};
    textbox max_features_box{fm};
    textbox quality_level_box{fm};
    textbox min_distance_box{fm};
    textbox block_size_box{fm};
    textbox win_size_height_box{fm};
    textbox win_size_width_box{fm};
    textbox max_level_pyramids_box{fm};
    textbox max_iterations_box{fm};
    textbox min_eigen_threshold_box{fm};
    textbox epsilon_criteria_box{fm};

    checkbox use_harris_detector_box{fm};
    checkbox use_opencv_lk_box_{fm};

    button btn_start{fm};
    button btn_quit{fm};

    place layout{fm};

    OptFlowProcessDirectoryVideo directory;

};

#endif