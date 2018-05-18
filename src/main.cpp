#include <iostream>
#include <stdio.h>
#include <nana/gui/wvl.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/widgets/label.hpp>
#include <opencv2/opencv.hpp>
#include "gui.hpp"

using namespace std;

int main(int argc, char** argv)
{
    cv::setUseOptimized(false);
    cv::setNumThreads(1);
    Gui gui;
    gui.init_gui();
    gui.execute();

    return 0;
}