#include <iostream>
#include <stdio.h>
#include <nana/gui/wvl.hpp>
#include <nana/gui/widgets/button.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/widgets/label.hpp>
#include "gui.hpp"

using namespace std;

int main(int argc, char** argv)
{
    Gui gui;
    gui.init_gui();
    gui.execute();

    return 0;
}