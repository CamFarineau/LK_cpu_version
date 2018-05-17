#include "tictoc.hpp"

TicToc::TicToc(){
    this->tt_tic = 0;
}

void TicToc::tic(){
    tt_tic = getTickCount();
}
double TicToc::toc(){
    double tt_toc = (getTickCount() - tt_tic)/(getTickFrequency());
    std::cout<<tt_toc<<std::endl;
    return tt_toc;
}