#ifndef TICTOC_H
#define TICTOC_H

#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

class TicToc{
	public:
	TicToc();
	double tt_tic;
 
	void tic();
	double toc();
};

#endif