// WebcamTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
using namespace cv;

int main()
{
	VideoCapture cap;

	if (!cap.open(0))
		return 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		Mat OutputImage;
		inRange(frame, Scalar(80, 55, 180), Scalar(130, 95, 255), OutputImage);
		imshow("Output", OutputImage);

		if (waitKey(10) == 27) break;

	}

	return 0;
}

