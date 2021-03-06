// WebcamTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
using namespace cv;

int main()
{
	VideoCapture cap;

	if (!cap.open(1))
		return 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		Mat OutputImage;

		inRange(frame, Scalar(20, 80, 70), Scalar(50, 150, 150), OutputImage);
		imshow("Output", OutputImage);

		if (waitKey(10) == 27) break;

		SimpleBlobDetector::Params params;

		params.minThreshold = 10;
		params.maxThreshold = 200;
		params.filterByArea = true;
		params.minArea = 200;
		params.filterByCircularity = false;
		params.minCircularity = 0.1;
		params.filterByConvexity = false;
		params.minConvexity = 0.87;
		params.filterByInertia = false;
		params.minInertiaRatio = 0.01;

		bitwise_not(OutputImage, OutputImage);
		std::vector<KeyPoint> keypoints;
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
		detector->detect(OutputImage, keypoints);

	}

	return 0;
}

