#include "pch.h"
#include "opencv2/opencv.hpp"
#include "windows.h"
using namespace cv;
int main(int argc, char** argv)
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
		inRange(frame, Scalar(70, 60, 50), Scalar(255, 110, 60), OutputImage);
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
		double x, y;
		for (int i = 0; i < keypoints.size(); i++) {
			std::cout << keypoints[i].pt;
			x = keypoints[i].pt.x;
			y = keypoints[i].pt.y;
		}

		Mat im_with_keypoints;
		drawKeypoints(frame, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("keypoints", im_with_keypoints);
	}
	// the camera will be closed automatically upon exit
	// cap.close();
	return 0;
}