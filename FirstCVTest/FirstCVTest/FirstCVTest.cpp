#include "pch.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
using namespace cv;
using namespace std;

int main() {

	Mat image = imread("C:/Users/jmsdesk/Desktop/coloured-balls.jpeg", 1);
	imshow("Image", image);
	Mat OutputImage;
	inRange(image, Scalar(0, 0, 200), Scalar(0, 0, 255), OutputImage);
	imshow("Output", OutputImage);
	waitKey(0);

	SimpleBlobDetector::Params params;

	params.minThreshold = 10;
	params.maxThreshold = 200;
	params.filterByArea = true;
	params.minArea = 10000;
	params.filterByCircularity = false;
	params.minCircularity = 0.1;
	params.filterByConvexity = false;
	params.minConvexity = 0.87;
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;

	vector<KeyPoint> keypoints;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	detector->detect(OutputImage, keypoints);

	Mat im_with_keypoints;
	drawKeypoints(OutputImage, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keypoints", im_with_keypoints);
	waitKey(0);
}