#include "pch.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	CascadeClassifier faceCascade;
	faceCascade.load("C:/Users/jmsdesk/Documents/OpenCV/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml");

	VideoCapture cap;

	if (!cap.open(0))
		return 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		//imshow("Output", frame);
		Mat frameGray;
		cvtColor(frame, frameGray, COLOR_BGR2GRAY);
		equalizeHist(frameGray, frameGray);
		vector<Rect> faces;
		faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
			Mat faceROI = frameGray(faces[i]);
		}
		imshow("Face Detection", frame);
		if (waitKey(10) == 27) break;


	}
	return 0;

}
