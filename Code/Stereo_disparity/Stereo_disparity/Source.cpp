#include <stdio.h>
#include <iostream>
#include <ostream>
#include <direct.h>
#include <string.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


void disp_calculate(Mat img_1, Mat img_2, Mat gt, String str, String out_dir, int  disp_ratio, float ratio, String disparity_type)
{

	unsigned long start_time = 0, finish_time = 0; //check processing time  
	start_time = getTickCount(); //check processing time 
	Ptr<Feature2D> f2d;
	Ptr<DescriptorMatcher> matcher;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	if(str=="sift")
	{
		f2d = SIFT::create();
		matcher = new FlannBasedMatcher(new flann::KDTreeIndexParams(4));
	}
	else if (str == "orb")
	{
		f2d= ORB::create();
		matcher = new FlannBasedMatcher(new flann::LshIndexParams(5, 20, 2));	//Using ORB, Flann-based Locality-sensitive hashing(LSH) for matching:

	}

	if (disparity_type == "dense")
	{
		for (int i = 0; i < img_1.cols; i++)
			for (int j = 0; j < img_1.rows; j++)			//Dense disparity
			{
				keypoints_1.push_back(KeyPoint(i, j, 1));
				keypoints_2.push_back(KeyPoint(i, j, 1));
			}
		f2d->compute(img_1, keypoints_1, descriptors_1);			// To get descriptors at each and every pixel(dense disparity)
		f2d->compute(img_2, keypoints_2, descriptors_2);			// To get descriptors at each and every pixel(dense disparity)		
	}

	else if (disparity_type == "sparse")
	{
		f2d->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);			//Sparse Disparity
		f2d->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);			//Sparse Disparity
	}
	vector< DMatch > matches;
	matcher->match(descriptors_1, descriptors_2, matches);

	// Calculating best matches of obtained matches by using Lowe's Ratio:
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//printf("\n-- Max dist : %f ", max_dist);
	//printf("\n-- Min dist : %f ", min_dist);
	std::vector< DMatch > good_matches;
	vector<Point2f>imgpts1, imgpts2;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance <= max(ratio*max_dist, 0.02)) {
			good_matches.push_back(matches[i]);
			imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
			imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
		}
	}

	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("Good Matches_" + str, cv::WINDOW_AUTOSIZE);
	imshow("Good Matches_" + str, img_matches);
	imwrite(out_dir + "/Good Matches_" + str + ".png", img_matches);

	//Calculating dispairity and disparity map:
	Mat disparity(img_1.size().height, img_1.size().width, CV_8U, Scalar(255));
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		int x = keypoints_1[good_matches[i].queryIdx].pt.x;
		int y = keypoints_1[good_matches[i].queryIdx].pt.y;
		int x1 = keypoints_2[good_matches[i].trainIdx].pt.x;
		//printf("\nx point %d, y point %d,disparity-->%d", x, x1, abs(x - x1));
		//printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  ", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
		disparity.at<uchar>(y, x) = abs(x - x1);
	}

	//Calculating the RMS value of the dispairty
	long float rms = 0, count = 0, max_disp = 0;
	for (int i = 0; i < gt.size().width; i++) {
		for (int j = 0; j < gt.size().height; j++) {
			if (disparity.at<uchar>(j, i) != 255)
			{
				int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
				if (a > max_disp)
					max_disp = a;
				//printf("%d, %d, %d, %d\n",disparity(j,i),gt.at<uchar>(j, i)/4, a, a*a);
				rms = rms + a*a;
				count = count + 1;
			}
		}
	}
	printf("\n No. of disparities--->%d,   rms--->%f",  (int)count, sqrt(rms / count));


	//Calculating the Bad pixel % at dispairties for thresholds-1,2,5
	int delta[3] = {1, 2, 5};
	float bad[3] = {0.0,0.0,0.0};
	for(int k=0;k<3;k++)
	{
		int threshold_temp = delta[k];
		for (int i = 0; i < gt.size().width; i++)
			for (int j = 0; j < gt.size().height; j++)
			{
				if (disparity.at<uchar>(j, i) != 255)
				{
					int a = abs(disparity.at<uchar>(j, i) - gt.at<uchar>(j, i) / disp_ratio);
					//printf("%d, %d, %d\n", disparity.at<uchar>(j, i),gt.at<uchar>(j, i)/4,a );
					if (a > threshold_temp)
						bad[k] = bad[k] + a;
				}
			}

		printf("\n bad pixel percent(delta>%d)--->%f",threshold_temp, bad[k] / count);
	}


	//Grayscale Normalization
	for (int i = 0; i < gt.size().width; i++)
		for (int j = 0; j < gt.size().height; j++)
			if (disparity.at<uchar>(j, i) != 255)
				disparity.at<uchar>(j, i) = disparity.at<uchar>(j, i)*disp_ratio * 255 / max_disp;
		
	finish_time = getTickCount(); //check processing time
	printf("\nElapsed Time : %.2lf sec \n", (finish_time - start_time) / getTickFrequency());         //check processing time  
			
	imshow("disparity_" + str + ";    rms:" + to_string(sqrt(rms / count)) + "    bp%: " + to_string(bad[0] / count), disparity);
	imwrite(out_dir + "/disparity_" + str + ".png", disparity);
	imshow("ground Truth",gt);
}



int main()
{
	String dataset_all[3] = { "Cones","Teddy","Art" }; 
	String disparity_type = "sparse";  //For sparse disparity
	
    //String dataset_all[2] = { "Cones","Teddy"};      //Uncomment this line to find the dense disparities
	//String disparity_type = "dense";			       //Uncomment this line to find the dense disparities
	
	int size = 0; for (auto c : dataset_all) { size++; };
	printf("Stereo correspondence using SIFT, ORB\n");
	for (int i = 0; i < size;i++) {
		String dataset = dataset_all[i];
		String out_dir = "../../../Output/" + dataset + "_out_" + disparity_type;
		_mkdir(out_dir.c_str());

		Mat img_1 = imread("../../../Input/" + dataset + "/left.png", IMREAD_GRAYSCALE);
		Mat img_2 = imread("../../../Input/" + dataset + "/right.png", IMREAD_GRAYSCALE);
		Mat gt = imread("../../../Input/" + dataset + "/left_gt.png");
		if (img_1.empty() || img_2.empty() || gt.empty())
		{
			printf("Image Not Loaded Properly\n");
			return -1;
		}

		int disp_ratio = 4;
		if (dataset == "Cones" || dataset == "Teddy")
			disp_ratio = 4;								// depends on the dataset ground truth scaling factor
		else
			disp_ratio = 1;
		float ratio_orb = 0.0;
		float ratio_sift = 0.0;
		if (disparity_type == "sparse") {
			if (dataset == "Cones") {ratio_sift = 0.4;ratio_orb = 0.4;}
			else if (dataset == "Teddy") {ratio_sift = 0.3;	ratio_orb = 0.3;}
			else if (dataset == "Art") {ratio_sift = 0.1;ratio_orb = 0.3;}
		}
		else {ratio_sift = 0.1;	ratio_orb = 0.1;}
	// Stereo corrspondence Using SIFT and Flann based kd-tree matching cost. 
		printf("\nSIFT-%s",dataset.c_str());
		disp_calculate(img_1, img_2, gt, "sift", out_dir, disp_ratio, ratio_sift, disparity_type);
		
	//Stereo ORB correspondence using Flann based Locality sensitive hashing matching cost 
		printf("\nORB-%s", dataset.c_str());
		disp_calculate(img_1, img_2, gt, "orb", out_dir, disp_ratio, ratio_orb, disparity_type);
	}
	waitKey(0);
}
