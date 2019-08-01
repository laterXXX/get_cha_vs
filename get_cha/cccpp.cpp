#include  <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <features2d\features2d.hpp>
#include <vector> 
#include <fstream>
#include <ctime>
using namespace cv;
using namespace std;

void main123()
{

	ifstream file;
	file.open("H:/rect_license/img_names.txt");
	assert(file.is_open());
	string img_name;
	while (getline(file, img_name))
	{
		//轮廓最小外接矩形的绘制
		string imgstr(img_name);
		Mat srcImg = imread(imgstr);
		int w = srcImg.cols;
		int h = srcImg.rows;
		resize(srcImg, srcImg, Size(1200, 400));
		Mat dstImg = srcImg.clone();
		cvtColor(srcImg, srcImg, CV_BGR2GRAY);
		threshold(srcImg, srcImg, 100, 255, CV_THRESH_BINARY); //二值化
		srcImg = 255 - srcImg;
		imshow("threshold", srcImg);
		waitKey(0);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarcy;
		findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		//vector<Rect> boundRect(contours.size());  //定义外接矩形集合
		vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
		Point2f rect[4];
		for (int i = 0; i < contours.size(); i++)
		{
			box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
													 //boundRect[i] = boundingRect(Mat(contours[i]));
			circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
			box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
								  //rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
			for (int j = 0; j < 4; j++)
			{
				line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
			}
		}
		imshow("dst", dstImg);
		waitKey(0);

		//移除过长或过短的轮廓  
		int cmin = 100; //最小轮廓长度  
		int cmax = 1000;    //最大轮廓  
		vector<vector<Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end())
		{
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}

		//在白色图像上绘制黑色轮廓  
		Mat result_erase(srcImg.size(), CV_8U, Scalar(255));
		drawContours(result_erase, contours,
			-1, //绘制所有轮廓  
			Scalar(0),  //颜色为黑色  
			2); //轮廓线的绘制宽度为2  

		namedWindow("contours_erase");
		imshow("contours_erase", result_erase);
		waitKey(0);
	}
}
void drawLine(cv::Mat &image, double theta, double rho, cv::Scalar color)
{
	if (theta < 3.1415 / 4. || theta > 3.*3.1415 / 4.)// ~vertical line
	{
		cv::Point pt1(rho / cos(theta), 0);
		cv::Point pt2((rho - image.rows * sin(theta)) / cos(theta), image.rows);
		cv::line(image, pt1, pt2, cv::Scalar(255), 1);
	}
	else
	{
		cv::Point pt1(0, rho / sin(theta));
		cv::Point pt2(image.cols, (rho - image.cols * cos(theta)) / sin(theta));
		cv::line(image, pt1, pt2, color, 1);
	}
}
//生成[0,1]之间符合均匀分布的数
double uniformRandom(void)
{
	return (double)rand() / (double)RAND_MAX;
}

//生成[0,1]之间符合高斯分布的数
double gaussianRandom(void)
{
	/* This Gaussian routine is stolen from Numerical Recipes and is their
	copyright. */
	static int next_gaussian = 0;
	static double saved_gaussian_value;

	double fac, rsq, v1, v2;

	if (next_gaussian == 0) {
		do {
			v1 = 2 * uniformRandom() - 1;
			v2 = 2 * uniformRandom() - 1;
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2 * log(rsq) / rsq);
		saved_gaussian_value = v1 * fac;
		next_gaussian = 1;
		return v2 * fac;
	}
	else {
		next_gaussian = 0;
		return saved_gaussian_value;
	}
}

//根据点集拟合直线ax+by+c=0，res为残差
void calcLinePara(vector<Point2d> pts, double &a, double &b, double &c, double &res)
{
	res = 0;
	Vec4f line;
	vector<Point2f> ptsF;
	for (unsigned int i = 0; i < pts.size(); i++)
		ptsF.push_back(pts[i]);

	fitLine(ptsF, line, CV_DIST_L2, 0, 1e-2, 1e-2);
	a = line[1];
	b = -line[0];
	c = line[0] * line[3] - line[1] * line[2];

	for (unsigned int i = 0; i < pts.size(); i++)
	{
		double resid_ = fabs(pts[i].x * a + pts[i].y * b + c);
		res += resid_;
	}
	res /= pts.size();
}

//得到直线拟合样本，即在直线采样点集上随机选2个点
bool getSample(vector<int> set, vector<int> &sset)
{
	int i[2];
	if (set.size() > 2)
	{
		do
		{
			for (int n = 0; n < 2; n++)
				i[n] = int(uniformRandom() * (set.size() - 1));
		} while (!(i[1] != i[0]));
		for (int n = 0; n < 2; n++)
		{
			sset.push_back(i[n]);
		}
	}
	else
	{
		return false;
	}
	return true;
}

//直线样本中两随机点位置不能太近
bool verifyComposition(const vector<Point2d> pts)
{
	cv::Point2d pt1 = pts[0];
	cv::Point2d pt2 = pts[1];
	if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
		return false;

	return true;
}

//RANSAC直线拟合
void fitLineRANSAC(vector<Point2d> ptSet, double &a, double &b, double &c, vector<bool> &inlierFlag)
{
	double residual_error = 2.99; //内点阈值

	bool stop_loop = false;
	int maximum = 0;  //最大内点数

					  //最终内点标识及其残差
	inlierFlag = vector<bool>(ptSet.size(), false);
	vector<double> resids_(ptSet.size(), 3);
	int sample_count = 0;
	int N = 500;

	double res = 0;

	// RANSAC
	srand((unsigned int)time(NULL)); //设置随机数种子
	vector<int> ptsID;
	for (unsigned int i = 0; i < ptSet.size(); i++)
		ptsID.push_back(i);
	while (N > sample_count && !stop_loop)
	{
		vector<bool> inlierstemp;
		vector<double> residualstemp;
		vector<int> ptss;
		int inlier_count = 0;
		if (!getSample(ptsID, ptss))
		{
			stop_loop = true;
			continue;
		}

		vector<Point2d> pt_sam;
		pt_sam.push_back(ptSet[ptss[0]]);
		pt_sam.push_back(ptSet[ptss[1]]);

		if (!verifyComposition(pt_sam))
		{
			++sample_count;
			continue;
		}

		// 计算直线方程
		calcLinePara(pt_sam, a, b, c, res);
		//内点检验
		for (unsigned int i = 0; i < ptSet.size(); i++)
		{
			Point2d pt = ptSet[i];
			double resid_ = fabs(pt.x * a + pt.y * b + c);
			residualstemp.push_back(resid_);
			inlierstemp.push_back(false);
			if (resid_ < residual_error)
			{
				++inlier_count;
				inlierstemp[i] = true;
			}
		}
		// 找到最佳拟合直线
		if (inlier_count >= maximum)
		{
			maximum = inlier_count;
			resids_ = residualstemp;
			inlierFlag = inlierstemp;
		}
		// 更新RANSAC迭代次数，以及内点概率
		if (inlier_count == 0)
		{
			N = 500;
		}
		else
		{
			double epsilon = 1.0 - double(inlier_count) / (double)ptSet.size(); //野值点比例
			double p = 0.99; //所有样本中存在1个好样本的概率
			double s = 2.0;
			N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
		}
		++sample_count;
	}

	//利用所有内点重新拟合直线
	vector<Point2d> pset;
	for (unsigned int i = 0; i < ptSet.size(); i++)
	{
		if (inlierFlag[i])
			pset.push_back(ptSet[i]);
	}

	calcLinePara(pset, a, b, c, res);
}


bool comp1(const Point &a, const Point &b)
{
	return a.x < b.x;
}

bool comp2(const RotatedRect &a, const RotatedRect &b)
{
	return a.center.x < b.center.x;
}

bool comp3(const Rect &a, const Rect &b)
{
	return a.x < b.x;
}

int main() {
	ifstream file;
	file.open("H:/rect_license/img_names.txt");
	assert(file.is_open());
	string img_name;
	while (getline(file, img_name))
	{
		Mat image = imread(img_name);
		resize(image, image, Size(1200, 300));
		
		Mat grayImage;
		cvtColor(image, grayImage, CV_BGR2GRAY);

		medianBlur(grayImage, grayImage, 9);
		imshow("grayImage", grayImage);

		//转换为二值图    
		Mat binaryImage;
		threshold(grayImage, binaryImage, 120, 255, CV_THRESH_BINARY);

		//Canny(grayImage, binaryImage, 100, 200);
		//imshow("binaryImage", binaryImage);

		//Mat element = getStructuringElement(MORPH_RECT, Size(30, 30), Point(-1, -1)); //定义结构元素
		//dilate(binaryImage, binaryImage, element); //膨胀
		//imshow("dilate", binaryImage);
		//erode(binaryImage, binaryImage, element);
		//imshow("erode", binaryImage);


		//二值图 这里进行了像素反转，因为一般我们用255白色表示前景（物体），用0黑色表示背景    
		Mat reverseBinaryImage;
		bitwise_not(binaryImage, reverseBinaryImage);
		imshow("bitwise_not", reverseBinaryImage);

		vector <vector<Point>>contours;
		findContours(reverseBinaryImage,
			contours,   //轮廓的数组  
			CV_RETR_EXTERNAL,   //获取外轮廓  
			CV_CHAIN_APPROX_NONE);  //获取每个轮廓的每个像素  
									//在白色图像上绘制黑色轮廓 

		Mat result(reverseBinaryImage.size(), CV_8U, Scalar(255));
		drawContours(result, contours,
			-1, //绘制所有轮廓  
			Scalar(0),  //颜色为黑色  
			2); //轮廓线的绘制宽度为2  

		namedWindow("contours");
		imshow("contours", result);

		//移除过长或过短的轮廓  
		int cmin = 100; //最小轮廓长度  
		int cmax = 1500;    //最大轮廓  
		vector<vector<Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end())
		{
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}

		double max_area = 0;
		vector<double> areas;
		int index = 0;
		Mat ROI;
		vector<Mat> rois;
		for (int i = 0; i < contours.size(); i++)
		{
			cout << contourArea(contours[i]) << endl;
		}

		Mat minAreaRect_erase(binaryImage.size(), CV_8U, Scalar(255));
		vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
		vector<Rect> boundRect(contours.size());  //定义外接矩形集合
		Point2f rect[4];
		vector<Point> p;
		Mat dst = binaryImage.clone();
		for (int i = 0; i < contours.size(); i++)
		{
			boundRect[i] = boundingRect(Mat(contours[i]));
			rectangle(image, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

			box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形

			p.push_back(box[i].center);	

			circle(binaryImage, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
			
			box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
			
			for (int j = 0; j < 4; j++)
			{
				line(binaryImage, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
			}

			//float angle;
			// cout << "angle=" << box[i].angle << endl;
			// angle = box[i].angle;
			////利用仿射变换进行旋转        另一种方法，透视变换
			//if (0< abs(angle) && abs(angle) <= 45)
			//	angle = angle;//负数，顺时针旋转
			//else if (45< abs(angle) && abs(angle)<90)
			//	angle = 90 - abs(angle);//正数，逆时针旋转
			//Point2f center = box[i].center;  //定义旋转中心坐标
			//double angle0 = angle;
			//double scale = 1;
			//Mat roateM = getRotationMatrix2D(center, angle0, scale);  //获得旋转矩阵,顺时针为负，逆时针为正
			//if (i == 0) {
			//	warpAffine(dst, dst, roateM, dst.size()); //仿射变换
			//}
			//		


			//int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			//x0 = boundRect[i].x;
			//y0 = boundRect[i].y;
			//w0 = boundRect[i].width;
			//h0 = boundRect[i].height;
			//ROI = dst(Rect(x0, y0, w0, h0));
			//imshow("pg", ROI);
			//waitKey(0);
		}

		sort(p.begin(), p.end(), comp1);
		sort(box.begin(), box.end(), comp2);
		sort(boundRect.begin(), boundRect.end(), comp3);

		for (int i = 0; i < box.size(); i++)
		{

			int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			x0 = boundRect[i].x;
			y0 = boundRect[i].y;
			w0 = boundRect[i].width;
			h0 = boundRect[i].height;
			ROI = dst(Rect(x0, y0, w0, h0));

			float angle;
			 cout << "angle=" << box[i].angle << endl;
			 angle = box[i].angle;
			//利用仿射变换进行旋转        另一种方法，透视变换
			if (0< abs(angle) && abs(angle) <= 45)
				angle = angle;//负数，顺时针旋转
			else if (45< abs(angle) && abs(angle)<90)
				angle = 90 - abs(angle);//正数，逆时针旋转
			Point2f center = box[i].center;  //定义旋转中心坐标
			double angle0 = angle;
			double scale = 1;
			Mat roateM = getRotationMatrix2D(center, angle0, scale);  //获得旋转矩阵,顺时针为负，逆时针为正
			
				warpAffine(ROI, ROI, roateM, dst.size()); //仿射变换
			

			imshow("pg", ROI);
			waitKey(0);
		}





		if(p.size() > 0){
		vector<Vec2f> lines;
		//HoughLines(binaryImage, lines, 1, CV_PI / 180, 150, 0, 0);
		cv::Vec4f lin;
		cv::fitLine(p,
			lin,
			CV_DIST_HUBER,
			0,
			0.01,
			0.01);
		double cos_theta = lin[0];
		double sin_theta = lin[1];
		double x0 = lin[2], y0 = lin[3];

		double phi = atan2(sin_theta, cos_theta) + 3.1415 / 2.0;
		double rho = y0 * cos_theta - x0 * sin_theta;

		double PI = 3.1415;
		std::cout << "phi = " << phi / PI * 180 << std::endl;
		std::cout << "rho = " << rho << std::endl;

		drawLine(binaryImage, phi, rho, cv::Scalar(0));
		}
#pragma region ransac
		//double A, B, C;
		//vector<bool> inliers;
		//fitLineRANSAC(p, A, B, C, inliers);

		//B = B / A;
		//C = C / A;
		//A = A / A;

		////绘制直线
		//Point2d ptStart, ptEnd;
		//ptStart.x = 0;
		//ptStart.y = -(A*ptStart.x + C) / B;
		//ptEnd.x = -(B*ptEnd.y + C) / A;
		//ptEnd.y = 0;
		//line(binaryImage, ptStart, ptEnd, Scalar(0, 255, 255));
		//cout << "A:" << A << " " << "B:" << B << " " << "C:" << C << " " << endl;
		//imshow("line fitting", binaryImage);
#pragma endregion ransac
		imshow("src", image);
		imshow("minAreaRect", binaryImage);


		////在白色图像上绘制黑色轮廓  
		//Mat result_erase(binaryImage.size(), CV_8U, Scalar(255));
		//drawContours(result_erase, contours,
		//	-1, //绘制所有轮廓  
		//	Scalar(0),  //颜色为黑色  
		//	2); //轮廓线的绘制宽度为2  

		//		//namedWindow("contours_erase");  
		//		//imshow("contours_erase", result_erase);  

		//		//测试包围盒  
		//Rect r0 = boundingRect(Mat(contours[0]));
		//rectangle(result_erase, r0, Scalar(128), 2);
		//Rect r1 = boundingRect(Mat(contours[1]));
		//rectangle(result_erase, r1, Scalar(128), 2);

		////测试最小包围圆  
		//float radius;
		//Point2f center;
		////minEnclosingCircle(Mat(contours[2]), center, radius);
		////circle(result_erase, Point(center), static_cast<int>(radius), Scalar(128), 2);

		////测试多边形近似  
		//vector <Point> poly;
		//approxPolyDP(Mat(contours[3]),
		//	poly,
		//	5,  //近似的精确度  
		//	true);  //这是个闭合形状  

		//			//遍历每个片段进行绘制  
		//vector<Point>::const_iterator itp = poly.begin();
		//while (itp != (poly.end() - 1))
		//{
		//	line(result_erase, *itp, *(itp + 1), Scalar(128), 2);
		//	++itp;
		//}

		////首尾用直线相连  
		//line(result_erase, *(poly.begin()), *(poly.end() - 1), Scalar(128), 2);

		////凸包是另一种多边形近似,计算凸包  
		//vector <Point> hull;
		//convexHull(Mat(contours[4]), hull);

		//vector<Point>::const_iterator ith = hull.begin();
		//while (ith != (hull.end() - 1))
		//{
		//	line(result_erase, *ith, *(ith + 1), Scalar(128), 2);
		//	++ith;
		//}
		//line(result_erase, *(hull.begin()), *(hull.end() - 1), Scalar(128), 2);

		////另一种强大的描述符力矩  
		////测试力矩  
		////遍历所有轮廓  
		//itc = contours.begin();
		//while (itc != contours.end())
		//{
		//	//计算所有的力矩  
		//	Moments mom = moments(Mat(*itc++));
		//	//绘制质心  
		//	circle(result_erase,
		//		Point(mom.m10 / mom.m00, mom.m01 / mom.m00),    //质心坐标转换为整数  
		//		2,
		//		Scalar(0),
		//		2); //绘制黑点  
		//}

		//namedWindow("contours_erase");
		//imshow("contours_erase", result_erase);
		waitKey(0);
	}
}
