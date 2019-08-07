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

//����[0,1]֮����Ͼ��ȷֲ�����
double uniformRandom(void)
{
	return (double)rand() / (double)RAND_MAX;
}

//����[0,1]֮����ϸ�˹�ֲ�����
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

//���ݵ㼯���ֱ��ax+by+c=0��resΪ�в�
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

//�õ�ֱ���������������ֱ�߲����㼯�����ѡ2����
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

//ֱ���������������λ�ò���̫��
bool verifyComposition(const vector<Point2d> pts)
{
	cv::Point2d pt1 = pts[0];
	cv::Point2d pt2 = pts[1];
	if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
		return false;

	return true;
}

//RANSACֱ�����
void fitLineRANSAC(vector<Point2d> ptSet, double &a, double &b, double &c, vector<bool> &inlierFlag)
{
	double residual_error = 2.99; //�ڵ���ֵ

	bool stop_loop = false;
	int maximum = 0;  //����ڵ���

					  //�����ڵ��ʶ����в�
	inlierFlag = vector<bool>(ptSet.size(), false);
	vector<double> resids_(ptSet.size(), 3);
	int sample_count = 0;
	int N = 500;

	double res = 0;

	// RANSAC
	srand((unsigned int)time(NULL)); //�������������
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

		// ����ֱ�߷���
		calcLinePara(pt_sam, a, b, c, res);
		//�ڵ����
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
		// �ҵ�������ֱ��
		if (inlier_count >= maximum)
		{
			maximum = inlier_count;
			resids_ = residualstemp;
			inlierFlag = inlierstemp;
		}
		// ����RANSAC�����������Լ��ڵ����
		if (inlier_count == 0)
		{
			N = 500;
		}
		else
		{
			double epsilon = 1.0 - double(inlier_count) / (double)ptSet.size(); //Ұֵ�����
			double p = 0.99; //���������д���1���������ĸ���
			double s = 2.0;
			N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
		}
		++sample_count;
	}

	//���������ڵ��������ֱ��
	vector<Point2d> pset;
	for (unsigned int i = 0; i < ptSet.size(); i++)
	{
		if (inlierFlag[i])
			pset.push_back(ptSet[i]);
	}

	calcLinePara(pset, a, b, c, res);
}


bool comp_which_point_on_the_left(const Point &a, const Point &b)
{
	return a.x < b.x;
}

bool comp_which_RotatedRect_on_the_left(const RotatedRect &a, const RotatedRect &b)
{
	return a.center.x < b.center.x;
}

bool comp_which_Rect_on_the_left(const Rect &a, const Rect &b)
{
	return a.x < b.x;
}

double getDist_P2L(Point pointP, Point pointA, Point pointB)
{
	//��ֱ�߷���
	int A = 0, B = 0, C = 0;
	A = pointA.y - pointB.y;
	B = pointB.x - pointA.x;
	C = pointA.x*pointB.y - pointA.y*pointB.x;
	//����㵽ֱ�߾��빫ʽ
	double distance = 0;
	distance = abs(((float)abs(A*pointP.x + B * pointP.y + C)) / ((float)sqrtf(A*A + B * B)));
	return distance;
}

void MyGammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}

}

void get_best_line(Mat img, vector <vector<Point>> &contours, vector<RotatedRect> &box, vector<Rect> &boundRect, vector<Point> &p, double &phi, double &rho)
{
	double cos_theta = 0, sin_theta = 0;
	double sin;
	double x0, y0;
	double k = 0;
	double k1 = -1;
	double b;
	double x;
	double y;
	double A, B, C, dis;
	double PI = 3.1415926535898;
	cv::Vec4f lin;

	while (k != k1) {

		if (p.size() > 4) {
			vector<Vec2f> lines;

			cv::fitLine(p, lin, CV_DIST_HUBER, 0, 0.01, 0.01);
			cos_theta = lin[0];
			sin_theta = lin[1];
			x0 = lin[2], y0 = lin[3];

			phi = atan2(sin_theta, cos_theta) + PI / 2.0;
			rho = y0 * cos_theta - x0 * sin_theta;
		}
		if (cos_theta == 0)
		{
			break;
		}
		k = sin_theta / cos_theta;
		b = y0 - k * x0;
		x = 0;
		y = k * x + b;

		vector<double> dis_arrays;
		if (p.size() > 0) {
			for (int i = 0; i < contours.size(); i++) {
				dis = getDist_P2L(p[i], Point(x, y), Point(x0, y0));
				dis_arrays.push_back(dis);	// ��¼���еľ���

				if (dis >(img.rows / 4))
				{
					p.erase(p.begin() + i);
					box.erase(box.begin() + i);
					boundRect.erase(boundRect.begin() + i);
					contours.erase(contours.begin() + i);
				}
			}
		}

		// �ų�������ʣ�µ�contours�������ֱ��
		if (p.size() > 4) {
			cv::fitLine(p, lin, CV_DIST_HUBER, 0, 0.01, 0.01);
			cos_theta = lin[0];
			sin_theta = lin[1];
			x0 = lin[2], y0 = lin[3];

			phi = atan2(sin_theta, cos_theta) + PI / 2.0;
			rho = y0 * cos_theta - x0 * sin_theta;
		}
		k1 = sin_theta / cos_theta;
	}

	//drawLine(img, phi, rho, cv::Scalar(255));
}


/**
this function return contours which already have rmoved some bad contours
*/
void get_best_contours(Mat img, vector <vector<Point>> &contours, vector<RotatedRect> &box, vector<Rect> &boundRect,  vector<Point> &p, double &phi, double &rho)
{
	int cmin = 100; //��С��������  
	int cmax = 1500;    //�������  
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end())
	{
		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}

	itc = contours.begin();
	while (itc != contours.end())
	{
		double area = contourArea(*itc);
		//cout << area << endl;
		if (area< 500 || area > 20000)
			itc = contours.erase(itc);
		else
			++itc;
	}



	double max_area = 0;
	vector<double> areas;
	int index = 0;
	Mat ROI;
	vector<Mat> rois;
	/*for (int i = 0; i < contours.size(); i++)
	{
	cout << contourArea(contours[i]) << endl;
	}
	cout <<"----------------------------" << endl;*/

	Mat minAreaRect_erase(img.size(), CV_8U, Scalar(255));
	//vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	//vector<Rect> boundRect(contours.size());  //������Ӿ��μ���
	Point2f rect[4];
	//vector<Point> p;
	
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(Mat(contours[i]));
		//rectangle(image, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

		box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���

		p.push_back(box[i].center);

		box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
	}


	get_best_line(img, contours, box, boundRect, p, phi, rho);

	sort(p.begin(), p.end(), comp_which_point_on_the_left);
	sort(box.begin(), box.end(), comp_which_RotatedRect_on_the_left);
	sort(boundRect.begin(), boundRect.end(), comp_which_Rect_on_the_left);
}

void daw_minAreaRect(Mat &res, vector <vector<Point>> contours, vector<RotatedRect> box)
{
	Point2f rect1[4];

	for (int i = 0; i < contours.size(); i++)
	{
		//boundRect_canny[i] = boundingRect(Mat(contours_canny[i]));
		//rectangle(image, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

		box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���

		circle(res, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�

		box[i].points(rect1);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����

		for (int j = 0; j < 4; j++)
		{
			line(res, rect1[j], rect1[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		}
	}
}

int main() {

	#pragma region read img_file_txt
	ifstream file;
	file.open("H:/rect_license/img_names.txt");
	assert(file.is_open());
	string img_name;
	
	#pragma endregion

	while (getline(file, img_name))
	{
		cout << img_name << endl;
		Mat image = imread(img_name);
		resize(image, image, Size(800, 200));
		imshow(img_name, image);
		double phi, rho;

		#pragma region OTSU
		Mat gama;
		MyGammaCorrection(image, gama, 1.5);
			
		
		Mat grayImage;
		cvtColor(gama, grayImage, CV_BGR2GRAY);

		//medianBlur(grayImage, grayImage, 9);
		imshow("grayImage", grayImage);

		//ת��Ϊ��ֵͼ    
		Mat binaryImage;
		
		threshold(grayImage, binaryImage, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

		// ֮ǰ�����д���������д�ģ� ��Ҫ�����һ�Ŷ�ά�룬���Ƕ�ά��Ļ� ����ܶ��߶Σ�
		// ֱ���ö�ֵ��Ȼ���������ǲ����Եģ����ȱ�Ե��⣬Ȼ�������ٸ�ʴ�������õ���һ���ܺõĶ�ά������
		imshow("binaryImage", binaryImage);

		//��ֵͼ ������������ط�ת����Ϊһ��������255��ɫ��ʾǰ�������壩����0��ɫ��ʾ����    
		Mat reverseBinaryImage;
		bitwise_not(binaryImage, reverseBinaryImage);
		GaussianBlur(reverseBinaryImage, reverseBinaryImage, Size(3, 3), 0.1);
		imshow("bitwise_not", reverseBinaryImage);

		vector <vector<Point>>contours;
		findContours(reverseBinaryImage,
			contours,   //����������  
			CV_RETR_EXTERNAL,   //��ȡ������  
			CV_CHAIN_APPROX_NONE);  //��ȡÿ��������ÿ������  
									//�ڰ�ɫͼ���ϻ��ƺ�ɫ���� 
		vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
		vector<Rect> boundRect(contours.size());  //������Ӿ��μ���
		vector<Point> p;
		get_best_contours(reverseBinaryImage, contours, box, boundRect, p, phi, rho);

		Mat result(reverseBinaryImage.size(), CV_8U, Scalar(255));
		drawContours(result, contours,
			-1, //������������  
			Scalar(0),  //��ɫΪ��ɫ  
			2); //�����ߵĻ��ƿ��Ϊ2  

		namedWindow("contours");
		drawLine(result, phi, rho, cv::Scalar(0));
		imshow("contours", result);
#pragma endregion


		#pragma region canny dilate 
		if(contours.size() < 8 || contours.size() > 15){
			Mat anny;
			MyGammaCorrection(image, anny, 1.5);
			Canny(anny, anny, 100, 200);
		
			Mat element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1)); //����ṹԪ��
			dilate(anny, anny, element); //����
			imshow("anny", anny);
			vector <vector<Point>>contours_canny;
			findContours(anny,
				contours_canny,   //����������  
				CV_RETR_EXTERNAL,   //��ȡ������  
				CV_CHAIN_APPROX_NONE);  //��ȡÿ��������ÿ������  
		
			vector<RotatedRect> box_canny(contours_canny.size());
			vector<Rect> boundRect_canny(contours_canny.size());
			vector<Point> p_canny;
			get_best_contours(anny, contours_canny, box_canny, boundRect_canny, p_canny, phi, rho);
		
			Mat result1(anny.size(), CV_8U, Scalar(255));
			drawContours(result1, contours_canny,
				-1, //������������  
				Scalar(0),  //��ɫΪ��ɫ  
				2); //�����ߵĻ��ƿ��Ϊ2  

			namedWindow("contours_canny");
			imshow("contours_canny", result1);

			// �ڰ�ɫ�ı����»�����С��Ӿ���
			//Mat minAreaRect_erase1(anny.size(), CV_8U, Scalar(255));
			daw_minAreaRect(result1, contours_canny, box_canny);
			drawLine(result1, phi, rho, cv::Scalar(0));
			imshow("box_canny", result1);
		}
#pragma endregion


		// 1������ÿ��contours��������ж��Ƿ���Ȼ������	2��contours����С��8���쳣����������13���쳣





#pragma region draw_minAreaRect
		Point2f rect[4];
		//vector<vector<Point2f>> cha_rects;
		//for (int i = 0; i < box.size(); i++)
		//{
		//	//rectangle(image, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
		//	box[i].points(rect);
		//	circle(binaryImage, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�

		//	for (int j = 0; j < 4; j++)
		//	{
		//		line(binaryImage, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
		//																				 //cha_rects[i].push_back(Point2f(rect[j], rect[(j + 1) % 4]));
		//	}
		//}
#pragma endregion


#pragma region get_every_char_mask

		vector<Mat> masks;// = Mat::zeros(image.size(), CV_8UC1);
		vector<float> angle_arr;
		for (int i = 0; i < box.size(); i++)
		{

			float angle;
			//cout << "angle=" << box[i].angle << endl;
			angle = box[i].angle;
			//���÷���任������ת        ��һ�ַ�����͸�ӱ任
			if (0< abs(angle) && abs(angle) <= 45)
				angle = angle;//������˳ʱ����ת
			else if (45< abs(angle) && abs(angle)<90)
				angle = 90 - abs(angle);//��������ʱ����ת
			angle_arr.push_back(angle);

			box[i].points(rect);
			Mat mask = Mat::zeros(binaryImage.size(), CV_8UC1);

			double A[4], B[4], C[4], dis[4];

			for (int j = 0; j < 4; j++)
			{
				A[j] = rect[(j + 1) % 4].y - rect[j].y;
				B[j] = rect[j].x - rect[(j + 1) % 4].x;
				C[j] = rect[(j + 1) % 4].x * rect[j].y - rect[j].x * rect[(j + 1) % 4].y;
			}
			double w, h;
			w = (abs(A[0] * box[i].center.x + B[0] * box[i].center.y + C[0]) / sqrt(A[0] * A[0] + B[0] * B[0])) * 2;
			h = (abs(A[1] * box[i].center.x + B[1] * box[i].center.y + C[1]) / sqrt(A[1] * A[1] + B[1] * B[1])) * 2;

			for (int r = 0; r < mask.rows; r++) {
				uchar* rdata = mask.ptr<uchar>(r);
				for (int c = 0; c < mask.cols; c++)
				{
					for (int j = 0; j < 4; j++)
					{
						dis[j] = abs(A[j] * c + B[j] * r + C[j]) / sqrt(A[j] * A[j] + B[j] * B[j]);
					}

					if (dis[0] <= w && dis[1] <= h && dis[2] <= w && dis[3] <= h)
					{
						rdata[c] = 255;
					}
				}
			}

			Mat im(reverseBinaryImage.size(), CV_8UC1, Scalar(255));
			binaryImage.copyTo(im, mask);
			masks.push_back(im);
		}
#pragma endregion
		Mat dst = binaryImage.clone();
		for (int i = 0; i < box.size(); i++)
		{
			Mat tmp;
			tmp = masks[i];

			Mat image_clone = image.clone();

			box[i].points(rect);
			double x1 = rect[0].x, x2 = 0, y1 = rect[0].y, y2 = 0;
			for (int j = 0; j < 4; j++)
			{
				if (x1 > rect[j].x) {
					x1 = (rect[j].x > 0) ? rect[j].x : 0;
				}
				else if (x2 < rect[j].x) {
					x2 = (rect[j].x < tmp.cols) ? rect[j].x : tmp.cols;
				}

				if (y1 > rect[j].y) {
					y1 = (rect[j].y > 0) ? rect[j].y : 0;
				}
				else if (y2 < rect[j].y) {
					y2 = (rect[j].y < tmp.rows) ? rect[j].y : tmp.rows;
				}
			}

			Mat cha = tmp(Rect(Point2i(int(x1), int(y1)), Point2i(int(x2), int(y2))));
			if (cha.empty())
				continue;
			cv::resize(cha, cha, Size(100, 200));
			cv::imshow("cha", cha);

			Mat result1(800, 800, CV_8U, Scalar(255));
			Mat rs = result1(Rect(result1.cols / 2 - cha.cols / 2, result1.rows / 2 - cha.rows / 2, cha.cols, cha.rows));
			cha.copyTo(rs);

			int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			x0 = boundRect[i].x;
			y0 = boundRect[i].y;
			w0 = boundRect[i].width;
			h0 = boundRect[i].height;
			ROI = dst(Rect(x0, y0, w0, h0));
			Point2f center(ROI.cols, ROI.rows);  //������ת��������



			double angle0 = angle_arr[i];

			if (i > 0)
			{
				if (abs(angle0 - angle_arr[i - 1]) > 10)
				{
					angle0 = angle_arr[i - 1];
					angle_arr[i] = angle_arr[i - 1];
				}
			}

			double scale = 1;
			Mat roateM = getRotationMatrix2D(center, angle0, scale);  //�����ת����,˳ʱ��Ϊ������ʱ��Ϊ��

			cv::warpAffine(result1, result1, roateM, result1.size()); //����任
			cv::imshow("result1", result1);
			result1.release();
			
		}


		//for (int i = 0; i < box.size(); i++)
		//{
		//	Mat result1(800, 800, CV_8U, Scalar(255));
		//	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
		//	x0 = boundRect[i].x;
		//	y0 = boundRect[i].y;
		//	w0 = boundRect[i].width;
		//	h0 = boundRect[i].height;
		//	ROI = dst(Rect(x0, y0, w0, h0));
		//	
		//	Mat rr = result1(Rect(result1.cols / 2 - w0 / 2, result1.rows / 2 - h0 / 2, ROI.cols, ROI.rows));
		//	ROI.copyTo(rr);

		//	imshow("ROI", ROI);
		//	float angle;
		//	//cout << "angle=" << box[i].angle << endl;
		//	angle = box[i].angle;
		//	//���÷���任������ת        ��һ�ַ�����͸�ӱ任
		//	if (0< abs(angle) && abs(angle) <= 45)
		//		angle = angle;//������˳ʱ����ת
		//	else if (45< abs(angle) && abs(angle)<90)
		//		angle = 90 - abs(angle);//��������ʱ����ת
		//	Point2f center(ROI.cols, ROI.rows);  //������ת��������
		//	double angle0 = angle;
		//	double scale = 1;
		//	Mat roateM = getRotationMatrix2D(center, angle0, scale);  //�����ת����,˳ʱ��Ϊ������ʱ��Ϊ��

		//	warpAffine(result1, result1, roateM, result1.size()); //����任

		//														  //Mat roi = result1(Rect(0, 32, src.cols, src.rows));

		//	imshow("pg", result1);
		//	imshow("ROI1", ROI);
		//	//waitKey(0);
		//}





#pragma region ransac


		//double A, B, C;
		//vector<bool> inliers;
		//fitLineRANSAC(p, A, B, C, inliers);

		//B = B / A;
		//C = C / A;
		//A = A / A;

		////����ֱ��
		//Point2d ptStart, ptEnd;
		//ptStart.x = 0;
		//ptStart.y = -(A*ptStart.x + C) / B;
		//ptEnd.x = -(B*ptEnd.y + C) / A;
		//ptEnd.y = 0;
		//line(binaryImage, ptStart, ptEnd, Scalar(0, 255, 255));
		//cout << "A:" << A << " " << "B:" << B << " " << "C:" << C << " " << endl;
		//imshow("line fitting", binaryImage);
#pragma endregion 





		////�ڰ�ɫͼ���ϻ��ƺ�ɫ����  
		//Mat result_erase(binaryImage.size(), CV_8U, Scalar(255));
		//drawContours(result_erase, contours,
		//	-1, //������������  
		//	Scalar(0),  //��ɫΪ��ɫ  
		//	2); //�����ߵĻ��ƿ��Ϊ2  

		//		//namedWindow("contours_erase");  
		//		//imshow("contours_erase", result_erase);  

		//		//���԰�Χ��  
		//Rect r0 = boundingRect(Mat(contours[0]));
		//rectangle(result_erase, r0, Scalar(128), 2);
		//Rect r1 = boundingRect(Mat(contours[1]));
		//rectangle(result_erase, r1, Scalar(128), 2);

		////������С��ΧԲ  
		//float radius;
		//Point2f center;
		////minEnclosingCircle(Mat(contours[2]), center, radius);
		////circle(result_erase, Point(center), static_cast<int>(radius), Scalar(128), 2);

		////���Զ���ν���  
		//vector <Point> poly;
		//approxPolyDP(Mat(contours[3]),
		//	poly,
		//	5,  //���Ƶľ�ȷ��  
		//	true);  //���Ǹ��պ���״  

		//			//����ÿ��Ƭ�ν��л���  
		//vector<Point>::const_iterator itp = poly.begin();
		//while (itp != (poly.end() - 1))
		//{
		//	line(result_erase, *itp, *(itp + 1), Scalar(128), 2);
		//	++itp;
		//}

		////��β��ֱ������  
		//line(result_erase, *(poly.begin()), *(poly.end() - 1), Scalar(128), 2);

		////͹������һ�ֶ���ν���,����͹��  
		//vector <Point> hull;
		//convexHull(Mat(contours[4]), hull);

		//vector<Point>::const_iterator ith = hull.begin();
		//while (ith != (hull.end() - 1))
		//{
		//	line(result_erase, *ith, *(ith + 1), Scalar(128), 2);
		//	++ith;
		//}
		//line(result_erase, *(hull.begin()), *(hull.end() - 1), Scalar(128), 2);

		////��һ��ǿ�������������  
		////��������  
		////������������  
		//itc = contours.begin();
		//while (itc != contours.end())
		//{
		//	//�������е�����  
		//	Moments mom = moments(Mat(*itc++));
		//	//��������  
		//	circle(result_erase,
		//		Point(mom.m10 / mom.m00, mom.m01 / mom.m00),    //��������ת��Ϊ����  
		//		2,
		//		Scalar(0),
		//		2); //���ƺڵ�  
		//}

		//namedWindow("contours_erase");
		//imshow("contours_erase", result_erase);
		waitKey(0);
	}
	return 0;
}
