/*
    ����ROI��һ��ͼ����ӵ���һ��ͼ���ָ��λ��
*/
 
#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>   
#include <iostream>  
using namespace std;
using namespace cv;
 
int main324324()
{
    //��1����������ͼ�񲢼��ͼ���Ƿ��ȡ�ɹ�  
    Mat srcImage = imread("H:/sclead/2/0EDEX.jpg");
    Mat signal = imread("H:/sclead/LPTphotoes/idea/16.jpg",0);
	Mat M(500, 500, CV_8UC3, Scalar(255, 255, 255));
	Mat src = imread("H:/sclead/LPTphotoes/idea/16.jpg");
	Mat roi = M(Rect(0, 32, src.cols, src.rows));
	src.copyTo(roi);
	imshow("1", M);
	waitKey();

    waitKey(0);
    return 0;
}