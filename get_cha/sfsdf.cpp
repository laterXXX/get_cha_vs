/*
    利用ROI将一幅图像叠加到另一幅图像的指定位置
*/
 
#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>   
#include <iostream>  
using namespace std;
using namespace cv;
 
int main324324()
{
    //【1】读入两幅图像并检查图像是否读取成功  
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