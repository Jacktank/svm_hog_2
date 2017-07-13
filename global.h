#include "MySVM.h"

#include <fstream>
#include <iostream>
#include <ctime>
using namespace std;

//函数名：Train
//函数功能：SVM训练每张图片的HOG特征
//参数说明：
//const char* positivePath：正样本路径
//int pCount：正样本个数
//const char* negativePath：负样本路径
//int nCount：负样本个数
//const char* classifierSavePath：分类器保存路径
//const char* detectorSavePath：检测器保存路径
//返回bool：训练是否成功（true：成功，false：失败）
bool Train(const char* positivePath, int pCount, const char* negativePath, int nCount, 
		   const char* classifierSavePath, const char* detectorSavePath,double p_C=0.018);

//函数名：CalDimension
//函数功能：计算每张图片的HOG特征维度
//参数说明：
//CvSize winSize：窗口大小
//CvSize blockSize：块大小
//CvSize blockStride：块位移大小
//CvSize cellSize：胞元大小
//int nbins：bin数
//返回int：HOG特征维度
//参考计算方式详细：http://blog.csdn.net/carson2005/article/details/7782726
//参考参数说明详细：http://blog.csdn.net/raodotcong/article/details/6239431
int CalDimension(CvSize winSize, CvSize blockSize, CvSize blockStride,	CvSize cellSize, int nbins);

//函数名：DetectMulti
//函数功能：用SVM+HOG分类器对图片做多尺度检测
//参数说明：
//const char* detectorSavePath：检测器保存路径
//const char* testPath：测试视频路径
//返回bool：检测是否成功（true：成功，false：失败）
//bool DetectMulti(const char* detectorSavePath, const char* testPath);
bool DetectMulti(const char* detectorSavePath, const char* testPath, const float hit_thresh);

bool DetectMulti_for_linear(const char* detectorSavePath,const char* positivePath,const char* negativePath,int n_begin_test_pos,int pCount,int n_begin_test_neg,int nCount);

bool DetectMulti_for_nonlinear(const char* detectorSavePath,const char* positivePath,const char* negativePath,int n_begin_test_pos,int pCount,int n_begin_test_neg,int nCount);

bool DetectMulti_pic(const char* detectorSavePath, const char* testPath,int num_pic,double hit_threshld);

//函数名：DetectSingle
//函数功能：用SVM+HOG分类器对图片做单尺度检测
//参数说明：
//const char* detectorSavePath：检测器保存路径
//const char* testPath：测试视频路径
//返回bool：检测是否成功（true：成功，false：失败）
bool DetectSingle(const char* classifierSavePath, const char* testPath);
