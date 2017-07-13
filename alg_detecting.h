#pragma once
#include <thread>
#include "global.h"

class DetectHead
{
public:
    enum {SUCCEED,ALG_ALREAD_START}};
    DetectHead();
    ~DetectHead(){};

    bool Start();
    bool Stop();
    static DetectHead* GetInstance();

private:
    static DetectHead* pDetectHead;
    static std::thread* pThread;    

//SVM分类器文件
const char* classifierSavePath = "./HOG_SVM.xml";
//HOG检测器文件
const char* detectorSavePath = "./HogDetector.txt";
//正负样本存储路径
const char* positivePath = "./pos_64_64/";
const char* negativePath = "./neg_64_64/";
//正负样本数目
const int pCount = 798;
const int nCount = 639;

//测试视频文件路径
const char* testVideoPath = "./test.avi";


}