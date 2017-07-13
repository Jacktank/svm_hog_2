#include "global.h"

//SVM分类器文件
const char* classifierSavePath = "./HOG_SVM.xml";
//HOG检测器文件
const char* detectorSavePath = "./HogDetector.txt";
//正负样本存储路径
const char* positivePath = "./pos_64_64/";
const char* negativePath = "./neg_64_64/";
//正负样本数目
const int pCount = 1777;
const int nCount = 2309;

//测试视频文件路径
const char* testVideoPath = "../record_/test2.avi";
//const char* testVideoPath = "rtsp://admin:12345goccia@10.0.0.3:554//Streaming/Channels/1";

int main(int argc, char* argv[])
{
	bool flag;
	if (argc != 4)
		return -1;

	bool retrain_flag = false;
	if(atoi(argv[1]) == 1)
	{
		retrain_flag = true;
	}

	double penalty_C = 0.0,hit_threshld = 0.0;
	sscanf( argv[2],"%lf",&penalty_C);
	sscanf( argv[3],"%lf",&hit_threshld);
	
	if (retrain_flag)
	{
		////////////////训练////////////////
		flag = Train(positivePath, pCount, negativePath, nCount, classifierSavePath, detectorSavePath,
								penalty_C);

		if (!flag)
		{
			cout << "Train error!\n";
			return -1;
		}
	}

	////////////////检测-单尺度///////////////
	// flag = DetectSingle(detectorSavePath, testVideoPath);
	// if (!flag)
	// {
	// 	cout<<"Detection error!\n";
	// 	return -1;
	// }

	////////////////检测-多尺度///////////////
  //flag = DetectMulti_for_linear(detectorSavePath, positivePath , negativePath,  pCount/2,pCount,nCount/2,nCount);
  //flag = DetectMulti_pic(detectorSavePath, "./picture/",81,hit_threshld);
  flag = DetectMulti(detectorSavePath, testVideoPath,hit_threshld);
	if (!flag)
	{
		cout<<"Detection error!\n";
		return -1;
	}

	// flag = DetectMulti_pic(detectorSavePath, positivePath);
	// if (!flag)
	// {
	// 	cout<<"Detection error!\n";
	// 	return -1;
	// }

	system("pause");
	return 0;
}

