#include "global.h"

///////////////////////参数设置///////////////////////////

CvSize winSize = cvSize(64, 64);	//等于训练样本图像大小
CvSize blockSize = cvSize(16, 16);	//block size
CvSize blockStride = cvSize(8, 8);	//block stride
CvSize winStride = cvSize(8, 8);	//window stride
CvSize cellSize = cvSize(8, 8);		//cell size
int nbins = 9;	//一般取9个梯度方向

////////////函数定义//////////////////
int CalDimension(CvSize winSize, CvSize blockSize, CvSize blockStride,	CvSize cellSize, int nbins)
{
	//一个窗口(winSize)内宽和高方向分别有多少个块(blockSize)
	//int hBlockNum = (winSize.height - 1) / cellSize.height;
	//int wBlockNum = (winSize.width - 1) / cellSize.width;
	int hBlockNum = (winSize.height - blockSize.height) / blockStride.height + 1;
	int wBlockNum = (winSize.width - blockSize.width) / blockStride.width + 1;

	//一个块(blockSize)里面有多少个单元(cellSize)
	int hCellNum = blockSize.height / cellSize.height;
	int wCellNum = blockSize.width / cellSize.width;

	//一个单元(cellSize)里面有多少HOG特征维度
	int hogNum = nbins;

	//计算一个窗口的HOG特征维度：block的个数 * block内部cell的个数 * 每个cell的HOG特征维度
	int totalHogNum = (hBlockNum * wBlockNum) * (hCellNum * wCellNum) * hogNum;

	return totalHogNum;
}

bool processImage(const cv::Mat &image,std::vector<cv::Mat>& outPut)
{
	if(image.empty()) return false;

    outPut.clear();
	for(int i = 0;i<8;i++)
	{
		outPut.push_back(cv::Mat::zeros(image.rows,image.cols,CV_8UC1));
	}

	outPut[0]=image;
	cv::transpose(image,outPut[7]);
	cv::flip(image, outPut[6], 0);  
	cv::flip(outPut[7], outPut[3], 0);  
	cv::flip(outPut[7], outPut[1], 1);  

	cv::flip(outPut[3], outPut[5], 1);  
	cv::flip(outPut[6], outPut[2], 1);  
	cv::flip(outPut[2], outPut[4], 0); 
	
	return true;
}

bool Train(const char* positivePath, int pCount, const char* negativePath, int nCount, 
		   const char* classifierSavePath, const char* detectorSavePath,double p_C)
{
	cout<<"******************** Train ********************"<<endl;

	int scale_image_ = 8;

	//首先计算图像的HOG特征维度
	int dim = CalDimension(winSize, blockSize, blockStride, cellSize, nbins);
	int totalCount = scale_image_*(pCount + nCount);

	cout<<"1: Start trainning for SVM:"<<endl;
	cout<<"total samples: "<<totalCount<<endl;
	cout<<"positive samples: "<<scale_image_*pCount<<endl;
	cout<<"negative samples: "<<scale_image_*nCount<<endl;
	cout<<"feature dimension is: "<<dim<<endl<<endl;

	//训练正样本
	cout<<"2: Start to train positive samples:"<<endl;

	CvMat *sampleFeaturesMat = cvCreateMat(totalCount , dim, CV_32FC1);
	//64*128的训练样本，该矩阵将是totalSample*3780
	//64*64的训练样本，该矩阵将是totalSample*1764
	cvSetZero(sampleFeaturesMat);  
	CvMat *sampleLabelMat = cvCreateMat(totalCount, 1, CV_32FC1);//样本标识  
	cvSetZero(sampleLabelMat);

	char positiveImgPath[256];
	for (int i = 0; i < pCount; i++)
	{
		//载入图像
		sprintf(positiveImgPath, "%s%d.jpg", positivePath, i + 1);
		string strPosPath(positiveImgPath);

		cv::Mat origin_img = cv::imread(strPosPath,0);
		if (origin_img.data == NULL)
		{

			sprintf(positiveImgPath, "%s%d.png", positivePath, i + 1);

			string strPosPath1(positiveImgPath);

			origin_img = cv::imread(strPosPath1, 0);

			if (origin_img.data == NULL)
			{
				cout << "positive image sample load error: " << strPosPath1 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
		std::cout << "succeed: " << i << std::endl;

		cv::Mat img;
		cv::resize(origin_img, img, winSize, cv::INTER_LINEAR);

		std::vector<cv::Mat> serial_img;
		if (!processImage(img, serial_img))
		{
			std::cout << "multipy image failed: rotate and flip origin image to serial images! " << std::endl;
			continue;
		}

		for (int k = 0; k < scale_image_; k++)
		{

			 //cv::imshow("a", serial_img[k]);
			 //cv::waitKey(0);

			cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
			vector<float> featureVec;

			hog.compute(serial_img[k], featureVec);//, winStride); //计算HOG特征向量
			int featureVecSize = featureVec.size();

			//加上类标，转化为CvMat
			for (int j = 0; j < featureVecSize; j++)
			{
				CV_MAT_ELEM(*sampleFeaturesMat, float, scale_image_ *i + k, j) = featureVec[j];
			}
			sampleLabelMat->data.fl[scale_image_ * i + k] = 1;
		}
	}
	cout << "End of training for positive samples." << endl
		 << endl;

	//训练负样本
	cout << "3: Start to train negative samples: " << endl;
	char negativeImgPath[256];
	for (int i = 0; i < nCount; i++)
	{
		//载入图像
		sprintf(negativeImgPath, "%s%d.jpg", negativePath, i + 1);

		string strNegPath(negativeImgPath);

		cv::Mat origin_img = cv::imread(strNegPath,0);
		if (origin_img.data == NULL)
		{

			sprintf(negativeImgPath, "%s%d.png", negativePath, i + 1);

			string strNegPath1(negativeImgPath);

			origin_img = cv::imread(strNegPath1, 0);

			if (origin_img.data == NULL)
			{
				cout << "negative image sample load error: " << strNegPath1 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
		std::cout << "succeed: " << i << std::endl;

		cv::Mat img;
		cv::resize(origin_img, img, winSize, cv::INTER_LINEAR);

		std::vector<cv::Mat> serial_img;
		if (!processImage(img, serial_img))
		{
			std::cout << "multipy image failed: rotate and flip origin image to serial images! " << std::endl;
			continue;
		}

		for (int k = 0; k < scale_image_; k++)
		{

			// cv::imshow("a", serial_img[k]);
			// cv::waitKey(3);

			cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
			vector<float> featureVec;

			hog.compute(serial_img[k], featureVec);//, winStride); //计算HOG特征向量
			int featureVecSize = featureVec.size();

			//加上类标，转化为CvMat
			for (int j = 0; j < featureVecSize; j++)
			{
				CV_MAT_ELEM(*sampleFeaturesMat, float, scale_image_ *(pCount + i) + k, j) = featureVec[j];
			}
			sampleLabelMat->data.fl[scale_image_ * (pCount + i) + k] = -1;
		}
	}

	cout<<"End of training for negative samples."<<endl;

	//SVM训练
	cout<<"4: Start to train SVM classifier: "<<endl;


	//设置SVM参数
	CvSVMParams params;
	int iteration = 1000;
	double penaltyFactor = p_C;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
//	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, iteration, FLT_EPSILON);
  params.C = penaltyFactor;
//	params.gamma = 0.9;
	
	//print
	cout<<"svm_type: C_SVC\nkernel_type: \ntermination type: CV_TERMCRIT_ITER"
		<<"\ntermination iteration: "<<iteration<<"\ntermination epsilon: "<<FLT_EPSILON
		<<"\npenalty factor: "<<penaltyFactor<<endl;

	MySVM svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, NULL, NULL, params ); //用线性SVM分类器训练
	svm.save(classifierSavePath);		//将SVM训练完的数据保存到指定的文件中



	/*
	 MySVM svm;
	 CvParamGrid CvParamGrid_C(pow(2.0, -3), pow(2.0, 12), pow(2.0, 1));
	 CvParamGrid CvParamGrid_gamma(pow(2.0, -5), pow(2.0, 9), pow(2.0, 1));
	 if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
	 	cout << "The grid is NOT VALID." << endl;
	 CvSVMParams paramz;
	 paramz.kernel_type = CvSVM::RBF;
	 paramz.svm_type = CvSVM::C_SVC;
	 paramz.C = 1;  //给参数赋初始值
	 //paramz.p = 5e-3;  //给参数赋初始值
	 paramz.gamma = 0.01;  //给参数赋初始值
	 paramz.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);
	
	 CvParamGrid gammaGrid = CvParamGrid(1,1,0.0);
	 CvParamGrid pGrid = CvParamGrid(1,1,0.0);
	 CvParamGrid nuGrid = CvParamGrid(1,1,0.0);
	 CvParamGrid coeffGrid = CvParamGrid(1,1,0.0);
	 CvParamGrid degreeGrid = CvParamGrid(1,1,0.0);
	 
	 svm.train_auto(sampleFeaturesMat, sampleLabelMat, Mat(), Mat(), paramz, 10,
	 			   CvParamGrid_C, CvParamGrid_gamma, pGrid, nuGrid, coeffGrid, degreeGrid, true);

	 paramz = svm.get_params();
	 svm.save(classifierSavePath); //将SVM训练完的数据保存到指定的文件中
	 float C = paramz.C;
   float P = paramz.p;
	 float gamma = paramz.gamma;
	 printf("\nParms: C = %f, P = %f,gamma = %f \n", C, P, gamma);
*/


	cvReleaseMat(&sampleFeaturesMat);
	cvReleaseMat(&sampleLabelMat);

	int supportVectorSize = svm.get_support_vector_count();
	cout<<"\nsupport vector size of SVM："<<supportVectorSize<<endl;
	cout<<"End of training SVM classifier."<<endl;

	//保存用于检测的HOG特征
	cout<<"5. Save SVM detector file: "<<endl;
	CvMat *sv,*alp,*re;//所有样本特征向量 
	sv  = cvCreateMat(supportVectorSize , dim, CV_32FC1);
	alp = cvCreateMat(1 , supportVectorSize, CV_32FC1);
	re  = cvCreateMat(1 , dim, CV_32FC1);
	CvMat *res  = cvCreateMat(1 , 1, CV_32FC1);

	cvSetZero(sv);
	cvSetZero(re);

	for(int i=0; i<supportVectorSize; i++)
	{
		memcpy( (sv->data.fl+i*dim*sizeof(float)), svm.get_support_vector(i), dim*sizeof(float));
	}

	double* alphaArr = svm.get_alpha();
	int alphaCount = svm.get_alpha_count();

	for(int i=0; i<supportVectorSize; i++)
	{
			alp->data.fl[i] = alphaArr[i];
	}
	
	cvMatMul(alp,sv,re);

	int posCount = 0;
	for (int i=0; i<dim; i++)
	{
		re->data.fl[i] *= -1;
	}

	//保存为文本文件
	FILE* fp = fopen(detectorSavePath,"wb");
	if( NULL == fp )
	{
		return false;
	}
	for(int i=0; i<dim; i++)
	{
		fprintf(fp,"%f \n",re->data.fl[i]);
	}
	float rho = svm.get_rho();
	fprintf(fp, "%f", rho);
	fclose(fp);
	cout<<"Save "<<detectorSavePath<<" OK!"<<endl;

	return true;
}

//使用detectMultiScale检测
bool DetectMulti_for_nonlinear(const char* detectorSavePath,const char* positivePath,const char* negativePath,int n_begin_test_pos,int pCount,int n_begin_test_neg,int nCount)
{
	cout<<"\n******************** Detection Multi********************"<<endl;

	vector<float> x;
	ifstream fileIn(detectorSavePath, ios::in);
	if(!fileIn.is_open())
	{
			std::cout<<"fail to load the svm txt file..."<<std::endl;
			return false;
	}

	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	CvSize winstride1 =  cvSize(4,4);
  std::vector<cv::Point>  found;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(x);
	double hitThread = 1.0;
  double scale0 = 1.05;
  double finalThreshold = 2;

	double timeSum = 0.0;

	cv::Size re_size(64,64);
  cv::Mat origin_img,img;
	int frameCount = 0;
	int pos_count_succeed = 0;

	char positiveImgPath[256];
	for (int i = n_begin_test_pos; i < pCount; i++)
	{
		//载入图像
		sprintf(positiveImgPath, "%s%d.jpg", positivePath, i + 0);
		string strPosPath(positiveImgPath);

		cv::Mat origin_img = cv::imread(strPosPath,-1);
		if (origin_img.data == NULL)
		{

			sprintf(positiveImgPath, "%s%d.png", positivePath, i + 0);

			string strPosPath0(positiveImgPath);

			origin_img = cv::imread(strPosPath0, 0);

			if (origin_img.data == NULL)
			{
				cout << "positive image sample load error: " << strPosPath0 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
	
		frameCount++;
	
    //cv::resize(origin_img, img, cv::Size(),ratio,ratio, cv::INTER_LINEAR);
		cv::resize(origin_img,img,re_size,0,0,cv::INTER_LINEAR);
	
		double begin = clock();
		//hog.detectMultiScale(img, found,hitThread, winstride1, cv::Size(1,1), scale0, finalThreshold);
		hog.detect(img, found, 0);

		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*999;
		timeSum += diff;
		cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > -1)
		{
			std::cout<<"succeed:"<<pos_count_succeed++<<std::endl;
		//	for (int i=0; i<found.size(); i++)
		//	{
		//			cv::Rect tempRect(found[i].x, found[i].y, winSize.width, winSize.height);

		//	  	cv::rectangle(img, tempRect,CV_RGB(255,0,0), 2);

		//	}

		}
		else
		{
			std::cout<<"failed:"<<std::endl;
		}

		cv::imshow("img", img);
  	if (cvWaitKey(2) == 27)
		{
			break;
		}
	}

	std::cout<<"test pos image: succeed: "<<pos_count_succeed<<std::endl;
	std::cout<<(double)pos_count_succeed/(double)(pCount-n_begin_test_pos)<<std::endl;

	int neg_count_succeed = 0;
	char negativeImgPath[256];
	for (int i = n_begin_test_neg; i < nCount; i++)
	{
		//载入图像
		sprintf(negativeImgPath, "%s%d.jpg", negativePath, i + 1);

		string strNegPath(negativeImgPath);

		cv::Mat origin_img = cv::imread(strNegPath,0);
		if (origin_img.data == NULL)
		{

			sprintf(negativeImgPath, "%s%d.png", negativePath, i + 1);

			string strNegPath1(negativeImgPath);

			origin_img = cv::imread(strNegPath1, 0);

			if (origin_img.data == NULL)
			{
				cout << "negative image sample load error: " << strNegPath1 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
	
		frameCount++;

		//cv::resize(origin_img, img, cv::Size(),ratio,ratio, cv::INTER_LINEAR);
	  cv::resize(origin_img,img,re_size,0,0,cv::INTER_LINEAR);

    double begin = clock();
	  //hog.detectMultiScale(img, found,hitThread, winstride1, cv::Size(1,1), scale0, finalThreshold);
  	hog.detect(img, found, 0);

		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*999;
		timeSum += diff;
		cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > -1)
		{
			//	for (int i=-1; i<found.size(); i++)
			//	{
		   // 		cv::rectangle(img, found[i],CV_RGB(254,0,0), 2);
			//	}
			//	std::cout<<"failed:"<<std::endl;

		}
		else
		{
				std::cout<<"succeed:"<<neg_count_succeed++<<std::endl;
		}

		cv::imshow("img", img);
		if (cvWaitKey(2) == 27)
		{
			break;
		}
	}

	std::cout<<"test neg image: succeed: "<<neg_count_succeed<<std::endl;
	std::cout<<(double)neg_count_succeed/(double)(nCount-n_begin_test_neg)<<std::endl;

  std::cout<< "Average detection time is: "<<timeSum / frameCount<<"ms"<<std::endl;
	return true;
}



//使用detectMultiScale检测
bool DetectMulti_for_linear(const char* detectorSavePath,const char* positivePath,const char* negativePath,int n_begin_test_pos,int pCount,int n_begin_test_neg,int nCount)
{
	cout<<"\n******************** Detection Multi********************"<<endl;

	vector<float> x;
	ifstream fileIn(detectorSavePath, ios::in);
	if(!fileIn.is_open())
	{
			std::cout<<"fail to load the svm txt file..."<<std::endl;
			return false;
	}

	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();


	CvSize winstride1 =  cvSize(4,4);
  std::vector<cv::Point>  found;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(x);
	double hitThreshold = 0.01;
  double scale0 = 1.05;
  double finalThreshold = 2;

	double timeSum = 0.0;

	cv::Size re_size(64,64);
  cv::Mat origin_img,img;
	int frameCount = 0;
	int pos_count_succeed = 0;

	char positiveImgPath[256];
	for (int i = n_begin_test_pos; i < pCount; i++)
	{
		//载入图像
		sprintf(positiveImgPath, "%s%d.jpg", positivePath, i + 0);
		string strPosPath(positiveImgPath);

		cv::Mat origin_img = cv::imread(strPosPath,0);
		if (origin_img.data == NULL)
		{

			sprintf(positiveImgPath, "%s%d.png", positivePath, i + 0);

			string strPosPath0(positiveImgPath);

			origin_img = cv::imread(strPosPath0, 0);

			if (origin_img.data == NULL)
			{
				cout << "positive image sample load error: " << strPosPath0 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
	
		frameCount++;
	
    //cv::resize(origin_img, img, cv::Size(),ratio,ratio, cv::INTER_LINEAR);
		cv::resize(origin_img,img,re_size,0,0,cv::INTER_LINEAR);
	
		double begin = clock();
		//hog.detectMultiScale(img, found,hitThread, winstride1, cv::Size(1,1), scale0, finalThreshold);
		hog.detect(img, found, hitThreshold);

		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*999;
		timeSum += diff;
		cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > 0)
		{
			std::cout<<"succeed:"<<pos_count_succeed++<<std::endl;
		//	for (int i=0; i<found.size(); i++)
		//	{
		//			cv::Rect tempRect(found[i].x, found[i].y, winSize.width, winSize.height);

		//	  	cv::rectangle(img, tempRect,CV_RGB(255,0,0), 2);

		//	}

		}
		else
		{
			std::cout<<"failed:"<<std::endl;
		}

		cv::imshow("img", img);
  	if (cvWaitKey(2) == 27)
		{
			break;
		}
	}

	int neg_count_succeed = 0;
	char negativeImgPath[256];
	for (int i = n_begin_test_neg; i < nCount; i++)
	{
		//载入图像
		sprintf(negativeImgPath, "%s%d.jpg", negativePath, i + 1);

		string strNegPath(negativeImgPath);

		cv::Mat origin_img = cv::imread(strNegPath,0);
		if (origin_img.data == NULL)
		{

			sprintf(negativeImgPath, "%s%d.png", negativePath, i + 1);

			string strNegPath1(negativeImgPath);

			origin_img = cv::imread(strNegPath1, 0);

			if (origin_img.data == NULL)
			{
				cout << "negative image sample load error: " << strNegPath1 << endl;
				//return false;
				//system("pause");
				continue;
			}
		}
	
		frameCount++;

		//cv::resize(origin_img, img, cv::Size(),ratio,ratio, cv::INTER_LINEAR);
	  cv::resize(origin_img,img,re_size,0,0,cv::INTER_LINEAR);

    double begin = clock();
	  //hog.detectMultiScale(img, found,hitThread, winstride1, cv::Size(1,1), scale0, finalThreshold);
  	hog.detect(img, found, hitThreshold);

		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*999;
		timeSum += diff;
		cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > 0)
		{
			//	for (int i=-1; i<found.size(); i++)
			//	{
		   // 		cv::rectangle(img, found[i],CV_RGB(254,0,0), 2);
			//	}
			//	std::cout<<"failed:"<<std::endl;

		}
		else
		{
				std::cout<<"succeed:"<<neg_count_succeed++<<std::endl;
		}

		cv::imshow("img", img);
		if (cvWaitKey(2) == 27)
		{
			break;
		}
	}

	std::cout<<"test pos image: succeed: "<<pos_count_succeed<<std::endl;
	std::cout<<(double)pos_count_succeed/(double)(pCount-n_begin_test_pos)<<std::endl;

	std::cout<<"test neg image: succeed: "<<neg_count_succeed<<std::endl;
	std::cout<<(double)neg_count_succeed/(double)(nCount-n_begin_test_neg)<<std::endl;
	  std::cout<< "Average detection time is: "<<timeSum / frameCount<<"ms"<<std::endl;
	return true;
}


//使用detectMultiScale检测
bool DetectMulti(const char* detectorSavePath, const char* testPath, const float hit_thresh)
{
	cout<<"\n******************** Detection Multi********************"<<endl;

	CvCapture* cap = cvCreateFileCapture(testPath);
	if (!cap)
	{
		cout<<"avi file load error..."<<endl;
		return false;
	}

	vector<float> x;
	ifstream fileIn(detectorSavePath, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	vector<cv::Rect>  found;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(x);
	
	IplImage* org_img = NULL;
	IplImage* img = NULL;
	cvNamedWindow("img", 0);
	//cvNamedWindow("video", 0);

	int frameCount = 0;
	double timeSum = 0.0;
	while(org_img=cvQueryFrame(cap))
	{

			frameCount++;

			if(frameCount<1000)
			{
				//	continue;
			}
		
		float scale = 1.0;
		//float scale = 0.5;
		CvSize sz;  
        sz.width = org_img->width*scale;  
        sz.height = org_img->height*scale;  
        img = cvCreateImage(sz,org_img->depth,org_img->nChannels);  
		cvResize(org_img,img,CV_INTER_CUBIC); 
				
		//cvShowImage("video", img);
	
		double begin = clock();
		hog.detectMultiScale(img, found, hit_thresh, winStride, cv::Size(4, 4), 1.05, 2);
		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*1000;
		timeSum += diff;
		//cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > 0)
		{
			//std::cout<<"succeed:"<<std::endl;
			for (int i=0; i<found.size(); i++)
			{
				CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);

				cvRectangle(img, cvPoint(tempRect.x,tempRect.y),
					cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);
			}
		}
		else
		{
			//std::cout<<"failed:"<<std::endl;
		}
		cvShowImage("img", img);
		if (cvWaitKey(3) == 27)
		{
			break;
		}
	}
	cvReleaseCapture(&cap);

	cout<< "Average detection time is: "<<timeSum / frameCount<<"ms"<<endl;
	return true;
}


bool DetectMulti_pic(const char* detectorSavePath, const char* testPath,int num_pic,double hit_threshld)
{
	cout<<"\n******************** Detection Multi********************"<<endl;
	
	vector<float> x;
	ifstream fileIn(detectorSavePath, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	vector<cv::Rect>  found;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(x);

	namedWindow("img", 0);
	namedWindow("video", 0);

	float ratio = 0.5;
	int count_succeed = 0;
	for(int i=0;i<num_pic;i++)
	{
			char testImgPath[256];
	
			//载入图像
			sprintf(testImgPath, "%s%d.jpg", testPath, i);

			string testImgPath_str(testImgPath);
			
			cv::Mat origin_img1 = cv::imread(testImgPath_str, 0);
			if (origin_img1.data == NULL)
			{
					cout << "test image sample load error: " << testImgPath_str << endl;
					return false;
			}
			cv::Mat origin_img;
			cv::resize(origin_img1,origin_img,cv::Size(),ratio,ratio);

			int frameCount = 0;
			double timeSum = 0.0;

			cv::imshow("video", origin_img);
			frameCount++;
			
			double begin = clock();
			hog.detectMultiScale(origin_img, found, hit_threshld, winStride, cv::Size(0, 0), 1.05, 2);
	
			if (found.size() > 0)
			{
					for (int i = 0; i < found.size(); i++)
					{
							CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
							
							cv::rectangle(origin_img, cv::Point(tempRect.x, tempRect.y),
													cv::Point(tempRect.x + tempRect.width, tempRect.y + tempRect.height),
													CV_RGB(255, 0, 0), 2);
					}
				 std::cout<<"succeed to detect head: "<<count_succeed++<<std::endl;

			}
			else
			{
					std::cout<<"fail to detect head..."<<std::endl;
			}
		
			cv::imshow("img", origin_img);
			cvWaitKey(0);
	}
	std::cout<<"Total : "<<num_pic<<std::endl;

	std::cout<<"Total succeed : "<<count_succeed<<std::endl;

	std::cout<<"Succeed ratio: "<<(double)count_succeed/num_pic<<std::endl;
	
	return true;
}


//使用detect检测
bool DetectSingle(const char* detectorSavePath, const char* testPath)
{
	cout<<"\n******************** Detection Single********************"<<endl;

	CvCapture* cap = cvCreateFileCapture(testPath);
	if (!cap)
	{
		cout<<"avi file load error..."<<endl;
		return false;
	}

	vector<float> x;
	ifstream fileIn(detectorSavePath, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();

	vector<cv::Point>  found;
	cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	hog.setSVMDetector(x);

	IplImage* img = NULL;
	cvNamedWindow("img", 0);
	cvNamedWindow("video", 0);

	int frameCount = 0;
	double timeSum = 0.0;
	while(img=cvQueryFrame(cap))
	{
		cvShowImage("video", img);
		frameCount++;

		double begin = clock();
		//检测：found为检测目标的左上角坐标点
		hog.detect(img, found, 0, winStride, cvSize(0,0));
		double end = clock();
		double diff = (end-begin)/CLOCKS_PER_SEC*1000;
		timeSum += diff;
		cout<< "Detection time is: "<<diff<<"ms"<<endl;

		if (found.size() > 0)
		{
			for (int i=0; i<found.size(); i++)
			{
				CvRect tempRect = cvRect(found[i].x, found[i].y, winSize.width, winSize.height);

				cvRectangle(img, cvPoint(tempRect.x,tempRect.y),
					cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);

			}
		}
		cvShowImage("img", img);
		if (cvWaitKey(1) == 27)
		{
			break;
		}
	}
	cvReleaseCapture(&cap);

	cout<< "Average detection time is: "<<timeSum / frameCount<<"ms"<<endl;
	return true;
}
