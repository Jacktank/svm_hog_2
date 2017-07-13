#ifndef  _MYSVM_H_
#define _MYSVM_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

class MySVM: public CvSVM
{
public:
	int get_alpha_count()
	{
		return this->sv_total;
	}

	int get_sv_dim()
	{
		return this->var_all;
	}

	int get_sv_count()
	{
		return this->decision_func->sv_count;
	}

	double* get_alpha()
	{
		return this->decision_func->alpha;
	}

	float** get_sv()
	{
		return this->sv;
	}

	float get_rho()
	{
		return this->decision_func->rho;
	}
};

#endif
