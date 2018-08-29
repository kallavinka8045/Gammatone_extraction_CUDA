#ifndef  __GAMMATONE_H__
#define __GAMMATONE_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

typedef float2 cufftComplex;

void GammatoneFilter(
	double **gammatone_filter,	/*출력 gammatone filter*/
	int samplingrate,	/*sampling rate*/
	double EarQ,
	double min_freq,	/*minimum freq*/ //center freq
	double max_freq,	/*maximum freq*/
	int N,				/*filter bank 수*/
	int order,
	double minBW,
	int TWIN			//연산 단위
	);

double B_f(double freq);

double* MakeERBFilters(
	int FMAX,
	int FMIN,
	int N,	
	double EarQ,
	double minBW,
	int order,
	double *cf
	);

void MaxIndex(double **in, int row_size, int col_size, int* out);
double Cal_Gain(cufftComplex* r, double* large_pi, int size, int phase);
//void Correlation(double* in_1, double* in_2, int size, double* coeff, int* delay, int tmp);
#endif