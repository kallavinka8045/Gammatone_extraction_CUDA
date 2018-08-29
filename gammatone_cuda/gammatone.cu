#include "gammatone.h"

////////////////////
/*핵		심*/
void GammatoneFilter(
	double **gammatone_filter,	/*출력 gammatone filter*/
	int samplingrate,			/*sampling rate*/
	double EarQ,
	double min_freq,			/*minimum freq*/ //center freq
	double max_freq,			/*maximum freq*/
	int N,						/*filter bank 수*/
	int order,
	double minBW,
	int TWIN	
	)
{
	/*
	* 공용 변수
	*/
	int i,j, check = 0;
	int length[4] = {TWIN, TWIN/2, TWIN/4, TWIN/8};
	int com_freqs[4] = {1000, 2500, 5000, 11025};
	/*
	*	gammatone
	*/
	double* cf = NULL; //center freq
	double* B = NULL;
	double* ERB = NULL;
	double tmp = 0;

	double T = 1.0 / samplingrate;
	int* size = NULL;

	double temp;
	double *energy;

	cf = (double*)malloc(sizeof(double) * N);
	B = (double*)malloc(sizeof(double) * N);
	size = (int*)malloc(sizeof(int) * N);
	energy = (double*)malloc(sizeof(double) * N);
	
	//////////////////////////////////
	//Equivalent Rectangular Bandwidth
	//////////////////////////////////

	ERB = MakeERBFilters(max_freq, min_freq, N, EarQ, minBW, order, cf);

	for(i = 0; i < N; i++)
		B[i] = B_f(ERB[i]);

	//////////////////////////////////
	//4th order Gammatone Filter
	//////////////////////////////////

	for(i = 0; i < N; i++){
		for(j = 0; j < TWIN; j++){
			if(cf[i] > com_freqs[check]){
				check++;
			}
			gammatone_filter[i][0] = length[check];
			gammatone_filter[i][j+1] = pow(j*T, 3.0) * exp(-2* PI * B[i] * T * j) * cos(2 * PI * cf[i] * T * j);
		}
	}

	////////////////////////////
	//Normalization 우진이형 버전
	double filter_sum = 0;
	for (i = 0; i < N; i++){
		filter_sum = 0;
		for (j = 0; j < (int)gammatone_filter[i][0]; j++)
			filter_sum += gammatone_filter[i][j + 1] * gammatone_filter[i][j + 1];

		for (j = 0; j < (int)gammatone_filter[i][0]; j++)
			gammatone_filter[i][j + 1] /= sqrt(filter_sum);
	}
	
	////////////////////////////

	////////////////////////////
	//Normalization 효진이형 버전
	//for (i = 0; i < N; i++)
	//{
	//	temp = 0;
	//	for (j = 1; j< gammatone_filter[i][0] + 1; j++)
	//	{
	//		temp += gammatone_filter[i][j] * gammatone_filter[i][j];
	//	}
	//	energy[i] = temp;
	//}
	//for (i = 0; i < N; i++)
	//{
	//	for (j = 1; j< gammatone_filter[i][0] + 1; j++)
	//	{
	//		gammatone_filter[i][j] = gammatone_filter[i][j] / (sqrt(energy[i] * gammatone_filter[i][0]));
	//	}
	//}

	//free(energy);
	////////////////////////////
	free(size);
	free(cf);
	free(B);
	free(ERB);
}

double* MakeERBFilters(
	int FMAX,
	int FMIN,
	int N,	
	double EarQ,
	double minBW,
	int order,
	double *cf
	){

	//////////////
	int i,j;

	double* ERB = (double *)malloc(sizeof(double) * N);

	/////1. center freq 계산
	for(i = 1; i < N+1; i++){
		cf[N-i] = -(EarQ * minBW) + exp(i * (-log(FMAX + EarQ * minBW) + log(FMIN + EarQ * minBW) ) / N) * (FMAX + EarQ * minBW);
	}
	cf[0] = FMIN;

	////2. ERB 계산
	for(i = 0; i < N; i++){
		ERB[i] = pow(pow((cf[i] / EarQ), order) + pow(minBW, order), 1.0/order);
	}

	//output
	return ERB;
}

double B_f(double freq){
	return 1.019 * freq;
}


void MaxIndex(double **in, int row_size, int col_size, int* out){
	int i,j;
	int maxindex = 0;
	double maxvalue = 0;

	maxvalue = in[0][0];

	for(i = 0; i < row_size; i++){
		for(j = 0; j < col_size; j++){
			if(maxvalue <= in[i][j]){
				maxvalue = in[i][j];
				out[0] = i;
				out[1] = j;
			}
		}
	}
}

double Cal_Gain(cufftComplex* r, double* large_pi, int size, int phase){
	int i = 0;
	double deno = 0, nume = 0;
	for(i = 0; i < size; i++){
		nume += large_pi[i] * r[i+phase].x;
		deno += large_pi[i] * large_pi[i];
	}
	return nume/deno;
}

/*
void Correlation(double* in_1, double* in_2, int size, double* coeff, int* delay, int tmp){
	double* temp = NULL;
	double* result = NULL;
	int i, j;
	double maxvalue = 0;
	int maxindex = 0;

	temp = (double *)calloc(size, sizeof(double));
	result = (double*)calloc(2*size, sizeof(double));

	for(i = 0; i < 2*size; i++){
		if(i < size){
			for(j = 0; j <= i; j++)
				temp[j] = in_2[size - 1 - i + j];
		}else{
			for(j = i-size; j < size; j++)
				temp[j] = in_2[j-(i-size)];
		}

		for(j = 0; j < size; j++){
			result[i] += temp[j] * in_1[j];
		}

		for(j = 0; j < size; j++)
			temp[j] = 0;
	}

	maxvalue = result[0];
	//max값 찾기
	for(i = 1; i < 2*size; i++){
		if(maxvalue < result[i]){
			maxvalue = result[i];
			maxindex = i;
		}
	}

	coeff[tmp] = maxvalue;
	delay[tmp] = maxindex;

	free(result);
	free(temp);
}*/