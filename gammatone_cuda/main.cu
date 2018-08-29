/*
*	2016-12-02
*	Gammatone 연습
*
*	작성자 : SHIN
*/

/*
*	헤더
*/
#include "gammatone.h"
#include <cufft.h>
/*
*	define
*/
#define FRAME 20480
#define OVERLAP_ADD FRAME/2
#define NUMOFSPIKE 12000
#define NUMOFLENGTH 4

//공용
#define PI 3.14159265358979323846
#define NUMOFGENRE 3
#define TRAININGDATA 1920
#define TESTDATA 192

//FFT
//#define SQRTFRAME 16

//ERB
#define EAR_Q 9.26449
#define MIN_BW 24.7
#define NUMOFCH 256
#define MIN_FRAQ 50
#define ORDER 1
#define TWIN 512

__global__ void initialize_over(cufftComplex *dev_overlap)
{
	int tid = blockIdx.x * 1*blockDim.x + 1*threadIdx.x;
	dev_overlap[tid].x = 0;
	dev_overlap[tid].y = 0;
}

__global__ void initialize_corr(double *dev_corr_signal)
{
	int tid = blockIdx.x * 1*blockDim.x + 1*threadIdx.x;
	int n = 0;
	while (n < NUMOFCH)
	{
		dev_corr_signal[tid] = 0;
		tid += FRAME;
		n++;
	}
}

__global__ void Corr_1(
	cufftComplex *dev_re,			//device input
	cufftComplex *dev_overlap		//device overlap memory
	)
{
	//OUTPUT : dev_overlap 배열에 dev_re를 조각내서 옮겨 담음
	int tid_re = blockIdx.x * 1*blockDim.x + 1*threadIdx.x;
	int tid_over = blockIdx.x * 1*blockDim.x * 2 + 1*threadIdx.x;

	dev_overlap[tid_over].x = dev_re[tid_re].x;
}

__global__ void Corr_2(
	cufftComplex *dev_atom,			//device dictionary
	cufftComplex *dev_overlap,		//device overlap memory
	cufftComplex *dev_corr_result,
	int numofch,
	int nofchsum
	)
{
	//OUTPUT : 주파수축 상의 ATOM과 RESIDUAL을 complex conjugate multiply
	int tid = blockIdx.x * 1*blockDim.x + 1*threadIdx.x;
	int tid_ori = tid;
	int tid_atom = nofchsum*(2 * TWIN) + 1*threadIdx.x;
	int tid_corr = nofchsum*(2 * (FRAME + TWIN)) + tid;

	int n = 0;
	/* complex conjugate multiply : (a+jb)(c+jd)^* = ac+bd + j(bc-ad) */
	while (n < numofch){
		dev_corr_result[tid_corr].x = dev_overlap[tid_ori].x * dev_atom[tid_atom].x + dev_overlap[tid_ori].y * dev_atom[tid_atom].y;
		dev_corr_result[tid_corr].y = dev_overlap[tid_ori].y * dev_atom[tid_atom].x - dev_overlap[tid_ori].x * dev_atom[tid_atom].y;
		tid_corr += (2 * (FRAME + TWIN));
		tid_atom += (2 * TWIN);
		n++;
	}
}

__global__ void Corr_3(
	cufftComplex *dev_corr_result,	//device overlap memory
	double *dev_corr_signal,		//correlation 결과를 임시로 담는 배열
	int numofch,
	int nofchsum
	)
{
	//OUTPUT : 약 1초의 audio에서 같은 길이의 atom 끼리들을 전부 비교해서 dev_maxcorr가 최대일 때의 "atom종류, phase"(dev_maxindex[2])
	int tid = 2*nofchsum*(FRAME + TWIN) + ((blockIdx.x * 2) + 1) * 1*blockDim.x + 1*threadIdx.x;
	int tid_2 = 2*nofchsum*(FRAME + TWIN) + ((blockIdx.x * 2) + 2) * 1*blockDim.x + 1*threadIdx.x;
	int tid_corr = nofchsum*(FRAME) + blockIdx.x * 1*blockDim.x + 1*threadIdx.x;

	int n = 0;
	while (n < numofch){
		dev_corr_signal[tid_corr] += dev_corr_result[tid].x;
		dev_corr_signal[tid_corr] += dev_corr_result[tid_2].x;
		tid += 2*(FRAME + TWIN);
		tid_2 += 2*(FRAME + TWIN);
		tid_corr += FRAME;
		n++;
	}
}

template <unsigned int blockSize>
__global__ void FindMaxInArray(
	double *g_idata,
	double *g_odata,
	int *max_value
	)
{
	__shared__ double sdata[1024];
	__shared__ unsigned int idata[1024];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	if (fabs(g_idata[i]) < fabs(g_idata[i + blockDim.x])){
		sdata[tid] = g_idata[i + blockDim.x];
		idata[tid] = i + blockDim.x;
	}
	else{
		sdata[tid] = g_idata[i];
		idata[tid] = i;
	}
	__syncthreads();

	if (blockSize >= 1024) { if (tid < 512) {
		if (fabs(sdata[tid]) < fabs(sdata[tid + 512])){
			sdata[tid] = sdata[tid + 512];
			idata[tid] = idata[tid + 512];
		} } __syncthreads(); }
	if (blockSize >= 512) {	if (tid < 256) {
		if (fabs(sdata[tid]) < fabs(sdata[tid + 256])){
			sdata[tid] = sdata[tid + 256];
			idata[tid] = idata[tid + 256];
		} } __syncthreads(); }
	if (blockSize >= 256) {	if (tid < 128) {
		if (fabs(sdata[tid]) < fabs(sdata[tid + 128])){
			sdata[tid] = sdata[tid + 128];
			idata[tid] = idata[tid + 128];
		} } __syncthreads(); }
	if (blockSize >= 128) {	if (tid < 64) {
		if (fabs(sdata[tid]) < fabs(sdata[tid + 64])){
			sdata[tid] = sdata[tid + 64];
			idata[tid] = idata[tid + 64];
		} } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 32])){
				sdata[tid] = sdata[tid + 32];
				idata[tid] = idata[tid + 32];
			} }
		if (blockSize >= 32){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 16])){
				sdata[tid] = sdata[tid + 16];
				idata[tid] = idata[tid + 16];
			} }
		if (blockSize >= 16){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 8])){
				sdata[tid] = sdata[tid + 8];
				idata[tid] = idata[tid + 8];
			} }
		if (blockSize >= 8){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 4])){
				sdata[tid] = sdata[tid + 4];
				idata[tid] = idata[tid + 4];
			} }
		if (blockSize >= 4){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 2])){
				sdata[tid] = sdata[tid + 2];
				idata[tid] = idata[tid + 2];
			} }
		if (blockSize >= 2){
			if (fabs(sdata[tid]) < fabs(sdata[tid + 1])){
				sdata[tid] = sdata[tid + 1];
				idata[tid] = idata[tid + 1];
			} }
	}
	if (tid == 0){
		g_odata[blockIdx.x] = sdata[0];
		max_value[blockIdx.x] = idata[0];
	}
}

//void Make_feature(int*** maxindex, double* y);
void Make_feature(int**, double*, double*);

void main(int argc, char *argv[]){
	/*
	* 공용 변수
	*/
	int i, j, p = 0, q, r, processing = 0, processing_count = 0, genre = 0, tw_count = 0;
	short *input = NULL;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int count_sp[TRAININGDATA] = { 0 };
	unsigned time_t = 316496; //(unsigned)time(NULL);

	int samplingrate = 22050;

	FILE **f_in = (FILE **)malloc(sizeof(FILE *) * NUMOFGENRE);
	FILE **f_out = (FILE **)malloc(sizeof(FILE *) * 2);

	/* gammatone */
	double **gammatonefilter = NULL;

	/* correlation */
	double max_co = 0;
	int row_index = 0;
	int col_index = 0;

	/* result */
	int** maxindex = NULL;
	int tmp_maxindex[2] = { 0, };
	double* gain = NULL;

	double y[NUMOFSPIKE * 3] = { 0 };

	int while_check = 0;
	int rand_index = 0;

	/*
	*	환경 변수 입력
	*/

	fopen_s(&f_in[0], "all_speech.pcm", "rb");
	fopen_s(&f_in[1], "all_music.pcm", "rb");
	fopen_s(&f_in[2], "all_effect.pcm", "rb");

	fopen_s(&f_out[0], "training_data_m.raw", "wb");
	fopen_s(&f_out[1], "test_data_m.raw", "wb");

	/*
	*	동적 할당
	*/
	input = (short *)malloc(sizeof(short) * FRAME);

	gammatonefilter = (double **)malloc(sizeof(double *) * NUMOFCH);
	for (i = 0; i < NUMOFCH; i++)
		gammatonefilter[i] = (double *)malloc(sizeof(double) * (TWIN + 1));

	maxindex = (int **)calloc(NUMOFSPIKE, sizeof(int *));
	for (i = 0; i < NUMOFSPIKE; i++)
		maxindex[i] = (int*)calloc(2, sizeof(int));

	gain = (double*)malloc(sizeof(double) * NUMOFSPIKE);

	//////////////////////////
	/*		processing		*/
	//////////////////////////

	//GammatoneFilter
	GammatoneFilter(gammatonefilter, samplingrate, EAR_Q, MIN_FRAQ, samplingrate / 2, NUMOFCH, ORDER, MIN_BW, TWIN);   //GammatoneFilter - window
	//GammatoneFilter - window
	for (i = 0; i < NUMOFCH; i++){
		for (j = 0; j < (int)gammatonefilter[i][0]; j++)
			gammatonefilter[i][j + 1] = gammatonefilter[i][j + 1] * sin(PI / gammatonefilter[i][0] * (j + 0.5));
	}
	while_check = 0;
	srand(time_t);
	//rand로 1920개의 training data 중 192개의 test data를 선별하는 과정
	while (while_check != TESTDATA){
		rand_index = rand() % TRAININGDATA;

		if (count_sp[rand_index] == 0){
			count_sp[rand_index] = 1;
			while_check++;
		}
	}


	//Algorithm
	cufftHandle plan[NUMOFLENGTH], plan_IFFT[NUMOFLENGTH];
	cufftComplex *residual;
	cufftComplex **atom;
	cufftComplex *dev_re;
	cufftComplex *dev_atom;
	cufftComplex *dev_overlap;
	cufftComplex *dev_corr_result;

	int *dev_idx_max_out;
	int *idx_max_out;
	double *dev_corr_signal;
	double *dev_corr_max_out;
	double *corr_max_out;

	int length[NUMOFLENGTH] = { TWIN, TWIN / 2, TWIN / 4, TWIN / 8 };
	int count_BW[NUMOFLENGTH] = { 0, };
	int count_BW_sum[NUMOFLENGTH] = { 0, };
	double *tmp_re;


	//////////////////////////////mem alloc
	cudaMalloc((void**)&dev_atom, sizeof(cufftComplex) * NUMOFCH * 2 * TWIN);
	cudaMalloc((void**)&dev_re, sizeof(cufftComplex) * (FRAME+TWIN));
	cudaMalloc((void**)&dev_overlap, sizeof(cufftComplex) * 2 * (FRAME+TWIN));
	cudaMalloc((void**)&dev_idx_max_out, sizeof(int) * NUMOFCH*FRAME / 1024);
	cudaMalloc((void**)&dev_corr_max_out, sizeof(double) * NUMOFCH*FRAME / 1024);
	cudaMalloc((void**)&dev_corr_signal, sizeof(double)*FRAME*NUMOFCH);
	cudaMalloc((void**)&dev_corr_result, sizeof(cufftComplex) * NUMOFCH * 2 * (FRAME+TWIN));

	residual = (cufftComplex *)calloc(FRAME+TWIN, sizeof(cufftComplex));

	atom = (cufftComplex **)calloc(NUMOFCH, sizeof(cufftComplex *));
	for (i = 0; i < NUMOFCH; i++)
		atom[i] = (cufftComplex *)calloc(2 * TWIN, sizeof(cufftComplex));

	tmp_re = (double *)calloc(TWIN, sizeof(double));
	idx_max_out = (int *)calloc(NUMOFCH*FRAME / 1024, sizeof(int));
	corr_max_out = (double *)calloc(NUMOFCH*FRAME / 1024, sizeof(double));

	//atom 배열의 앞부분은 zero padding된 상태에서 뒷자리에 채워넣음
	for (p = 0; p < NUMOFCH; p++){
		int len = gammatonefilter[p][0];
		for (i = 0; i < len; i++)
		{
			atom[p][len + i].x = gammatonefilter[p][i + 1];
		}
	}

	//cuFFT로 미리 atom들을 FFT해서 저장해 놓음
	for (i = 0; i < NUMOFLENGTH; i++) cufftPlan1d(&plan[i], 2 * length[i], CUFFT_C2C, 1);

	int index = 0;
	for (p = 0; p < NUMOFCH; p++){
		if (length[index] != gammatonefilter[p][0]) index++;
		count_BW[index]++;
		cudaMemcpy(&dev_atom[p * 2 * TWIN + length[index]], &atom[p][length[index]], sizeof(cufftComplex)*length[index], cudaMemcpyHostToDevice);
		cufftExecC2C(plan[index], &dev_atom[p * 2 * TWIN], &dev_atom[p * 2 * TWIN], CUFFT_FORWARD);
	}

	//count_BW_sum 배열 채우기
	for (p = 1; p < NUMOFLENGTH; p++) for (i = 0; i < p; i++) count_BW_sum[p] += count_BW[i];
	//plan 재설정
	for (i = 0; i < NUMOFLENGTH; i++){
		cufftDestroy(plan[i]);
		cufftPlan1d(&plan[i], 2 * length[i], CUFFT_C2C, (FRAME / length[i]) + 1);
		cufftPlan1d(&plan_IFFT[i], 2 * length[i], CUFFT_C2C, ((FRAME / length[i]) + 1) * count_BW[i]);
	}

	////////////////////////////////////////////////////////////////////
	FILE *f_restored;
	double *restored_output;
	short *output;
	restored_output = (double *)calloc(FRAME + TWIN, sizeof(double));
	output = (short *)calloc(FRAME, sizeof(short));
	fopen_s(&f_restored, "restored_output.pcm", "wb");

	//double corr_signal[2*(22016+512)];
	//cufftComplex corr_result[2 * (22016 + 512)];
	//cufftComplex overlap[2 * (22016 + 512)];

	//FILE *f_r;
	//fopen_s(&f_r, "aaaa.pcm", "wb");
	//FILE *f_t;
	//fopen_s(&f_t, "aaaa.xls", "wt");
	////////////////////////////////////////////////////////////////////
	
	for (genre = 0; genre < 1; genre++){
		fread(input, sizeof(short), TWIN, f_in[genre]);
		for (i = 0; i < TWIN; i++) residual[i].x = (double)input[i];
		for (processing_count = 0; processing_count < 1; processing_count++){
			if (fread(input, sizeof(short), FRAME, f_in[genre]) == NULL) break;
			for (i = 0; i < FRAME; i++) residual[i + TWIN].x = (double)input[i];
			processing = 0;
			while (processing != NUMOFSPIKE){
				cudaMemcpy(dev_re, residual, sizeof(cufftComplex) * (FRAME+TWIN), cudaMemcpyHostToDevice);
				initialize_corr << < FRAME / TWIN, TWIN >> >(dev_corr_signal);
				for (index = 0; index < NUMOFLENGTH; index++){
					initialize_over << < (FRAME / length[index]) + 1, 2 * length[index] >> >(dev_overlap);
					//Corr_1 : overlap된 배열들로 각각 나누어서 메모리 복사
					Corr_1 << < (FRAME / length[index]) + 1, length[index] >> > (
						dev_re,
						dev_overlap
						);
					//나뉘어진 residual을 2*N SIZE로 FFT
					cufftExecC2C(plan[index], dev_overlap, dev_overlap, CUFFT_FORWARD);
					//Corr_2 : FFT된 ATOM과 RESIDUAL을 complex conjugate multiply
					Corr_2 << < (FRAME / length[index]) + 1, 2 * length[index] >> > (
						dev_atom,
						dev_overlap,
						dev_corr_result,
						count_BW[index],
						count_BW_sum[index]
						);

					//multiply한 결과를 2*N SIZE로 IFFT
					cufftExecC2C(plan_IFFT[index],
						&dev_corr_result[2 * (FRAME + TWIN)*count_BW_sum[index]],
						&dev_corr_result[2 * (FRAME + TWIN)*count_BW_sum[index]],
						CUFFT_INVERSE);

					//Corr_3 : overlap add
					Corr_3 << < FRAME / length[index], length[index] >> > (
						dev_corr_result,
						dev_corr_signal,
						count_BW[index],
						count_BW_sum[index]
						);
				}
				FindMaxInArray <1024><<< NUMOFCH*FRAME /(1024*2), 1024 >>>(
					dev_corr_signal,
					dev_corr_max_out,
					dev_idx_max_out
					);
				cudaMemcpy(corr_max_out, dev_corr_max_out, sizeof(double) * (NUMOFCH*FRAME / 1024), cudaMemcpyDeviceToHost);
				cudaMemcpy(idx_max_out, dev_idx_max_out, sizeof(int) * (NUMOFCH*FRAME / 1024), cudaMemcpyDeviceToHost);

				for (i = (NUMOFCH*FRAME / 1024) - 1; i > 0; i--){
					if (fabs(corr_max_out[i - 1]) < fabs(corr_max_out[i]))
					{
						corr_max_out[i - 1] = corr_max_out[i];
						idx_max_out[i - 1] = idx_max_out[i];
					}
				}
				maxindex[processing][0] = idx_max_out[0] / FRAME;
				maxindex[processing][1] = idx_max_out[0] % FRAME;
				//corr 최대값 찾기
				//maxindex -> 1 col = maxindex row
				//		   -> 2 col = maxindex col

				for (i = 0; i < gammatonefilter[maxindex[processing][0]][0]; i++)
					tmp_re[i] = gammatonefilter[maxindex[processing][0]][i + 1];

				//gain
				gain[processing] = Cal_Gain(residual, tmp_re, gammatonefilter[maxindex[processing][0]][0], maxindex[processing][1]);
				//residual update & 복원
				for (i = 0; i < gammatonefilter[maxindex[processing][0]][0]; i++){
					residual[i + maxindex[processing][1]].x -= gain[processing] * tmp_re[i];
					restored_output[i + maxindex[processing][1]] += gain[processing] * tmp_re[i];
				}
				processing++;

				printf("\r열심히 일하는 중 : %d - %04d - %03d", genre, processing_count, processing);
			}

			for (i = 0; i < FRAME; i++)
				output[i] = (short)restored_output[i];

			fwrite(output, sizeof(short), FRAME, f_restored);
			for (i = 0; i < FRAME; i++)
				restored_output[i] = 0;

			for (i = 0; i < TWIN; i++) residual[i].x = (double)input[FRAME - TWIN + i];

			Make_feature(maxindex, gain, y);

			//50% overlap
			if (count_sp[processing_count] == 0)
				fwrite(y, sizeof(double), NUMOFSPIKE * 3, f_out[0]);
			else
				fwrite(y, sizeof(double), NUMOFSPIKE * 3, f_out[1]);
		}
		printf("\n");
	}

	/*
	*	메모리 해제
	*/
	for (i = 0; i < NUMOFLENGTH; i++){
		cufftDestroy(plan[i]);
		cufftDestroy(plan_IFFT[i]);
	}

	cudaFree(dev_atom);
	cudaFree(dev_re);
	cudaFree(dev_idx_max_out);
	cudaFree(dev_corr_max_out);
	cudaFree(dev_corr_result);
	cudaFree(dev_corr_signal);
	
	free(residual);
	free(input);
	free(tmp_re);
	free(idx_max_out);
	free(corr_max_out);

	for (i = 0; i < NUMOFCH; i++){
		free(gammatonefilter[i]);
		free(atom[i]);
	}
	free(gammatonefilter);
	free(atom);

	for (i = 0; i < NUMOFSPIKE; i++)
		free(maxindex[i]);
	free(maxindex);
	free(gain);

	fclose(f_in[0]);
	fclose(f_in[1]);
	fclose(f_in[2]);

	fclose(f_out[0]);
	fclose(f_out[1]);

	free(f_in);
	free(f_out);
}

void Make_feature(
	int** m,	//m[][0] : dic
	//m[][1] : phase
	double* g,	//g[]    : gain
	double* y   //y[]    : output
	)
{
	int i, j;

	double tmp_double;
	int tmp_int;
	double maxvalue_double;
	int maxvalue_int;

	//Phase 기준 sorting
	for (i = 0; i < NUMOFSPIKE - 1; i++){
		for (j = 0; j < NUMOFSPIKE - i - 1; j++){
			/* For decreasing order use < */
			if (m[j][1] > m[j + 1][1]){
				//dic sort
				tmp_int = m[j][0];
				m[j][0] = m[j + 1][0];
				m[j + 1][0] = tmp_int;

				//phase sort
				tmp_int = m[j][1];
				m[j][1] = m[j + 1][1];
				m[j + 1][1] = tmp_int;

				//int sort
				tmp_double = g[j];
				g[j] = g[j + 1];
				g[j + 1] = tmp_double;
			}
			else if (m[j][1] == m[j + 1][1]){
				if (m[j][0] > m[j + 1][0]){
					//dic sort
					tmp_int = m[j][0];
					m[j][0] = m[j + 1][0];
					m[j + 1][0] = tmp_int;

					//phase sort
					tmp_int = m[j][1];
					m[j][1] = m[j + 1][1];
					m[j + 1][1] = tmp_int;

					//int sort
					tmp_double = g[j];
					g[j] = g[j + 1];
					g[j + 1] = tmp_double;
				}
			}
		}
	}

	for (i = 0; i < NUMOFSPIKE; i++){
		y[3 * i] = m[i][0];
		y[3 * i + 1] = m[i][1];
		y[3 * i + 2] = fabs(g[i]);
	}
}
/*
void Make_feature(
int*** maxindex,
double* y
)
{
int i, j;

double temp[NUMOFSPIKE] = {0};
double temp_var[NUMOFSPIKE] = {0};

//dictionary
for(i = 0; i < NUMOFSPIKE;i++){
temp[i] = 0;
for(j = 0; j < TEXTUREWINDOW; j++){
temp[i] += maxindex[j][i][0];
}
temp[i] /= TEXTUREWINDOW;
}

for(i = 0; i < NUMOFSPIKE; i++){
temp_var[i] = 0;
for(j = 0; j < TEXTUREWINDOW; j++){
temp_var[i] += (maxindex[j][i][0]-temp[i])*(maxindex[j][i][0]-temp[i]);
}
temp_var[i] /= TEXTUREWINDOW;
}

for(i = 0; i < NUMOFSPIKE; i++){
y[2*i    ] = temp[i];
y[2*i + 1] = temp_var[i];
}

//phase
for(i = 0; i < NUMOFSPIKE;i++){
temp[i] = 0;
for(j = 0; j < TEXTUREWINDOW; j++){
temp[i] += maxindex[j][i][1];
}
temp[i] /= TEXTUREWINDOW;
}

for(i = 0; i < NUMOFSPIKE; i++){
temp_var[i] = 0;
for(j = 0; j < TEXTUREWINDOW; j++){
temp_var[i] += (maxindex[j][i][1]-temp[i])*(maxindex[j][i][1]-temp[i]);
}
temp_var[i] /= TEXTUREWINDOW;
}

for(i = 0; i < NUMOFSPIKE; i++){
y[2*i     + NUMOFSPIKE*2] = temp[i];
y[2*i + 1 + NUMOFSPIKE*2] = temp_var[i];
}
}*/