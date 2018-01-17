#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DX 1
#define DY 1
#define DT 1
#define ERRMAX 1E-5
#define NX 201
#define NY 201
#define NTMAX 50000
#define NTHREAD 512
#define NBLOCK 256
#define Q 9
#define RE 1000.
#define RHO0 1.
#define U0 0.1
#define OUTPUTFILENAME "output.dat"

struct attribute{
	double f[Q];
	double fnew[Q];
	double g[Q];
	double u[2];
	double u0[2];
	double rho;
	double vor;
};

__constant__ double dev_CS2[1], dev_NU[1], dev_TAO[1], dev_w[Q];
__constant__ int dev_e[Q * 2];
const double w[Q] = { 4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36,\
	1. / 36, 1. / 36, 1. / 36 };
int C = DX / DT;
int e[Q * 2] = { 0, 0, C, 0, 0, C, -C, 0, 0, -C, C, C, -C, C, -C, -C, C, -C };
double *CS2, *NU, *TAO;
attribute *domain;

cudaError_t LBM();
int Ord2(int x, int y, int nx);
double Cfeq(double u[2], double rho, int k);
void Init();
void OutWatch(int t, double err, double rhoav);
void Outp();
__device__ int d_Ord2(int x, int y, int nx);
__device__ void d_Ord2r(int id, int *x, int *y, int nx);
__device__ double d_Cfeq(double u[2], double rho, int k);
__global__ void Collision(attribute *domain);
__global__ void PreStreaming(attribute *domain);
__global__ void Streaming(attribute *domain);
__global__ void Physics(attribute *domain);
__global__ void MovingBoundary(attribute *domain);
__global__ void StaticBoundary(attribute *domain);
__global__ void Error(attribute *domain, double *err);
__global__ void Watch(attribute *domain, double *rhoav);
__global__ void Cvor(attribute *domain);

int main()
{
	Init();
	cudaError_t cudaStatus = LBM();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda LBM Failed!");
		return 1;
	}
	Outp();

	free(CS2);
	free(NU);
	free(TAO);
	free(domain);

	return 0;
}

cudaError_t LBM(){
	attribute *dev_domain;
	cudaError_t cudaStatus;
	clock_t start, finish;
	double *err = (double*)malloc(sizeof(double));
	double *rhoav = (double*)malloc(sizeof(double));
	double *dev_err, *dev_rhoav;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) goto Error;

	cudaStatus = cudaMalloc((void**)&dev_err, sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_rhoav, sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMalloc((void**)&dev_domain, NX * NY * sizeof(attribute));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpy(dev_domain, domain, NX * NY * sizeof(attribute),\
		cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) goto Error;

	cudaStatus = cudaMemcpyToSymbol(dev_w, w, Q * sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpyToSymbol(dev_e, e, Q * 2 * sizeof(int));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpyToSymbol(dev_CS2, CS2, sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpyToSymbol(dev_NU, NU, sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;
	cudaStatus = cudaMemcpyToSymbol(dev_TAO, TAO, sizeof(double));
	if (cudaStatus != cudaSuccess) goto Error;

	start = clock();
	for (int t = 0;; t++){
		Collision <<<NBLOCK, NTHREAD >>>(dev_domain);
		PreStreaming <<<NBLOCK, NTHREAD >>>(dev_domain);
		Streaming <<<NBLOCK, NTHREAD >>>(dev_domain);
		Physics <<<NBLOCK, NTHREAD >>>(dev_domain);
		MovingBoundary <<<NBLOCK, NTHREAD >>>(dev_domain);
		StaticBoundary <<<NBLOCK, NTHREAD >>>(dev_domain);
		Error <<<1, NTHREAD >>>(dev_domain, dev_err);

		cudaStatus = cudaMemcpy(err, dev_err, sizeof(double),\
			cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) goto Error;
		
		if (t % 100 == 0){
			Watch <<<1, NTHREAD >>>(dev_domain, dev_rhoav);
			cudaStatus = cudaMemcpy(rhoav, dev_rhoav, sizeof(double),\
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;
			OutWatch(t, *err, *rhoav);
		}
		
		if ((*err < ERRMAX) || (t > NTMAX)) break;
	}
	Cvor <<<NBLOCK, NTHREAD >>>(dev_domain);
	finish = clock();
	printf("Time Take : %7.2lfs\n", (double)(finish - start) / CLOCKS_PER_SEC);

	cudaStatus = cudaMemcpy(domain, dev_domain, NX * NY * sizeof(attribute),\
		cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;

Error:
	free(err);
	free(rhoav);
	cudaFree(dev_err);
	cudaFree(dev_rhoav);
	cudaFree(dev_domain);

	return cudaStatus;
}

int Ord2(int x, int y, int nx){
	return y * nx + x;
}

double Cfeq(double u[2], double rho, int k){
	double v1 = e[k * 2] * u[0] + e[k * 2 + 1] * u[1];
	double v2 = u[0] * u[0] + u[1] * u[1];
	return (rho * w[k] * (1.0 + v1 / (*CS2) + v1 * v1 / (2. * (*CS2)\
		* (*CS2)) - v2 / (2. * (*CS2))));
}

void Init(){
	CS2 = (double*)malloc(sizeof(double));
	NU = (double*)malloc(sizeof(double));
	TAO = (double*)malloc(sizeof(double));
	*CS2 = C * C / 3.;
	*NU = U0 * (NX - 1) * DX / RE;
	*TAO = *NU / DT / *CS2 + 0.5;
	domain = (attribute*)malloc(NX * NY * sizeof(attribute));
	memset(domain, 0, NX * NY * sizeof(attribute));
	for (int i = 0; i < NX; i++)
		for (int j = 0; j < NY; j++){
			domain[Ord2(i, j, NX)].rho = RHO0;
			domain[Ord2(i, j, NX)].u[0] = (j == NY - 1) ? U0 : 0.;
			domain[Ord2(i, j, NX)].u[1] = 0.;
			for (int k = 0; k < Q; k++)
				domain[Ord2(i, j, NX)].f[k] = Cfeq(domain[Ord2(i, j, NX)].u, \
				domain[Ord2(i, j, NX)].rho, k);
		}
	return;
}

void OutWatch(int t, double err, double rhoav){
	printf("Time Step %d, ", t);
	printf("Error = %lf, ", err);
	printf("Average Density = %lf\n", rhoav);
}

void Outp(){
	FILE *fp = fopen(OUTPUTFILENAME, "w");
	fprintf(fp, "Title=\"LBM Lid Driven Cavity\"\n");
	fprintf(fp, "VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"P\",\"OMG\"\n");
	fprintf(fp, "ZONE T=\"BOX\",I=%d,J=%d,F=POINT\n", NX, NY);
	for (int j = 0; j < NY; j++)
		for (int i = 0; i < NX; i++){
			fprintf(fp, "%lf ", (double)i / (NX - 1) / DX);
			fprintf(fp, "%lf ", (double)j / (NY - 1) / DY);
			fprintf(fp, "%lf ", domain[Ord2(i, j, NX)].u[0]);
			fprintf(fp, "%lf ", domain[Ord2(i, j, NX)].u[1]);
			fprintf(fp, "%lf ", domain[Ord2(i, j, NX)].rho * (*CS2));
			fprintf(fp, "%lf\n", domain[Ord2(i, j, NX)].vor);
		}
	fclose(fp);
	return;
}

__device__ int d_Ord2(int x, int y, int nx){
	return y * nx + x;
}

__device__ void d_Ord2r(int id, int *x, int *y, int nx){
	*y = id / nx;
	*x = id % nx;
	return;
}

__device__ double d_Cfeq(double u[2], double rho, int k){
	double v1 = dev_e[k * 2] * u[0] + dev_e[k * 2 + 1] * u[1];
	double v2 = u[0] * u[0] + u[1] * u[1];
	return (rho * dev_w[k] * (1.0 + v1 / (*dev_CS2) + v1 * v1 / (2.\
		* (*dev_CS2) * (*dev_CS2)) - v2 / (2. * (*dev_CS2))));
}

__global__ void Collision(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		for (int k = 0; k < Q; k++){
			double feq = d_Cfeq(domain[tid].u, domain[tid].rho, k);
			domain[tid].fnew[k] = domain[tid].f[k] - (domain[tid].f[k]\
				- feq) / (*dev_TAO);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void PreStreaming(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		for (int k = 0; k < Q; k++)
			domain[tid].g[k] = domain[tid].fnew[k];
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Streaming(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		for (int k = 0; k < Q; k++){
			int xn = (x + dev_e[k * 2] + NX) % NX;
			int yn = (y + dev_e[k * 2 + 1] + NY) % NY;
			domain[d_Ord2(xn, yn, NX)].fnew[k] = domain[tid].g[k];
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void Physics(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			domain[tid].u0[0] = domain[tid].u[0];
			domain[tid].u0[1] = domain[tid].u[1];
			domain[tid].rho = 0.;
			domain[tid].u[0] = 0.;
			domain[tid].u[1] = 0.;
			for (int k = 0; k < Q; k++){
				domain[tid].f[k] = domain[tid].fnew[k];
				domain[tid].rho += domain[tid].f[k];
				domain[tid].u[0] += dev_e[k * 2] * domain[tid].f[k];
				domain[tid].u[1] += dev_e[k * 2 + 1] * domain[tid].f[k];
			}
			domain[tid].u[0] /= domain[tid].rho;
			domain[tid].u[1] /= domain[tid].rho;
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void MovingBoundary(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX){
		domain[d_Ord2(tid, NY - 1, NX)].rho = domain[d_Ord2(tid, NY - 2, NX)].rho;
		domain[d_Ord2(tid, NY - 1, NX)].u[0] = U0;
		for (int k = 0; k < Q; k++){
			domain[d_Ord2(tid, NY - 1, NX)].f[k] = d_Cfeq(domain[d_Ord2(tid, \
				NY - 1, NX)].u, domain[d_Ord2(tid, NY - 1, NX)].rho, k)\
				+ domain[d_Ord2(tid, NY - 2, NX)].f[k] - d_Cfeq(domain[d_Ord2(tid, \
				NY - 2, NX)].u, domain[d_Ord2(tid, NY - 2, NX)].rho, k);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}

__global__ void StaticBoundary(attribute *domain){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid0 = tid;
	while (tid < NX){
		domain[d_Ord2(tid, 0, NX)].rho = domain[d_Ord2(tid, 1, NX)].rho;
		for (int k = 0; k < Q; k++){
			domain[d_Ord2(tid, 0, NX)].f[k] = d_Cfeq(domain[d_Ord2(tid, \
				0, NX)].u, domain[d_Ord2(tid, 0, NX)].rho, k)\
				+ domain[d_Ord2(tid, 1, NX)].f[k] - d_Cfeq(domain[d_Ord2(tid, \
				1, NX)].u, domain[d_Ord2(tid, 1, NX)].rho, k);
		}
		tid += blockDim.x * gridDim.x;
	}
	tid = tid0;
	while (tid < NY - 1){
		if (tid > 0){
			domain[d_Ord2(0, tid, NX)].rho = domain[d_Ord2(1, tid, NX)].rho;
			domain[d_Ord2(NX - 1, tid, NX)].rho = domain[d_Ord2(NX - 2,\
				tid, NX)].rho;
			for (int k = 0; k < Q; k++){
				domain[d_Ord2(0, tid, NX)].f[k] = d_Cfeq(domain[d_Ord2(0,\
					tid, NX)].u, domain[d_Ord2(0, tid, NX)].rho, k)\
					+ domain[d_Ord2(1, tid, NX)].f[k] - d_Cfeq(domain[d_Ord2(1,\
					tid, NX)].u, domain[d_Ord2(1, tid, NX)].rho, k);
				domain[d_Ord2(NX - 1, tid, NX)].f[k] = d_Cfeq(domain[d_Ord2(NX - 1,\
					tid, NX)].u, domain[d_Ord2(NX - 1, tid, NX)].rho, k)\
					+ domain[d_Ord2(NX - 2, tid, NX)].f[k] - d_Cfeq(domain[d_Ord2(NX - 2,\
					tid, NX)].u, domain[d_Ord2(NX - 2, tid, NX)].rho, k);
			}
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}	

__global__ void Error(attribute *domain, double *err){
	__shared__ double cache[NTHREAD][2];
	int x, y;
	int tid = threadIdx.x;
	cache[threadIdx.x][0] = 0.;
	cache[threadIdx.x][1] = 0.;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			cache[threadIdx.x][0] += (domain[tid].u[0] - domain[tid].u0[0])\
				* (domain[tid].u[0] - domain[tid].u0[0]) + (domain[tid].u[1]\
				- domain[tid].u0[1]) * (domain[tid].u[1] - domain[tid].u0[1]);
			cache[threadIdx.x][1] += domain[tid].u[0] * domain[tid].u[0]\
				+ domain[tid].u[1] * domain[tid].u[1];
		}
		tid += blockDim.x;
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i > 0){
		if (threadIdx.x < i){
			cache[threadIdx.x][0] += cache[threadIdx.x + i][0];
			cache[threadIdx.x][1] += cache[threadIdx.x + i][1];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
		*err = sqrt(cache[threadIdx.x][0] / (cache[threadIdx.x][1] + 1.E-10));
	return;
}

__global__ void Watch(attribute *domain, double *rhoav){
	__shared__ int cachen[NTHREAD];
	__shared__ double cache[NTHREAD];
	int x, y;
	int tid = threadIdx.x;
	cachen[threadIdx.x] = 0;
	cache[threadIdx.x] = 0.;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			cachen[threadIdx.x]++;
			cache[threadIdx.x] += domain[tid].rho;
		}
		tid += blockDim.x;
	}
	__syncthreads();
	int i = blockDim.x / 2;
	while (i > 0){
		if (threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
			cachen[threadIdx.x] += cachen[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
		*rhoav = cache[threadIdx.x] / ((double)cachen[threadIdx.x] + 1.E-10);
	return;
}

__global__ void Cvor(attribute *domain){
	int x, y;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NX * NY){
		d_Ord2r(tid, &x, &y, NX);
		if ((x > 0) && (y > 0) && (x < NX - 1) && (y < NY - 1)){
			domain[tid].vor = (domain[d_Ord2(x + 1, y, NX)].u[1]\
				- domain[d_Ord2(x - 1, y, NX)].u[1]) / (2 * DX);
			domain[tid].vor -= (domain[d_Ord2(x, y + 1, NX)].u[0]\
				- domain[d_Ord2(x, y - 1, NX)].u[0]) / (2 * DY);
		}
		tid += blockDim.x * gridDim.x;
	}
	return;
}
