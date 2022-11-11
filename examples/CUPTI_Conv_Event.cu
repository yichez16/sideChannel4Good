#include <cstdio>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <fstream>
#include <cupti_profiler.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

const char *path_0 = "conv_event.csv";
#define N 128 //Default matrix size NxN
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout
#define PROFILE_ALL_EVENTS_METRICS 0
int counter1 = 200000;

int numARows = 32;
int numACols = 32;
int numBCols = 32;

__global__ void convolution(int *A, int *C)
{
	//Filter
	int filter[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };

	//Needs for row-major layout
	int cols = N + 2;
	//int i = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int threadBlockSize = (N+2)/ blockDim.x;//The amount of processing per thread

	for (int b = threadIdx.x * threadBlockSize; b < (threadIdx.x + 1) * threadBlockSize; b++){
		
		i = b;
		
		for (int j = 0; j < N + 1; j++){//columns
			
			if (0 < i && i < N + 1 && 0 < j && j < N + 1)
			{
				int value = 0;
				value = value + A(i - 1, j - 1)	*  filter[0][0];
				value = value + A(i - 1, j)		*  filter[0][1];
				value = value + A(i - 1, j + 1)	*  filter[0][2];
				value = value + A(i, j - 1)		*  filter[1][0];
				value = value + A(i, j)			*  filter[1][1];
				value = value + A(i, j + 1)		*  filter[1][2];
				value = value + A(i + 1, j - 1)	*  filter[2][0];
				value = value + A(i + 1, j)		*  filter[2][1];
				value = value + A(i + 1, j + 1)	*  filter[2][2];
				C(i, j) = value;
			}
		}
	}


}

__global__ void matMul(float* A, float* B, float* C, int numARows, int numACols, int numBCols) {
    // compute global thread coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // linearize coordinates for data access
    int offset = row * numBCols + col;

    if ((row < numARows) && (col < numBCols)) {
        float cumSum = 0;
        for (int k = 0; k < numACols; k++) {
            cumSum += A[row*numACols + k] * B[k*numBCols + col];
        }
        C[offset] = cumSum;
    }
}

__global__ void
vecMul(const int *A, const int *B, int *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

static void
initVec(int *vec, int n)
{
    for (int i = 0; i < n; i++)
        vec[i] = i;
}


static void compute_vecmul()
 {
    size_t size = numARows * sizeof(int);

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    
    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, numARows);
    initVec(h_B, numARows);
    memset(h_C, 0, size);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    // int priority_hi = -1;
    // cudaStream_t st_hi;
    // cudaStreamCreateWithPriority(&st_hi,  cudaStreamNonBlocking, priority_hi);
    vecMul << <64,128  >> > (d_A, d_B, d_C, numARows);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
 }

static void compute_mat() {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t sizeA = numARows * numACols * sizeof(float);
    size_t sizeB = numACols * numBCols * sizeof(float);
    size_t sizeC = numARows * numBCols * sizeof(float);

    // allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    // initialize host matrices
    int i, j, offset;
    for (i = 0; i <  numARows; i++) {
        for (j = 0; j < numACols; j++) {
            offset = i*numACols + j;
            h_A[offset] = i;
        }
    }
    for (i = 0; i <  numACols; i++) {
        for (j = 0; j < numBCols; j++) {
            offset = i*numBCols + j;
            h_B[offset] = i;
        }
    }

    // allocate device matrices
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // transfer to GPU
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // kernel launch
    

    // dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // dim3 blockPerGrid(ceil(numBCols/(float)BLOCK_SIZE), ceil(numACols/(float)BLOCK_SIZE), 1);


    cudaEventRecord(start);
    matMul<<<64,128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);


    free(h_A); free(h_B); free(h_C); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}



static void compute()
{
    //Host variables
	int A[N+2][N+2] = {};//+2 for padding matrix
	int *C;
	
	//Device variables
	int *A_d = 0, *C_d = 0;


	//Calculate memory size 
	int memorySize = (N + 2) * (N + 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init matrix by 0
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
			A[i][j] = 0;
		}
}

	//Generate random values between 0 and 9
srand(time(NULL));
for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) {
    A[i + 1][j + 1] = rand() % 10;
  }
}

C = (int *)malloc(sizeof(*C)*memorySize);

cudaMalloc((void**)&A_d, sizeof(*A_d)*memorySize);
cudaMalloc((void**)&C_d, sizeof(*C_d)*memorySize);

//Copy from host to device
cudaMemcpy(A_d, A, sizeof(*A_d)*memorySize, cudaMemcpyHostToDevice);

// cudaEventRecord(start);
convolution << <64,128 >> >(A_d, C_d);//Block-thread
// cudaEventRecord(stop);
// cudaEventSynchronize(stop);

//Copy from device to host
cudaMemcpy(C, C_d, sizeof(*C)*memorySize, cudaMemcpyDeviceToHost);


//Free memory
cudaFree(C_d);
cudaFree(A_d);
free(C);
}














int main()  
{
freopen(path_0,"w",stdout);

using namespace std;
CUdevice device;

DRIVER_API_CALL(cuInit(0));
DRIVER_API_CALL(cuDeviceGet(&device, 0));

#if PROFILE_ALL_EVENTS_METRICS
const auto event_names = cupti_profiler::available_events(device);
const auto metric_names = cupti_profiler::available_metrics(device);
#else
  vector<string> event_names {    
    "fb_subp0_read_sectors",
//    "fb_subp1_read_sectors",
    //  "fb_subp0_write_sectors",
    // "fb_subp1_write_sectors",
    "l2_subp0_read_sector_misses",
    "tex0_cache_sector_queries",
    "inst_issued0",
    "inst_executed",
    // "atom_count",
    // "inst_executed",
    // "inst_issued0",
    // "shared_load",
    // "generic_load",
    // "global_load",
    // "local_load",
    // "shared_ld_bank_conflict",
    // "shared_ld_transactions",
    // "l2_subp1_read_sector_misses",
    //  "l2_subp0_write_sector_misses",
    // "l2_subp1_write_sector_misses"



  };
  vector<string> metric_names {
// "l2_read_transactions",// works
//"nvlink_data_receive_efficiency",
// "nvlink_data_transmission_efficiency",
//"nvlink_overhead_data_received",
//"nvlink_overhead_data_transmitted",
//"nvlink_receive_throughput",
// "inst_control",
// "inst_fp_32",
// "inst_fp_64",
// "inst_integer",
// "inst_executed", not work
// "inst_issued",
// "inst_per_warp",


//  "nvlink_total_data_received",// works
//  "nvlink_total_data_transmitted",// works
//  "nvlink_total_nratom_data_transmitted" , // works
// // "nvlink_total_ratom_data_transmitted" ,
//  "nvlink_total_response_data_received" ,// works
// // "nvlink_total_write_data_transmitted",
//  "nvlink_transmit_throughput", //works
// "nvlink_user_data_received",
// "nvlink_user_data_transmitted",
// "nvlink_user_nratom_data_transmitted" ,
// "nvlink_user_ratom_data_transmitted",
// "nvlink_user_response_data_received",
// "nvlink_user_write_data_transmitted",

// "l2_write_transactions",  // error
// "dram_read_transactions",
//"dram_write_transactions",

                    
  };

  
  #endif
CUcontext context;
cuCtxCreate(&context, 0, 0);


for(int i=0;i<2;i++)
{
	for(int j=0;j<50;j++)
	{
	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
	struct timeval ts,te;
	p->start();
	gettimeofday(&ts,NULL);
	
	compute_mat();
	compute_vecmul();
	p->stop();
	gettimeofday(&te,NULL);

	p->print_event_values(std::cout,ts,te);
	// p->print_metric_values(std::cout,ts,te);
	// p->print_events_and_metrics(std::cout);
	free(p);	
	
	}
	for(int m=0;m<1;m++)
	{
	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
	struct timeval ts,te;
	p->start();
	gettimeofday(&ts,NULL);
	
	compute();
	// compute_mat();
	// compute_vecmul();


	p->stop();
	gettimeofday(&te,NULL);

	p->print_event_values(std::cout,ts,te);
	// p->print_metric_values(std::cout,ts,te);
	
	free(p);
	}

	// free(p);
}


// for(int i=0;i<10;i++)
// {
// 	for(int j=0;j<75;j++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute_mat();
// 	compute_vecmul();
// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	// p->print_events_and_metrics(std::cout);

	
// 	}
// 	for(int m=0;m<1;m++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute();
// 	// compute_mat();
// 	// compute_vecmul();


// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	}

// 	// free(p);
// }

// for(int i=0;i<10;i++)
// {
// 	for(int j=0;j<40;j++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute_mat();
// 	compute_vecmul();
// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	// p->print_events_and_metrics(std::cout);

	
// 	}
// 	for(int m=0;m<1;m++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute();
// 	// compute_mat();
// 	// compute_vecmul();


// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	}

// 	// free(p);
// }

// for(int i=0;i<10;i++)
// {
// 	for(int j=0;j<50;j++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute_mat();
// 	compute_vecmul();
// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	// p->print_events_and_metrics(std::cout);

	
// 	}
// 	for(int m=0;m<1;m++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute();
// 	// compute_mat();
// 	// compute_vecmul();


// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	}

// 	// free(p);
// }

// for(int i=0;i<10;i++)
// {
// 	for(int j=0;j<30;j++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute_mat();
// 	compute_vecmul();
// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	// p->print_events_and_metrics(std::cout);

	
// 	}
// 	for(int m=0;m<1;m++)
// 	{
// 	cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
// 	struct timeval ts,te;
// 	p->start();
// 	gettimeofday(&ts,NULL);
	
// 	compute();
// 	// compute_mat();
// 	// compute_vecmul();


// 	p->stop();
// 	gettimeofday(&te,NULL);

// 	p->print_event_values(std::cout,ts,te);
// 	// p->print_metric_values(std::cout,ts,te);
// 	}

// 	// free(p);
// }

  
fclose(stdout);
return 0;
}
