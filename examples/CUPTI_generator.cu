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

const char *path_0 = "CUPTI_counter.csv";
const char *path_1 = "backup.csv";

#define N 25 //Default matrix size NxN
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout
#define PROFILE_ALL_EVENTS_METRICS 0
int counter1 = 200000;

int numARows = 32;
int numACols = 32;
int numBCols = 32;



__global__ void sideChannelGenerator(int *A, int *C)
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



static void compute_mat(int stride, int Verbose) {

    ////////////////////////////Profiler//////////////////////////////////////////

    using namespace std;
    CUdevice device;
    
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    
    
    #if PROFILE_ALL_EVENTS_METRICS
    const auto event_names = cupti_profiler::available_events(device);
    const auto metric_names = cupti_profiler::available_metrics(device);
    #else
        vector<string> event_names {    
        // "fb_subp0_write_sectors",
        // "l2_subp0_read_tex_hit_sectors",
        // "tex0_cache_sector_queries",
        // "inst_executed",
        // // "inst_issued0",
        // "global_store",
        // "global_load",
        "active_warps",
    
        // "atom_count",
        // "shared_load",
        // "generic_load",
        // "global_load",
        // "local_load",
        // "shared_ld_bank_conflict",
        // "shared_ld_transactions",
    
    
    
        };
        vector<string> metric_names {
    
                        
        };
        
        #endif
    
    
    
    CUcontext context;
    cuCtxCreate(&context, 0, 0);

          


    ///////////////////////Profiler////////////////////////////////////
    

    size_t sizeA = numARows * numACols * sizeof(float);
    size_t sizeB = numACols * numBCols * sizeof(float);
    size_t sizeC = numARows * numBCols * sizeof(float);

    //Host variables
	int A[N+2][N+2] = {};//+2 for padding matrix
	int *C;
    //Device variables
	int *A_d = 0, *C_d = 0;
	//Calculate memory size 
	int memorySize = (N + 2) * (N + 2);
	//Init matrix by 0
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
			A[i][j] = 0;
		}
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          A[i + 1][j + 1] = rand() % 10;
        }
      }    

    C = (int *)malloc(sizeof(*C)*memorySize);      




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


    cudaMalloc((void**)&A_d, sizeof(*A_d)*memorySize);
    cudaMalloc((void**)&C_d, sizeof(*C_d)*memorySize);


    /////////////// transfer to GPU/////////////////////////////



    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);


    cudaMemcpy(A_d, A, sizeof(*A_d)*memorySize, cudaMemcpyHostToDevice);




    // kernel launch 
    
    cupti_profiler::profiler *p1= new cupti_profiler::profiler(event_names, metric_names, context);
    struct timeval ts1,te1;   
    p1->start();
    gettimeofday(&ts1,NULL);
    matMul<<<32, 128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    p1->stop();
    gettimeofday(&te1,NULL);
    p1->print_event_values(std::cout,ts1,te1);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
    struct timeval ts,te;  
    p->start();
    gettimeofday(&ts,NULL);
    matMul<<<32,128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    p->stop();
    gettimeofday(&te,NULL);
    p->print_event_values(std::cout,ts,te);


    for(int i = 0; i < stride; i++){





        int frequency = 100000/stride;

        for (int j = 0; j < frequency; j++) {

            p->start();
            gettimeofday(&ts,NULL);
            matMul<<<32,128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
            p->stop();
            gettimeofday(&te,NULL);
            p->print_event_values(std::cout,ts,te);
            
        }

        if(Verbose){
            printf("--------------------Spike------------------------\n");
        }
        p1->start();
        gettimeofday(&ts1,NULL);
        matMul<<<32, 128,  0, stream0>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
        /////////// Embedded with side channel spike generator ////////////
        sideChannelGenerator <<<128, 128,  0, stream1>>>(A_d, C_d);
        /////////// Embedded with side channel spike generator ////////////
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        p1->stop();
        gettimeofday(&te1,NULL);
        p1->print_event_values(std::cout,ts1,te1);
        if(Verbose){
            printf("--------------------Spike------------------------\n");
        }


        
    }
















    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);





    for(int iter = 0; iter < 100; iter++){
        p->start();
        gettimeofday(&ts,NULL);
        matMul<<<32,128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
        p->stop();
        gettimeofday(&te,NULL);
        p->print_event_values(std::cout,ts,te);
    }














    ////////////////////////////Profiler /////////


    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, C_d, sizeof(*C)*memorySize, cudaMemcpyDeviceToHost);
    free(h_A); free(h_B); free(h_C); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(C_d);
    cudaFree(A_d);
    free(C);

}






int main(int argc, char **argv)  
{
    int stride;
    bool Verbose;
    int temp;
    char y[5];

    printf("Please enter the frequency of impulses.\n");
    scanf("%d", &stride);
    printf("Verbose or not?\n");
    scanf("%s", y);

    if(strcmp(y, "True") == 0){
        temp = 1;
    }

    else if(strcmp(y, "False") == 0){
        temp = 0;
    }

    Verbose = temp;




    if(Verbose){
        for(int j=0;j<2;j++)
        {
            compute_mat(stride, Verbose);
        }
    }
    else{
        printf("Save to file CUPTI_counter.csv\n");


        freopen(path_0,"w",stdout);
        for(int j=0;j<1;j++)
        {
            compute_mat(stride, Verbose);
        }
        // compute_mat(stride, Verbose);
        fclose(stdout);
        
    }

        
    return 0;
}
