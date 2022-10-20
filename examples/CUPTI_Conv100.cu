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

const char *path_0 = "conv100.csv";
#define N 100 //Default matrix size NxN
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout
#define PROFILE_ALL_EVENTS_METRICS 0
int counter1 = 200000;

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
convolution << <128, 128 >> >(A_d, C_d);//Block-thread
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
  const auto event_names = cupt 
  i_profiler::available_events(device);
  const auto metric_names = cupti_profiler::available_metrics(device);
#else
  vector<string> event_names {
    //"elapsed_cycles_sm",

    // "active_warps",

    // "inst_issued0",

    // "inst_executed",


    // "fb_subp0_read_sectors",
    // "elapsed_cycles_pm",
    //"l2_subp0_write_sector_misses",
    //"l2_subp1_read_sector_misses",
    //"branch",

    // "gld_inst_8bit",

    //"elapsed_cycles_sm",

    // "tex1_cache_sector_queries",

    // "l2_subp0_read_tex_sector_queries",

    // "l2_subp1_write_tex_sector_queries",
  
    // "active_warps",

    // "elapsed_cycles_sm",

    // "l2_subp1_write_sysmem_sector_queries",
    // "l2_subp0_read_sysmem_sector_queries",

 
    

    // "inst_executed",

    // "inst_issued0",

    // "branch",

                    
  };
  vector<string> metric_names {
                    "dram_read_transactions",
                    "local_hit_rate",
                    "dram_write_transactions",
                    "inst_executed",
                    //"stall_memory_dependency",      //*This metrics will cause profiler to be very slow*//
                    //"stall_inst_fetch",            //*This metrics will cause profiler to be very slow*//
                    //"cf_issued",
                    //"tex_fu_utilization",
                    //"l2_write_transactions",
                    //"shared_store_transactions",
                    //"tex_cache_transactions",
                    
  };

  
  #endif
CUcontext context;
cuCtxCreate(&context, 0, 0);


for(int j=0;j<counter1;j++)
{
  cupti_profiler::profiler *p= new cupti_profiler::profiler(event_names, metric_names, context);
  struct timeval ts,te;
  p->start();
  gettimeofday(&ts,NULL);
  
  compute();
  p->stop();
  gettimeofday(&te,NULL);

  p->print_event_values(std::cout,ts,te);
  p->print_metric_values(std::cout,ts,te);

  free(p);
}



  
fclose(stdout);
return 0;
}
