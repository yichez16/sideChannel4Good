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

#define PROFILE_ALL_EVENTS_METRICS 0
#define N 100
int counter1 = 200000000;

const char *path_0 = "/home/yichez/cupti_profiler/Experiment/blackbox5.6/vectormultiply.csv";

__global__ void
matMul(const int *A, const int *B, int *C, int numElements)
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


static void compute()
 {
    size_t size = N * sizeof(int);

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
    initVec(h_A, N);
    initVec(h_B, N);
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
    matMul << <1, 100  >> > (d_A, d_B, d_C, N);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
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
 
 
     // "tex1_cache_sector_queries",
     "fb_subp1_write_sectors",
     "fb_subp0_read_sectors",
     "l2_subp0_write_sector_misses",
     "l2_subp1_read_sector_misses",
     "branch",
     // "l2_subp0_write_sector_misses",
     // "l2_subp1_read_sector_misses",
     // "branch",
 
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
                     // "dram_read_transactions",
                     // //"local_hit_rate",
                     // "dram_write_transactions",
                     //"inst_executed",
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
 
