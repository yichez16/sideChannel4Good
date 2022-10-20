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
int counter1 = 200000000;
// #define BLOCK_SIZE 16
const char *path_0 = "/home/yichez/cupti_profiler/Experiment/blackbox5.6/matmul.csv";

int numARows = 100;
int numACols = 100;
int numBCols = 100;

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




static void compute() {

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
    matMul<<<1,100>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);


    free(h_A); free(h_B); free(h_C); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

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
 
