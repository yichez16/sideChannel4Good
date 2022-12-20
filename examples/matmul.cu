#include <cstdio>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <fstream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

#define PROFILE_ALL_EVENTS_METRICS 0
int counter1 = 1;

int numARows = 64;
int numACols = 64;
int numBCols = 64;

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
    for(int i = 0 ; i<10; i++){
    matMul<<<16,128>>>(d_A, d_B, d_C, numARows, numACols, numBCols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);


    free(h_A); free(h_B); free(h_C); 
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

}

 int main()  
 {
  
 
   compute();
 
 
 
   
 }
 
