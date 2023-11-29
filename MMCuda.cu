#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void matrixMul(int *a, int *b, int *c, int n, int m, int p){ 
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;
  if((row<n) && (col < p)){
    for(int k = 0; k < m; k++){
      sum += a[row * m + k] * b[k * p + col];
    }
    c[row * p + col] = sum;
  }
   __syncthreads();
 }


void fillMatrixRandom(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 100;
        }
    }
}

void print_matrix(int *matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d\t", matrix[i * n + j]);
        }
        printf("\n");
    }
}


int main() {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


   int n = 1000;
   int m = 800;
   int p = 1200;

   printf("Matrix sizes: A(%d x %d), B(%d x %d)\n", n, m, m, p);

   size_t bytes_a = n * m * sizeof(int);
   size_t bytes_b = m * p * sizeof(int);
   size_t bytes_c = n * p * sizeof(int);

   int *h_a, *h_b, *h_c;

   h_a = (int*) malloc(bytes_a);
   h_b = (int*) malloc(bytes_b);
   h_c = (int*) malloc(bytes_c);   

   int *d_a, *d_b, *d_c;

   cudaMalloc(&d_a,bytes_a);
   cudaMalloc(&d_b,bytes_b);
   cudaMalloc(&d_c,bytes_c);
   
   fillMatrixRandom(h_a, n, m);
   fillMatrixRandom(h_b, m, p);

   cudaMemcpy(d_a,h_a,bytes_a, cudaMemcpyHostToDevice);
   cudaMemcpy(d_b,h_b,bytes_b, cudaMemcpyHostToDevice);

   int BLOCK_SIZE = 16;

   dim3 grid((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
   dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

   cudaEventRecord(start);

   matrixMul<<<grid,threads>>>(d_a,d_b,d_c,n,m,p);
   cudaDeviceSynchronize();

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);

   float seconds = milliseconds / 1000.0;

   printf("Kernel execution time: %f seconds\n", seconds);

   cudaMemcpy(h_c,d_c,bytes_c,cudaMemcpyDeviceToHost); 

   printf("COMPLETED SUCCESSFULLY\n");

   free(h_a);
   free(h_b);
   free(h_c);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   

   return 0; }
