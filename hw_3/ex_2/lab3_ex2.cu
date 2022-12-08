#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <stdarg.h>

#define MAXTHREADS 1024

#ifdef FLOAT
#define DataType float
#endif
#ifndef FLOAT
#define DataType double
#endif

#ifdef VERBOSE
//COPIED FROM STACK OVERFLOW
void myprint(const char *fmt, ...){
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}
#endif
#ifndef VERBOSE
void myprint(const char *format, ...){}
#endif


__host__  __device__ int divUp(int numerator, int denominator) {
    // Returns numerator / denominator rounded up. 
    return (numerator + denominator - 1) / denominator;
}

__host__  __device__ int asMultipleOf(int value, int factor) {
    // Returns the smallest value larger or equal to 
    // "value" that is a multiple of "factor"
    // int rest = value % factor;
    return factor * divUp(value, factor); 
}

clock_t CLOCK;

void start_clock() {
    CLOCK = clock();
}

//@@ Insert code to implement timer stop

double stop_clock() {
    clock_t stop_time = clock();
    return ((double) (stop_time - CLOCK)) / CLOCKS_PER_SEC;
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  //const int c_i = blockIdx.y; //* blockDim.x; 
  //const int inner_block = blockIdx.y;
  //const int inner = blockIdx.x * blockDim.x + threadIdx.x;

  const int group_size = blockDim.x;
  const int tid = threadIdx.x;

  const int INNER = numAColumns;
  const int blocks_per_inner = divUp(INNER, group_size); 

  const int c_i = blockIdx.x / blocks_per_inner;
  const int inner = (blockIdx.x % blocks_per_inner) * group_size + tid;
  
  extern __shared__ DataType val[];

  const int c_r = c_i / numBColumns;
  const int c_c = c_i % numBColumns;

  const int ai = c_r * numAColumns + inner;
  const int bi = inner * numBColumns + c_c; 


  if (inner < INNER) {
    val[tid] = A[ai] * B[bi];
  } else {
    val[tid] = 0.0; 
  }

  for(int i = 1; i < group_size; i = i << 1){
    __syncthreads();
    if (tid % (i << 1) == 0 and tid+i < group_size) {
      val[tid] += val[tid+i];
    }
  }
  if (tid == 0){
    atomicAdd(C + c_i, val[0]);
  }

  /*
     a      b    c    d   e
     !      /    !    /   .
     ab     b    cd   d   e
     !           /        .
     abcd   b    cd   d   e
     !                    /
     abcde
  */
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numelsA;
  int numBRows;    // number of rows in the matrix B (The same as number of columns in matrix A)
  int numBColumns; // number of columns in the matrix B
  int numelsB;
  int numCRows;    // number of rows in the result matrix C (The same as the number of rows in A)
  int numCColumns; // number of columns in the result matrix C (The same as the number of columns in B)
  int numelsC;
  double elapsed; 

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numelsA = numARows * numAColumns;

  numBRows = numAColumns;
  numBColumns = atoi(argv[3]);
  numelsB = numBRows * numBColumns;

  numCRows = numARows;
  numCColumns = numBColumns;
  numelsC = numCRows * numCColumns;

  int INNER = numAColumns;

  myprint("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Initialize the grid and block dimensions here
  int TPB = asMultipleOf(divUp(INNER, divUp(INNER, MAXTHREADS)), 32); 
  //int TPB = min(asMultipleOf(INNER, 32), 1024); //INNER;
  int blocks_per_inner = divUp(INNER, TPB); 
  int BLOCKS = blocks_per_inner * numelsC; 
  int SHARED = TPB * sizeof *deviceC;

  myprint("blocks: %d (%d * %d), tpb: %d\n", BLOCKS, blocks_per_inner, numelsC, TPB); 
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*) malloc(numelsA * sizeof *hostA);
  hostB = (DataType*) malloc(numelsB * sizeof *hostB);
  hostC = (DataType*) malloc(numelsC * sizeof *hostC);
  resultRef = (DataType*) malloc(numelsC * sizeof *resultRef);
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  std::normal_distribution<DataType> distribution(0.0, 1.0);
  std::default_random_engine gen(1337);
  
  for (int i=0; i<numelsA; ++i){
      hostA[i] = distribution(gen);
  }
  for (int i=0; i<numelsB; ++i){
      hostB[i] = distribution(gen);
  }

  #ifdef CHECK
  start_clock();
  for (int i=0; i<numelsC; ++i){
      int c_r = i / numCColumns;
      int c_c = i % numCColumns;
      for (int k=0; k<INNER; ++k){
          int ai = c_r * numAColumns + k;
          int bi = k * numBColumns + c_c;
          resultRef[i] += hostA[ai] * hostB[bi];
      }
  }
  elapsed = stop_clock();
  myprint("host           : %f\n", elapsed); 
  #endif

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, numelsA * sizeof *deviceA);
  cudaMalloc(&deviceB, numelsB * sizeof *deviceB);
  cudaMalloc(&deviceC, numelsC * sizeof *deviceC);

  //@@ Insert code to below to Copy memory to the GPU here
  
  start_clock();
  cudaMemcpy(deviceA, hostA, numelsA * sizeof *hostA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numelsB * sizeof *hostB, cudaMemcpyHostToDevice);
  cudaMemset(deviceC, 0, numelsC * sizeof *hostC); 
  elapsed = stop_clock();
  myprint("device to host : %f\n", elapsed); 



  //@@ Launch the GPU Kernel here
  
  start_clock();
  gemm<<<BLOCKS, TPB, SHARED>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  elapsed = stop_clock();
  myprint("kernel         : %f\n", elapsed); 
  //@@ Copy the GPU memory back to the CPU here
  
  start_clock();
  cudaMemcpy(hostC, deviceC, numelsC * sizeof *hostC, cudaMemcpyDeviceToHost);
  elapsed = stop_clock();
  myprint("device to host : %f\n", elapsed); 

  //@@ Insert code below to compare the output with the reference
  #ifdef CHECK
  double err = 0.0;
  double delta = 0.0;
  for (int i= 0; i < numelsC; ++i){
      delta = pow(hostC[i] - resultRef[i], 2);
      if (delta > 0.1) {
          myprint("%d %d\n", i / numCColumns, i % numCColumns);
          myprint("(%f - %f)^2: %f\n\n", hostC[i], resultRef[i], delta);
      }
      err += delta;
  }

  printf("norm of difference: %f\n", sqrt(err));
  #endif

  // Free the GPU memory

  for (DataType *ptr : {deviceA, deviceB, deviceC}) {
    cudaFree(ptr);
  }

  // Free the CPU memory

  for (DataType *ptr : {hostA, hostB, hostC, resultRef}) {
    free(ptr);
  }

  return 0;
}
