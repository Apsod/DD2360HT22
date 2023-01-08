#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <typeinfo>
#include <iostream>

#ifndef DTYPE
#define DTYPE float
#endif

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)


struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElapsed %.0f microseconds \n", info, time);
}

double cputimer_get(){
  gettimeofday(&t_end, 0);
  return (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
}


__global__ void range_init(DTYPE *ptr, int numel){
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numel){
    ptr[id] = (DTYPE) id;
  }
}

__global__ void linspace(DTYPE *ptr, int numel){
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numel){
    ptr[id] = ((DTYPE) id) / (numel - 1);
  }
}

template<typename T>
T cdiv(T denom, T numer){
  return (denom + numer - 1) / numer;
}

template<typename T>
T asfactor(T num, T factor){
  return cdiv(num, factor) * factor; 
}

void printarr(DTYPE *ptr, int numel){
  printf("%.4f", ptr[0]);
  for(int i = 1; i < numel; ++i){
    printf(", %.4f", ptr[i]);
  }
  printf("\n"); 
}


int main(int argc, char **argv) {
  int factors = argc - 2; 
  size_t *F;
  size_t B;
  size_t numelX = 1;
  size_t numelW = 0;
  size_t fsum = 0;
  unsigned long long int flop = 0;
  DTYPE *RESULT;
  DTYPE *X;  
  DTYPE *Y;
  DTYPE *W;
  DTYPE one = 1.0;
  DTYPE zero = 0.0;

  std::cout << typeid(one).name() << '\n';
  std::cout << sizeof(DTYPE) << '\n';
  cublasHandle_t cublasHandle;      // cuBLAS handle

  F = (size_t*) malloc(factors * sizeof *F); 
  B = (size_t) atoi(argv[1]);
  
  for (int i=0; i < factors; ++i){
    F[i] = (size_t) atoi(argv[i+2]); 
    numelX *= F[i];
    fsum += F[i];
    numelW += F[i] * F[i];
  }
  numelX *= B;
  flop = ((unsigned long long int) numelX) * ((unsigned long long int) fsum);
  // Print input arguments
  printf("State size : %zd\n", numelX / B);
  printf("Batches    : %zd\n", B);
  printf("State tot  : %zd\n", numelX);
  printf("State tot  : %.2e B\n", ((double) (numelX * sizeof(DTYPE))));
  printf("Weight size: %zd\n", numelW);
  printf("FLOPS: %llu\n", flop);
  printf("Factors: %zd", F[0]);
  for (int i = 1; i < factors; ++i){
    printf(" x %zd", F[i]);
  }
  printf("\n");
  
  RESULT = (DTYPE*) malloc(numelX * sizeof (DTYPE)); 
  cudaMalloc(&X, numelX * sizeof(DTYPE));
  cudaMalloc(&Y, numelX * sizeof(DTYPE));
  cudaMalloc(&W, numelW * sizeof(DTYPE));
  
  size_t blocks;
  size_t threads;
  blocks = cdiv(numelX, (size_t) 1024);
  threads = asfactor(cdiv(numelX, blocks), (size_t) 32);
  printf("<<<%zd, %zd>>>\n", blocks, threads);
  linspace<<<blocks, threads>>>(X, numelX);
  blocks = cdiv(numelW, (size_t) 1024);
  threads = asfactor(cdiv(numelW, blocks), (size_t) 32);
  printf("<<<%zd, %zd>>>\n", blocks, threads);
  linspace<<<blocks, threads>>>(W, numelW);
  cudaMemset(Y, 0, numelX * sizeof(DTYPE));

  
  cublasCreate(&cublasHandle);
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
  
  int w_offset = 0;
  DTYPE *S = Y;
  DTYPE *T = X;
  DTYPE *tmp;
  printf("STARTING\n");
  cputimer_start();
  for (int i=0; i < factors; ++i){
    int f = F[i]; 
    tmp = S;
    S = T;
    T = tmp;
    cublasSgemm(
        cublasHandle,         //1
        CUBLAS_OP_T,          //2
        CUBLAS_OP_T,          //3
        numelX / f, f,  f,    //4-6
        &one,                 //7
        S, f,                 //8-9
        W + w_offset, f,
        &zero,
        T, numelX / f
    );
    w_offset += f * f;
  }
  if (B > 1) {
    tmp = S;
    S = T;
    T = tmp;
    cublasSgeam(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        numelX/B, B,
        &one,
        S, B,
        &zero,
        T, numelX/B,
        T, numelX/B
    );
  }
  cudaDeviceSynchronize();
  double elapsed = cputimer_get();
  printf("gemm: %.1e microseconds\n", elapsed); 
  printf("gemm: %.1e flop/s\n", ((double) flop) / (elapsed * 1e-6)); 
  printf("DONE\n");
#ifdef CHECK
  cudaMemcpy(RESULT, T, numelX * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  printarr(RESULT, numelX); 
#endif

  cublasDestroy(cublasHandle);


  //@@ Insert the code for deallocating memory

  cudaFree(X);
  cudaFree(Y);
  cudaFree(W);
  free(F);

  return 0;
}
