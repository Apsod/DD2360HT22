
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <random>
#include <vector>

#define DataType double
#ifndef STREAMS
#define STREAMS 4 
#endif

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}


//@@ Insert code to implement timer start

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//@@ Insert code to implement timer stop

inline __host__ __device__ int divUp(int denom, int numer){
  return (denom + numer - 1) / numer;
}

inline __host__ __device__ int asMultipleOf(int value, int factor) {
  // Returns the smallest value larger or equal to 
  // "value" that is a multiple of "factor"
  return divUp(value, factor) * factor;
}


int main(int argc, char **argv) {
  
  int inputLength;
  int segment_size;
  float elapsed; 
  DataType *hostMemory;

  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;

  DataType *resultRef;

  DataType *deviceMemory;

  
  // Initialize the 1D grid and block dimensions here

  // Read in inputLength from args
  inputLength = atoi(argv[1]);
  segment_size = atoi(argv[2]);

  segment_size = min(segment_size, inputLength); 


  const int segments = divUp(inputLength, segment_size);
  const int BLOCKS = divUp(segment_size, 1024);
  const int TPB = divUp(segment_size, BLOCKS);
  const int n_streams = min(segments, STREAMS);

  printf("The input length is   : %d\n", inputLength);
  printf("number of streams     : %d\n", n_streams);
  printf("number of segments    : %d\n", segments);
  printf("number of blocks/str  : %d\n", BLOCKS);
  printf("number of threads/str : %d\n", TPB);
  
  cudaEvent_t start;
  cudaEvent_t stop; 

  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));

  // Allocate Host memory for input and output
  //checkCuda(cudaMallocHost(&hostMemory, inputLength * 3 * sizeof *hostMemory));
  checkCuda(cudaHostAlloc(&hostMemory, inputLength * 3 * sizeof(DataType), cudaHostAllocDefault));

  hostInput1 = &hostMemory[0*inputLength];
  hostInput2 = &hostMemory[1*inputLength];
  hostOutput = &hostMemory[2*inputLength];


  resultRef = (DataType*) malloc(inputLength * sizeof *resultRef);
  
  
  // Initialize hostInput1 and hostInput2 to random numbers.

  std::normal_distribution<DataType> distribution(0.0, 10.0);
  std::default_random_engine gen(1337);
  
  for (DataType *ptr : {hostInput1, hostInput2}) {
    for (int i=0; i<inputLength; ++i){
      ptr[i] = distribution(gen);
    }
  }
  
  cudaEventRecord(start);
  // Create reference result in CPU
  for (int i=0; i < inputLength; ++i)
    resultRef[i] = hostInput1[i] + hostInput2[i];
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop); 
  printf("host execution time   (ms) : %f\n", elapsed);

  // Allocate GPU memory
  
  cudaEventRecord(start);
  checkCuda(cudaMalloc(&deviceMemory, n_streams * segment_size * 3 * sizeof(DataType)));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop); 
  printf("cudamalloc            (ms) : %f\n", elapsed);

  cudaStream_t streams[n_streams];
  DataType *deviceInput1[n_streams];
  DataType *deviceInput2[n_streams];
  DataType *deviceOutput[n_streams];
  for (int i=0; i < n_streams; ++i){
    cudaStreamCreate(&streams[i]);
    int offset = i * segment_size * 3;
    deviceInput1[i] = &deviceMemory[offset+0*segment_size];
    deviceInput2[i] = &deviceMemory[offset+1*segment_size];
    deviceOutput[i] = &deviceMemory[offset+2*segment_size];
  }

  // Copy memory to the GPU
  


  // Launch the streams
  cudaEventRecord(start);
  for (int i=0; i < segments; ++i){
    int offset = i * segment_size;
    int len = min(segment_size, inputLength - offset);
    int bytes = len * sizeof(DataType); 
    int stream_ix = i % n_streams;
    cudaStream_t stream = streams[stream_ix];
    cudaMemcpyAsync(deviceInput1[stream_ix], &hostInput1[offset], bytes,
        cudaMemcpyHostToDevice, stream
        ); 
    cudaMemcpyAsync(deviceInput2[stream_ix], &hostInput2[offset], bytes,
        cudaMemcpyHostToDevice, stream
        ); 
    vecAdd<<<BLOCKS, TPB, 0, stream>>>(deviceInput1[stream_ix], deviceInput2[stream_ix], deviceOutput[stream_ix], len);
    cudaMemcpyAsync(&hostOutput[offset], deviceOutput[stream_ix], bytes,
        cudaMemcpyDeviceToHost, stream
        ); 
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop); 
  printf("kernel execution time (ms) : %f\n", elapsed);
  printf("GFLOP/s                    : %f\n", ((float) inputLength) / (elapsed * 1e6));

  DataType m = hostOutput[0] - resultRef[0];
  DataType m2 = 0.0;

  DataType maximumError = 0.0;

  int errs = 0;

  for (int i=1; i<inputLength; ++i) {
    DataType delta;
    DataType weight;
    DataType diff;

    weight = DataType(i);
    diff = hostOutput[i] - resultRef[i];

    errs += diff == 0.0 ? 0: 1; 

    delta = diff - m;

    m += delta / (weight + 1.0);
    m2 += pow(delta, 2) * weight / (weight + 1.0);
    maximumError = max(abs(diff), maximumError);
  }

  m2 = sqrt(m2 / DataType(inputLength-1));

  printf("mean difference       : %f\n", m);
  printf("standard deviation    : %f\n", m2);
  printf("maximum error         : %f\n", maximumError);
  printf("number of errors      : %d\n", errs);
  
  // Free the GPU memory

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int i=0; i < n_streams; ++i)
    cudaStreamDestroy(streams[i]);

  cudaFree(deviceMemory);
  cudaFreeHost(hostMemory);
  free(resultRef);

  return 0;
}
