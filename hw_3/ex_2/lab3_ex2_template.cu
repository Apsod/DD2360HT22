
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

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
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output

  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU


  //@@ Insert code below to allocate GPU memory here


  //@@ Insert code to below to Copy memory to the GPU here


  //@@ Initialize the grid and block dimensions here


  //@@ Launch the GPU Kernel here


  //@@ Copy the GPU memory back to the CPU here


  //@@ Insert code below to compare the output with the reference


  //@@ Free the GPU memory here


  //@@ Free the CPU memory here


  return 0;
}
