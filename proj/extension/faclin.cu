#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>

cublasStatus_t gemm_sb(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    long long int strideA,
    const float *B, int ldb,
    long long int strideB,
    const float *beta,
    float       *C, int ldc,
    long long int strideC,
    int batchCount
    ){
  return cublasSgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      strideA,
      B, ldb,
      strideB,
      beta,
      C, ldc,
      strideC,
      batchCount
      );
}

cublasStatus_t gemm_sb(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda,
    long long int strideA,
    const double *B, int ldb,
    long long int strideB,
    const double *beta,
    double       *C, int ldc,
    long long int strideC,
    int batchCount
    ){
  return cublasDgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      strideA,
      B, ldb,
      strideB,
      beta,
      C, ldc,
      strideC,
      batchCount
      );
}

//cublasStatus_t gemm(
//    cublasHandle_t handle,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n, int k,
//    const float *alpha,
//    const float *A, int lda,
//    const float *B, int ldb,
//    const float *beta,
//    float       *C, int ldc){
//  return cublasSgemm(
//      handle,
//      transa, transb,
//      m, n, k,
//      alpha,
//      A, lda,
//      B, ldb,
//      beta,
//      C, ldc);
//}
//
//
//cublasStatus_t gemm(
//    cublasHandle_t handle,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n, int k,
//    const double *alpha,
//    const double *A, int lda,
//    const double *B, int ldb,
//    const double *beta,
//    double       *C, int ldc){
//  return cublasDgemm(
//      handle,
//      transa, transb,
//      m, n, k,
//      alpha,
//      A, lda,
//      B, ldb,
//      beta,
//      C, ldc);
//}
//
//cublasStatus_t geam(
//    cublasHandle_t handle,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n,
//    const float *alpha,
//    const float *A, int lda,
//    const float *beta,
//    const float *B, int ldb,
//    float       *C, int ldc){
//  return cublasSgeam(
//      handle,
//      transa, transb,
//      m, n,
//      alpha,
//      A, lda,
//      beta,
//      B, ldb,
//      C, ldc
//      );
//}
//
//cublasStatus_t geam(
//    cublasHandle_t handle,
//    cublasOperation_t transa, cublasOperation_t transb,
//    int m, int n,
//    const double *alpha,
//    const double *A, int lda,
//    const double *beta,
//    const double *B, int ldb,
//    double       *C, int ldc){
//  return cublasDgeam(
//      handle,
//      transa, transb,
//      m, n,
//      alpha,
//      A, lda,
//      beta,
//      B, ldb,
//      C, ldc
//      );
//}

template<typename scalar_t>
void faclin_forward_inner(
    cublasHandle_t handle,
    std::vector<int> factors,
    size_t numelX,
    size_t B,
    torch::Tensor X,
    //const scalar_t * __restrict__ X,
    //const scalar_t * __restrict__ W,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> hidden
    ) {
  scalar_t one = 1.0;
  scalar_t zero = 0.0;
  int x_stride = numelX / B;

  scalar_t const *S = X.data_ptr<scalar_t>();


  for (int i=0; i < factors.size(); ++i){
    auto f = factors[i];
    auto T = hidden[i].data_ptr<scalar_t>();
    auto W = ws[i].data_ptr<scalar_t>();
    gemm_sb(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        x_stride / f, f, f,
        &one,
        S, f,
        x_stride,
        W, f,
        0,
        &zero,
        T, x_stride / f,
        x_stride,
        B
        );
    //w_offset += f * f;
    S = T;
  }
}

std::vector<torch::Tensor> faclin_forward(
    torch::Tensor state,
    std::vector<torch::Tensor> weights
    ){
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  size_t numelX = at::numel(state);
  size_t B = state.size(0);

  std::vector<int> factors;
  std::vector<torch::Tensor> hidden;
  factors.reserve(weights.size());
  hidden.reserve(weights.size());
  for (auto w: weights){
    factors.push_back(w.size(0));
    hidden.push_back(torch::empty_like(state));
  }

  AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "faclin_fwd", ([&] {
    //std::vector<scalar_t*> hptr;
    //std::vector<const scalar_t* __restrict__> wptr;
    //for (auto h: hidden){
    //  hptr.push_back(h.data_ptr<scalar_t>());
    //}
    //for (auto w: weights){
    //  wptr.push_back(w.data_ptr<scalar_t>());
    //}
    faclin_forward_inner<scalar_t>(
        handle,
        factors,
        numelX,
        B,
        state,
        weights,
        hidden);
  }));
  return hidden;
}

//template<typename scalar_t>
//void faclin_backward_inner(
//    cublasHandle_t handle,
//    std::vector<int> factors,
//    size_t numelX,
//    size_t B,
//    const scalar_t * __restrict__ X,
//    const scalar_t * __restrict__ W,
//    scalar_t * __restrict__ mem1,
//    scalar_t * __restrict__ mem2
//    ) {
//  scalar_t one = 1.0;
//  scalar_t zero = 0.0;
//  size_t w_offset = 0;
//
//  bool flag = true;
//  scalar_t const *S = X;
//  scalar_t *T = mem1;
//
//  for (int f: factors){
//    gemm(
//        handle,
//        CUBLAS_OP_T,
//        CUBLAS_OP_N,
//        numelX / f, f, f,
//        &one,
//        S, f,
//        W + w_offset, f,
//        &zero,
//        T, numelX / f
//        );
//    w_offset += f * f;
//    if (flag){
//      S = mem1;
//      T = mem2;
//    } else {
//      S = mem2;
//      T = mem1;
//    }
//    flag = not flag;
//  }
//
//  if (B > 1) {
//    geam(
//        handle,
//        CUBLAS_OP_T,
//        CUBLAS_OP_N,
//        numelX/B, B,
//        &one,
//        S, B,
//        &zero,
//        T, numelX/B,
//        T, numelX/B
//    );
//  }
//}
//
//std::vector<torch::Tensor> faclin_backward(
//    torch::Tensor state,
//    torch::Tensor weights,
//    std::vector<int> factors,
//    torch::Tensor grad
//    ){
//  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
//  size_t numelX = at::numel(state);
//  size_t B = state.size(0);
//
//  auto mem1 = torch::empty_like(state);
//  auto mem2 = torch::empty_like(state);
//
//  AT_DISPATCH_FLOATING_TYPES(state.scalar_type(), "faclin_bwd", ([&] {
//    faclin_backward_inner<scalar_t>(
//        handle,
//        factors,
//        numelX,
//        B,
//        grad.data_ptr<scalar_t>(),
//        weights.data_ptr<scalar_t>(),
//        mem1.data_ptr<scalar_t>(),
//        mem2.data_ptr<scalar_t>());
//  }));
//  if ((factors.size() + ((B > 1) ? 1 : 0)) % 2 == 0 ){
//    return {mem2};
//  } else {
//    return {mem1}; 
//  }
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &faclin_forward, "factorized linear forward");
  //m.def("backward", &faclin_backward, "factorized linear backward");
}
