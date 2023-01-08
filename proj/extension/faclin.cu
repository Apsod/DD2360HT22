#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>

torch::Tensor faclin(
    torch::Tensor state,
    torch::Tensor weights,
    std::vector<int> factors
    ){
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  size_t numelX = at::numel(state);
  size_t w_offset = 0;
  size_t B = numelX;
  auto mem1 = torch::empty_like(state);
  auto mem2 = torch::empty_like(state);

  float one = 1.0;
  float zero = 0.0;

  torch::Tensor *S = &state;
  torch::Tensor *T = &mem1;
  bool flag = true;
  for (int f: factors){
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        numelX / f, f,  f,
        &one,
        (float*) S->data_ptr(), f,
        ((float*) weights.data_ptr()) + w_offset, f,
        &zero,
        (float*) T->data_ptr(), numelX / f
    );
    w_offset += f * f;
    B /= f;
    if (flag){
      S = &mem1;
      T = &mem2;
    } else {
      S = &mem2;
      T = &mem1;
    }
    flag = not flag;
  }
  if (B > 1) {
    cublasSgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        numelX/B, B,
        &one,
        (float*) S->data_ptr(), B,
        &zero,
        (float*) T->data_ptr(), numelX/B,
        (float*) T->data_ptr(), numelX/B
    );
  } else {
    T = S;
  }
  return *T;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &faclin, "factorize linear forward");
}
