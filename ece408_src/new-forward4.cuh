
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TW 16
#define MAX_WSIZE 12000

__constant__ float wConst[MAX_WSIZE];

__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    const int H_out = H - K + 1; // 66
    const int W_out = W - K + 1; // 66

    int W_grid = ceil((1.0*W_out)/TW); // 5
    int x_TW = TW + K - 1; // 20
    __shared__ float shmem[1000];

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z / W_grid) * TW;
    int w_base = (blockIdx.z % W_grid) * TW;
    int h = h_base + h0;
    int w = w_base + w0;


    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) wConst[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x2d_shared(i1, i0) shmem[(i1) * (x_TW) + i0]

    // Compute partial sum of y element
    float sum = 0;

    for(int c=0; c < C; c++){

      // Load the input elements for each channel into shared mem. Strategy 1 style
      for(int i = h; i < h_base + x_TW; i += TW){
        for(int j = w; j < w_base + x_TW; j += TW){
          x2d_shared(i-h_base, j-w_base) = x4d(b,c,i,j);
        }
      }

      __syncthreads();

      // Calculate partial sum y element for channel c
      #pragma unroll 5
      for(int p=0; p<K; p++){
        sum += k4d(m,c,p,0) * x2d_shared(h0+p, w0+0);
        sum += k4d(m,c,p,1) * x2d_shared(h0+p, w0+1);
        sum += k4d(m,c,p,2) * x2d_shared(h0+p, w0+2);
        sum += k4d(m,c,p,3) * x2d_shared(h0+p, w0+3);
        sum += k4d(m,c,p,4) * x2d_shared(h0+p, w0+4);
      }

    }

    if((h < H_out) && (w < W_out))
        y4d(b,m,h,w) = sum;

    #undef y4d
    #undef x4d
    #undef k4d
    #undef x2d_shared
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Set the kernel dimensions
    int W_grid = ceil((1.0*W_out)/TW);
    int H_grid = ceil((1.0*H_out)/TW);
    dim3 blockDim(TW,TW,1);
    dim3 gridDim(B,M,H_grid*W_grid);

    // Call the kernel

    // Load the filter into constant memory
    int w_size = M * C * K * K * sizeof(float);
    cudaMemcpyToSymbol(wConst, w.dptr_,w_size, 0, cudaMemcpyDeviceToDevice);
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
