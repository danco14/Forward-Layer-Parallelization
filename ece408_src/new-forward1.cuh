
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TW 16
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil((1.0*W_out)/TW);
    int x_TW = TW + K - 1;
    extern __shared__ float shmem[];
    float* x_shared = &shmem[0];
    float* k_shared = &shmem[x_TW*x_TW];
    
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
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x2d_shared(i1, i0) x_shared[(i1) * (x_TW) + i0] 
    #define k2d_shared(i1, i0) k_shared[(i1) * (K) + i0]

    // Compute partial sum of y element
    float sum = 0;

    for(int c=0; c < C; c++){
        // Load the filter elements for each channel into shared mem
        if((h0<K) && (w0<K))
            k2d_shared(h0, w0) = k4d(m,c,h0,w0);
        

        // Load the input elements for each channel into shared mem. Strategy 1 style
        for(int i = h; i < h_base + x_TW; i += TW){
            for(int j = w; j < w_base + x_TW; j += TW){
                if((i < H) && (j < W))
                    x2d_shared(i-h_base, j-w_base) = x4d(b,c,i,j);
                else if(i-h_base < x_TW && j-w_base < x_TW)
                    x2d_shared(i-h_base, j-w_base) = 0;
            }
        }
            
        __syncthreads();

        // Calculate partial sum y element for channel c
        for(int p=0; p<K; p++)
            for(int q=0; q<K; q++)
                    sum += k2d_shared(p, q) * x2d_shared(h0+p, w0+q);

        __syncthreads();
    }

    if((h < H_out) && (w < W_out))
        y4d(b,m,h,w) = sum;



    #undef y4d
    #undef x4d
    #undef k4d
    #undef x2d_shared
    #undef k2d_shared
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

    // Get the size of shared memory
    int x_TW = TW + K - 1;
    int shmem_size = sizeof(float) * (x_TW*x_TW + K*K);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // printf("H: %d W: %d K: %d \n", H, W, K);

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