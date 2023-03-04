
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define MAX_WSIZE 9000
#define TW 32
int is_l2=0;
__constant__ float wConst[MAX_WSIZE];


__global__ void forward_kernel_l2(const float * __restrict__ X, float * __restrict__ Y,const float * __restrict__ w,const int B, const int M, const int C, const int H, const int W, const int K) {

    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int numBRows = C*K*K;
    int numCRows = M;
    int numCColumns = H_out*W_out;

    #define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define y4d(i3, i2, i1, i0) Y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Define the subtiles of each input matrix to be in shared memory
    __shared__ float subTileA[2][TW][TW];
    __shared__ float subTileB[2][TW][TW];

    int buf_id = 0;

    // Shorthand def of blocks and threads
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Get current row/col of the element in output C. Each thread calculates an element of C
    int row = by*TW + ty;
    int col = bx*TW + tx;

    // Go through each pair of sub-tiles that make up an inner product for elements in C
    float Y_val = 0;
    int numIter=ceil((1.0*numBRows)/TW);
      #pragma unroll
    for(int m = 0; m < numIter; m++){
        // First have each thread load an element pair from inputs to sub-tiles. Only read if in bounds
        int tcol = m*TW + tx;
        int trow = m*TW + ty;

        // Get the indices for accessing W
        int W_m = row;
        int W_c = tcol / (K*K);
        int W_h = (tcol % (K*K)) / K;
        int W_w = (tcol % (K*K)) % K;

        // Get the indices for accessing X
        int X_b = blockIdx.z;
        int X_c = trow / (K*K);
        int X_p = (trow % (K*K))/K;
        int X_q = (trow % (K*K))%K;
        int X_h = col / W_out;
        int X_w = col % W_out;

        // Load elements into the sub-tiles
        if (tcol < numBRows && row < M) subTileA[buf_id][ty][tx] = w4d(W_m, W_c, W_h, W_w);
        else subTileA[buf_id][ty][tx] = 0;
        if (trow < numBRows && col < H_out*W_out) subTileB[buf_id][ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
        else subTileB[buf_id][ty][tx] = 0;

        __syncthreads(); // Wait for threads to sync before using sub-tiles

        // Calculate the sub inner product for the current tile pair.
        if(row < numCRows && col < numCColumns){
            #pragma unroll
            for(int i = 0; i < TW; i++)
                Y_val += subTileA[buf_id][ty][i]*subTileB[buf_id][i][tx];
        }
        // Switch to next buffer
        buf_id = (buf_id + 1) % 2;
    }

    // Write to Y if in bounds
    int Y_b = blockIdx.z;
    int Y_m = row;
    int Y_h = col / W_out;
    int Y_w = col % W_out;
    if((row < numCRows) && (col < numCColumns))
        y4d(Y_b, Y_m, Y_h, Y_w) = Y_val;

    #undef x4d
    #undef y4d
    #undef w4d
}

__global__ void forward_kernel_l1(const float * __restrict__ X, float * __restrict__ Y,const int B, const int M, const int C, const int H, const int W, const int K) {

    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int numBRows = C*K*K;
    int numCRows = M;
    int numCColumns = H_out*W_out;

    #define x4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define y4d(i3, i2, i1, i0) Y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define w4d(i3, i2, i1, i0) wConst[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Define the subtiles of each input matrix to be in shared memory

    __shared__ float subTileB[2][16][16];
    int buf_id = 0;

    // Shorthand def of blocks and threads
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Get current row/col of the element in output C. Each thread calculates an element of C
    int row = by*16 + ty;
    int col = bx*16 + tx;

    // Go through each pair of sub-tiles that make up an inner product for elements in C
    float Y_val = 0;
    int numIter = ceil((1.0*numBRows)/16);
    for(int m = 0; m < numIter; m++){
        #pragma unroll
        // First have each thread load an element pair from inputs to sub-tiles. Only read if in bounds
        int trow = m*16 + ty;

        // Get the indices for accessing X
        int X_b = blockIdx.z;
        int X_c = trow / (K*K);
        int X_p = (trow % (K*K))/K;
        int X_q = (trow % (K*K))%K;
        int X_h = col / W_out;
        int X_w = col % W_out;

        // Load elements into the sub-tiles
        if (trow < numBRows && col < H_out*W_out) subTileB[buf_id][ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
        else subTileB[buf_id][ty][tx] = 0;

        __syncthreads(); // Wait for threads to sync before using sub-tiles

        // Calculate the sub inner product for the current tile pair.
        if(row < numCRows && col < numCColumns){
            #pragma unroll
            for(int i = 0; i < 16; i++){
                int tcol = m*16 + i;

                // Get the indices for accessing W
                int W_m= row;
                int W_c = tcol / (K*K);
                int W_h = (tcol % (K*K)) / K;
                int W_w = (tcol % (K*K)) % K;
                if (tcol < numBRows && row < M)Y_val += w4d(W_m, W_c, W_h, W_w)*subTileB[buf_id][i][tx];
            }
        }
        // Switch to next buffer
        buf_id = (buf_id + 1) % 2;
    }

    // Write to Y if in bounds
    int Y_b = blockIdx.z;
    int Y_m = row;
    int Y_h = col / W_out;
    int Y_w = col % W_out;
    if((row < numCRows) && (col < numCColumns))
        y4d(Y_b, Y_m, Y_h, Y_w) = Y_val;

    #undef x4d
    #undef y4d
    #undef w4d
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
    int H_unroll = H_out * W_out;

    if(!is_l2){
    // Load the filter into constant memory
        int w_size = M * C * K * K * sizeof(float);
        // printf("H: %d W: %d K: %d B %d M %d C %d \n", H, W, K, B, M, C);
        dim3 DimGrid(ceil((1.0*H_unroll)/16), ceil((1.0*M)/16), B);
        dim3 DimBlock(16, 16, 1);

        int shmem_size = 4 * 16 * 16 * sizeof(float);
        MSHADOW_CUDA_CALL(cudaMemcpyToSymbol(wConst, w.dptr_,w_size));
        forward_kernel_l1<<<DimGrid, DimBlock, shmem_size>>>(x.dptr_, y.dptr_,B, M, C, H, W, K);
        is_l2=1;
    }
    else{
        dim3 DimGrid(ceil((1.0*H_unroll)/32), ceil((1.0*M)/32), B);
        dim3 DimBlock(32, 32, 1);

        int shmem_size = 2 * 32 * 32 * sizeof(float);
        forward_kernel_l2<<<DimGrid, DimBlock, shmem_size>>>(x.dptr_, y.dptr_, w.dptr_,B, M, C, H, W, K);
    }

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
