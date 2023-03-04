
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TW 32
#define MAX_NUM_THREADS 1024
#define MAX_WSIZE 12000

__constant__ float wConst[MAX_WSIZE];

__global__ void unroll_kernel(int C, int H, int W, int K, float* X, float* X_unroll){

    int c, row_out, col_out, row_unroll, col_unroll, row_base;
    int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    #define x3d(i2, i1, i0) X[(i2) * (H * W) + (i1) * (W) + i0]
    #define xu2d(i1, i0) X_unroll[(i1) * (W_unroll) + i0]

    if(t < C*W_unroll){
        c = t/W_unroll;
        row_out = (t % W_unroll) / W_out;
        col_out = (t % W_unroll) % W_out;
        col_unroll = row_out * W_out + col_out;
        row_base = c * K * K;

        for(int p=0; p < K; p++)
            for(int q=0; q < K; q++){
                row_unroll = row_base + p*K + q;
                xu2d(row_unroll, col_unroll) = x3d(c, row_out+p, col_out+q);
            }
    }

    #undef x3d
    #undef x2d
}


void unroll_gpu(int C, int H, int W, int K, float* X, float* X_unroll){
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int num_threads = C * H_out * W_out;

    int num_blocks = ceil((num_threads)/(1.0*MAX_NUM_THREADS));
    unroll_kernel<<<num_blocks, MAX_NUM_THREADS>>>(C, H, W, K, X, X_unroll);
}


__global__ void tile_mm(float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    // Define the subtiles of each input matrix to be in shared memory
    __shared__ float subTileA[TW][TW];
    __shared__ float subTileB[TW][TW];

    // Shorthand def of blocks and threads
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Get current row/col of the element in output C. Each thread calculates an element of C
    int row = by*TW + ty;
    int col = bx*TW + tx;

    // Go through each pair of sub-tiles that make up an inner product for elements in C
    float C_val = 0;
    for(int m = 0; m < ceil((1.0*numBRows)/TW); m++){
        // First have each thread load an element pair from inputs to sub-tiles. Only read if in bounds
        int AElemPos = row*numBRows + m*TW + tx;
        int BElemPos = (m*TW + ty)*numBColumns + col;

        if (AElemPos < numAColumns*numARows) subTileA[ty][tx] = wConst[AElemPos];
        else subTileA[ty][tx] = 0;
        if (BElemPos < numBColumns*numBRows) subTileB[ty][tx] = B[BElemPos];
        else subTileB[ty][tx] = 0;

        __syncthreads(); // Wait for threads to sync before using sub-tiles

        // Calculate the sub inner product for the current tile pair.
        // Edge Case: For matrix dims that are not multiples of TW, sub inner product is less than TW.
        if(row < numCRows && col < numCColumns){
            for(int i = 0; i < TW; i++)
            C_val += subTileA[ty][i]*subTileB[i][tx];  
        }
        __syncthreads(); // Sync all threads before loading the next sub-tiles
    }

    // Write to C if in bounds
    if((row < numCRows) && (col < numCColumns))
        C[row*numCColumns + col] = C_val;
}

void gemm(float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {
        
        dim3 DimGrid(ceil((1.0*numCColumns)/TW), ceil((1.0*numCRows)/TW), 1);
        dim3 DimBlock(TW, TW, 1);

        int shmem_size = 2 * TW * TW;
        tile_mm<<<DimGrid, DimBlock, shmem_size>>>(B, C,
                                                  numARows, numAColumns,
                                                  numBRows, numBColumns,
                                                  numCRows, numCColumns);


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
    int W_unroll = C * K * K;
    
    // Allocate global memory for unrolled input
    float *x_unroll;
    MSHADOW_CUDA_CALL(cudaMalloc(&x_unroll, W_unroll * H_unroll * sizeof(float)));
    
    // // Load the filter into constant memory
    int w_size = M * C * K * K * sizeof(float);
    MSHADOW_CUDA_CALL(cudaMemcpyToSymbol(wConst, w.dptr_,w_size));
    
    // printf("H: %d W: %d K: %d B %d M %d C %d \n", H, W, K, B, M, C);

    for(int b=0; b < B; b++){
        float *x_b = &x.dptr_[b*(C * H * W)];
        float *y_b = &y.dptr_[b*(M * H_out * W_out)];
        unroll_gpu(C, H, W, K, x_b, x_unroll);
        gemm(x_unroll, y_b, M, W_unroll, W_unroll, H_unroll, M, H_unroll);
        // mshadow::expr::BLASEngine<gpu, float>::gemm(y.stream_, false, false, M, H_unroll, W_unroll, 1.0, w.dptr_, M, x_unroll, W_unroll, 0, y_b, M);
    }



    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    MSHADOW_CUDA_CALL(cudaFree(x_unroll));
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