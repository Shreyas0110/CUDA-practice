
/* Assume Square Matix */
__global__ void naiveMatMulKernel(float* A, float* B, float* out, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width){
        float val = 0;
        for (int i = 0; i < width; i++){
            val += A[row * width + i] * B[width * i + col];
        }
        out[row * width + col] = val;
    }
}


/* Think Tile Width must be same as sqrt of threads in block */
#define TILE_WIDTH 32
__global__ void tilingMatMulKernel(float* A, float* B, float* out, int width){

    __shared__ float A_shared [TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared [TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    float val = 0;

    for (int tile_index = 0; tile_index < width/TILE_WIDTH; tile_index++){
        A_shared [ty][tx] = A [row * width + tile_index*TILE_WIDTH + tx];
        B_shared [ty][tx] = B [(tile_index*TILE_WIDTH + ty)*width + col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++){
            val += A_shared[ty][i] * B_shared[i][tx]; 
        }
        __syncthreads();
    }
    out[row * width + col] = val;
}