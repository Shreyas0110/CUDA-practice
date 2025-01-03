
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