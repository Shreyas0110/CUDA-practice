#include "matmul.cu"
#include <fstream>
#include <iostream>

#define WIDTH 256
#define SIZE (WIDTH*WIDTH)

int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *a, *b, *out, *dev_a, *dev_b, *dev_out;
    a = (float *) malloc(SIZE * sizeof(float));
    b = (float *) malloc(SIZE * sizeof(float));
    out = (float *) malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; ++i){
        a[i] = i;
        b[i] = i;
    }

    cudaMalloc((void **) &dev_a, SIZE * sizeof(float));
    cudaMalloc((void **) &dev_b, SIZE * sizeof(float));
    cudaMalloc((void **) &dev_out, SIZE * sizeof(float));

    cudaMemcpy(dev_a, a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32, 1);
    int gridSize = (WIDTH+31)/32; 
    std::cout << gridSize << std::endl;
    dim3 dimGrid(gridSize, gridSize, 1);

    naiveMatMulKernel <<<dimGrid, dimBlock>>> (dev_a, dev_b, dev_out, WIDTH);

    cudaMemcpy(out, dev_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_out);
    free(a);
    free(b);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "Time Taken: " << time << std::endl;

    std::fstream outfile;
    outfile.open("mymatmul.txt", std::fstream::out);
    for (int i = 0; i < SIZE; i++){
        if (i % WIDTH == 0 && i != 0)
            outfile << "\n";
        outfile << out[i] << " ";
    }
    outfile.close();
    free(out);
}