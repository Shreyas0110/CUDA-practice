#include <fstream>
#include <iostream>

#define WIDTH 256
#define SIZE (WIDTH*WIDTH)

void multiplySquareMatrices(const float* matA, const float* matB, float* result, int N) {
    // Initialize result matrix with zeros
    for (int i = 0; i < N * N; ++i) {
        result[i] = 0;
    }

    // Perform matrix multiplication
    for (int i = 0; i < N; ++i) {           // Row of matA
        for (int j = 0; j < N; ++j) {       // Column of matB
            for (int k = 0; k < N; ++k) {   // Element in row/column
                result[i * N + j] += matA[i * N + k] * matB[k * N + j];
            }
        }
    }
}

int main(){

    float *a, *b, *out;
    a = (float *) malloc(SIZE * sizeof(float));
    b = (float *) malloc(SIZE * sizeof(float));
    out = (float *) malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; ++i){
        a[i] = i;
        b[i] = i;
    }

    multiplySquareMatrices(a, b, out, WIDTH);

    free(a);
    free(b);

    std::fstream outfile;
    outfile.open("c++matmul.txt", std::fstream::out);
    for (int i = 0; i < SIZE; i++){
        if (i % WIDTH == 0 && i != 0)
            outfile << "\n";
        outfile << out[i] << " ";
    }
    outfile.close();
}