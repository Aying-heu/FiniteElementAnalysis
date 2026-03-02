// test_cuda.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU\n");
}

int main0() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}