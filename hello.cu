#include <stdio.h>
#include "hello.h"


__global__ void cuda_hello()
{
    printf("Hello World from GPU!\n");
}


void say_hello(){
    cuda_hello<<<10, 100>>>();

    cudaDeviceSynchronize();
}
