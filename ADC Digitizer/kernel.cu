#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>


#define BLOCK_SIZE 512

/*
* Generate random float numbers
*/
float* generateData(int n) {
    float* darr = (float*)malloc(sizeof(float) * n);

    srand(time(0));
    for (int i = 0; i < n; i++) {
        darr[i] = ((float)rand() / RAND_MAX) * 1000.0f;  // Generate random float between 0 and 1000
    }

    return darr;
}

/*
* Summation Kernel
*/
__global__ void summationKernel(const float* data, int size, float* result) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i + 3 < size) {
        shared_data[tid] = (data[i] - data[i + 2]) * (data[i + 1] - data[i + 3]);
    }
    else {
        shared_data[tid] = 0.0f;
    }

    //shared_data[tid] = (data[i] - data[i + 2]) * (data[i + 1] - data[i + 3]);

    __syncthreads();

    // Perform warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write the result for each block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

/*
* Main program
*/
int main() {

    // Generate random data
    const int n = 10000000000000;    
    float* inputArray = generateData(n);
    size_t size = n * sizeof(float);

    // Calculate total blocks
    int blocks = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //int blocks = 256;

    // Allocate GPU memory
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, blocks * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, inputArray, size , cudaMemcpyHostToDevice);

    // Calculate the time
    clock_t t;
    t = clock();

    // Launch the kernel
    summationKernel <<<blocks, BLOCK_SIZE >>> (d_input, size, d_output);

    // Copy the result from device to host
    float* result = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(result, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum the partial results from each block
    float final_result = 0.0f;
    for (int i = 0; i < blocks; i++) {
        final_result += result[i];
    }

    t = clock() - t; // End the clock
    double time_taken = ((double)t) / CLOCKS_PER_SEC * 1000; // Convert to seconds
    printf("fun() took %f milliseconds to execute \n", time_taken);

    printf("Final result: %f\n", final_result);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(result);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}