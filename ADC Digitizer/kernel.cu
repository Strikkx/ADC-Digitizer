#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <time.h>


#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 512


// Generate random float numbers
float* generateData(int n) {
    float* darr = (float*)malloc(sizeof(float) * n);

    srand(time(0));
    for (int i = 0; i < n; i++) {
        darr[i] = ((float)rand() / RAND_MAX) * 1000.0f;  // Generate random float between 0 and 1000
    }

    return darr;
}


__global__ void subtractionKernel(const float* data, int size, float* result) {
    __shared__ float shared_data_sub[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i + 3 < size) {
        shared_data_sub[tid] = (data[i] - data[i + 2]);
    }
    else {
        shared_data_sub[tid] = 0.0f;
    }

    __syncthreads();

    // Warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data_sub[tid] += shared_data_sub[tid + s];
        }
        __syncthreads();
    }

    // Write the result for each block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data_sub[0];
    }

}__global__ void multiplicationKernel(const float* data, int size, float* result) {
    __shared__ float shared_data_mult[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i + 3 < size) {
        shared_data_mult[tid] = (data[i] - data[i + 2]) * (data[i + 1] - data[i + 3]);
    }
    else {
        shared_data_mult[tid] = 0.0f;
    }

    __syncthreads();

    // Warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data_mult[tid] += shared_data_mult[tid + s];
        }
        __syncthreads();
    }

    // Write the result for each block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data_mult[0];
    }
}

__global__ void summationKernel(const float* data, int size, float* result) {
    __shared__ float shared_data_sum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i + 3 < size) {
        shared_data_sum[tid] = (data[i] - data[i + 2]) * (data[i + 1] - data[i + 3]);
    }
    else {
        shared_data_sum[tid] = 0.0f;
    }

    __syncthreads();

    // Warp-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data_sum[tid] += shared_data_sum[tid + s];
        }
        __syncthreads();
    }

    // Write the result for each block to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_data_sum[0];
    }
}



int main() {

    // Generate random data
    const int n = 1000000;
    float* inputArray = generateData(n);
    size_t size = n * sizeof(float);

    // Calculate total blocks
    int blocks_sum = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE / 3;
    int blocks_sub = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE / 3;
    int blocks_mult = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE / 3;

    // Allocate GPU memory
    float* d_input_sum, * d_output_sum, * d_input_sub, * d_output_sub, * d_input_mult, * d_output_mult;
    cudaMalloc((void**)&d_input_sum, size);
    cudaMalloc((void**)&d_output_sum, blocks_sum * sizeof(float));

    cudaMalloc((void**)&d_input_sub, size);
    cudaMalloc((void**)&d_output_sub, blocks_sum * sizeof(float));

    cudaMalloc((void**)&d_input_mult, size);
    cudaMalloc((void**)&d_output_mult, blocks_sum * sizeof(float));

    // Calculate the time
    cudaEvent_t start_sum, stop_sum, start_mult, stop_mult, start_sub, stop_sub;
    cudaEventCreate(&start_sum);
    cudaEventCreate(&stop_sum);

    cudaEventCreate(&start_sub);
    cudaEventCreate(&stop_sub);

    cudaEventCreate(&start_mult);
    cudaEventCreate(&stop_mult);

    // Copy data to device
    cudaMemcpy(d_input_sum, inputArray, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_sub, inputArray, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_mult, inputArray, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    cudaEventRecord(start_sum);
    summationKernel <<<blocks_sum, BLOCK_SIZE>>> (d_input_sum, size, d_output_sum);
    cudaEventRecord(stop_sum);

    cudaEventRecord(start_sub);
    subtractionKernel <<<blocks_sub, BLOCK_SIZE >>> (d_input_sub, size, d_output_sub);
    cudaEventRecord(stop_sub);

    cudaEventRecord(start_mult);
    multiplicationKernel <<<blocks_mult, BLOCK_SIZE >>> (d_input_mult, size, d_output_mult);
    cudaEventRecord(stop_mult);


    // Copy the result from device to host
    float* result_sum = (float*)malloc(blocks_sum * sizeof(float));
    float* result_sub = (float*)malloc(blocks_sub * sizeof(float));
    float* result_mult = (float*)malloc(blocks_mult * sizeof(float));

    cudaMemcpy(result_sum, d_output_sum, blocks_sum * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_sub, d_output_sub, blocks_sub * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_mult, d_output_mult, blocks_mult * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_sum);
    cudaEventSynchronize(stop_sub);
    cudaEventSynchronize(stop_mult);


    float milliseconds_sum = 0;
    float milliseconds_sub = 0;
    float milliseconds_mult = 0;
    float milliseconds_total = 0;

    cudaEventElapsedTime(&milliseconds_sum, start_sum, stop_sum);
    cudaEventElapsedTime(&milliseconds_sub, start_sub, stop_sub);
    cudaEventElapsedTime(&milliseconds_mult, start_mult, stop_mult);

    milliseconds_total = milliseconds_sum + milliseconds_sub + milliseconds_mult;

    // Sum the partial results from each block
    float final_result = 0.0f;
    for (int i = 0; i < blocks_sum; i++) {
        final_result += result_sum[i];
    }
    // Print time taken for each kernel
    printf("Summation took %f milliseconds to execute \n", milliseconds_sum);
    printf("Subtraction took %f milliseconds to execute \n", milliseconds_sub);
    printf("Multiplication took %f milliseconds to execute \n", milliseconds_mult);
    printf("All 3 kernels took %f milliseconds to execute \n", milliseconds_total);


    printf("Final result: %f\n", final_result);

    // Free device memory
    cudaFree(d_input_sum);
    cudaFree(d_output_sum);
    free(result_sum);

    cudaFree(d_input_sub);
    cudaFree(d_output_sub);
    free(result_sub);

    cudaFree(d_input_mult);
    cudaFree(d_output_mult);
    free(result_mult);

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
