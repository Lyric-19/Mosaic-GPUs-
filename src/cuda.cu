#include "cuda.cuh"
#include "cuda_runtime.h"
#include <cstring>
#include "device_launch_parameters.h"
#include "helper.h"
#include <stdlib.h>
#include <string.h>

void checkCUDAError(const char*);
///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;

int W, C;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    W = input_image->width;
    C = input_image->channels;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));

    // CUDA_CALL(cudaMalloc(&d_whole_image_sum, input_image->channels * sizeof(unsigned long long)));
}


__global__ void mosaic_sum(int W, int C, unsigned char* input_image_data, unsigned long long* d_mosaic_sum) {
    /*
    //  sum the row first
    int row_r = 0, row_g = 0, row_b = 0;
    int t_index = (blockIdx.y * W / TILE_SIZE + blockIdx.x) * C;
    int t_offset_index = (blockIdx.y * W * TILE_SIZE + blockIdx.x * TILE_SIZE + threadIdx.x * W) * C;
    // int p_offset_index = (threadIdx.y * TILE_SIZE + threadIdx.x ) * parameter[2];
    
    if (threadIdx.y == 0) {
        for (int i = 0; i < TILE_SIZE; i++) {
            row_r += input_image_data[t_offset_index + i * 3];
        }
        atomicAdd(&d_mosaic_sum[t_index], row_r);
        // __syncthreads();
    }

    if (threadIdx.y == 1) {
        for (int i = 0; i < TILE_SIZE; i++) {
            row_g += input_image_data[t_offset_index + i * 3 + 1];
        }
        atomicAdd(&d_mosaic_sum[t_index + 1], row_g);
        // __syncthreads();
    }

    if (threadIdx.y == 2) {
        for (int i = 0; i < TILE_SIZE; i++) {
            row_b += input_image_data[t_offset_index + i * 3 + 2];
        }
        atomicAdd(&d_mosaic_sum[t_index + 2], row_b);
        // __syncthreads();
    }
    */
    
    // suffle
    int t_index = (blockIdx.y * W / TILE_SIZE + blockIdx.x) * C;
    int t_offset_index = (blockIdx.y * W * TILE_SIZE + blockIdx.x * TILE_SIZE) * C;
    int p_index = (threadIdx.y * W + threadIdx.x) * C;
    
     unsigned long long row_r = input_image_data[t_offset_index + p_index];
    unsigned long long row_g = input_image_data[t_offset_index + p_index + 1];
    unsigned long long row_b = input_image_data[t_offset_index + p_index + 2];

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        row_r += __shfl_down(row_r, offset);
        row_g += __shfl_down(row_g, offset);
        row_b += __shfl_down(row_b, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&d_mosaic_sum[t_index], row_r);
        atomicAdd(&d_mosaic_sum[t_index + 1], row_g);
        atomicAdd(&d_mosaic_sum[t_index + 2], row_b);

    }
    
}


 void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);

     dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
     // dim3 threadsPerBlock(TILE_SIZE, 3, 1);   // add over rows
     dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);  // shuffle
     mosaic_sum << <blocksPerGrid, threadsPerBlock >> > (W, C, d_input_image_data, d_mosaic_sum);
     cudaDeviceSynchronize();
     checkCUDAError("Stage1");


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned long long* cuda_mosaic_sum;
    cuda_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    CUDA_CALL(cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    validate_tile_sum(&cuda_input_image, cuda_mosaic_sum);
    free(cuda_mosaic_sum);
#endif
}

 __global__ void mosaic_ave(int W, int C, unsigned long long* d_mosaic_sum, unsigned char* d_mosaic_value, unsigned long long* d_global_pixel_sum) {

     // unsigned int cuda_TILES_X, cuda_TILES_Y;
     // cuda_TILES_X = cuda_input_image.width / TILE_SIZE;
     // cuda_TILES_Y = cuda_input_image.height / TILE_SIZE;

     int row_sum_r = 0, row_sum_g = 0, row_sum_b = 0;
     int x = blockIdx.x;
     int thread_index = (threadIdx.x * W / TILE_SIZE + blockIdx.x * W) * C;
     if (threadIdx.x < W / TILE_SIZE) {
         for (int i = 0; i < W / TILE_SIZE; ++i) {
             d_mosaic_value[thread_index + i * 3] = (unsigned char)(d_mosaic_sum[thread_index + i * 3] / TILE_PIXELS);
             d_mosaic_value[thread_index + i * 3 + 1] = (unsigned char)(d_mosaic_sum[thread_index + i * 3 + 1] / TILE_PIXELS);
             d_mosaic_value[thread_index + i * 3 + 2] = (unsigned char)(d_mosaic_sum[thread_index + i * 3 + 2] / TILE_PIXELS);

             row_sum_r += d_mosaic_value[thread_index + i * 3];
             row_sum_g += d_mosaic_value[thread_index + i * 3 + 1];
             row_sum_b += d_mosaic_value[thread_index + i * 3 + 2];
         }


         atomicAdd(&d_global_pixel_sum[0], row_sum_r);
         atomicAdd(&d_global_pixel_sum[1], row_sum_g);
         atomicAdd(&d_global_pixel_sum[2], row_sum_b);
     }

     //__syncthreads();
     
 }

    
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, d_mosaic_sum, compact_mosaic, global_pixel_average);
    unsigned long long whole_image_sum[4] = { 0,0,0,0 };

    int block_num = (int)(cuda_TILES_X / TILE_SIZE);

    if (block_num == 0) {
        block_num = 1;
    }
    dim3 blocksPerGrid(block_num, 1, 1);
    dim3 threadsPerBlock(TILE_SIZE, 1, 1);
    mosaic_ave << <blocksPerGrid, threadsPerBlock >> > (W, C, d_mosaic_sum, d_mosaic_value, d_global_pixel_sum);
    cudaDeviceSynchronize();
    checkCUDAError("Stage2");
    CUDA_CALL(cudaMemcpy(&whole_image_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    output_global_average[0] = (unsigned char)(whole_image_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[1] = (unsigned char)(whole_image_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[2] = (unsigned char)(whole_image_sum[2] / (cuda_TILES_X * cuda_TILES_Y));

    CUDA_CALL(cudaFree(d_global_pixel_sum));

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned char* cuda_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(cuda_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    unsigned long long* cuda_mosaic_sum;
    cuda_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    CUDA_CALL(cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, cuda_mosaic_sum, cuda_mosaic_value, output_global_average);
    free(cuda_mosaic_value);
    free(cuda_mosaic_sum);
#endif    
}

__global__ void mosaic_broad(int W, int C, unsigned char* d_mosaic_value, unsigned char* d_output_image_data) {

    __shared__ unsigned char value_r, value_g, value_b;
    int t_index = (blockIdx.y * W / TILE_SIZE + blockIdx.x) * C;
    int t_offset_index = (blockIdx.y * W * TILE_SIZE + blockIdx.x * TILE_SIZE) * C;
    int p_offset_index = (threadIdx.y * W + threadIdx.x) * C;


    value_r = d_mosaic_value[t_index];
    value_g = d_mosaic_value[t_index + 1];
    value_b = d_mosaic_value[t_index + 2];

    d_output_image_data[t_offset_index + p_offset_index] = value_r;
    d_output_image_data[t_offset_index + p_offset_index + 1] = value_g;
    d_output_image_data[t_offset_index + p_offset_index + 2] = value_b;
    __syncthreads();
}

void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);

    dim3 blocksPerGrid2(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock2(TILE_SIZE, TILE_SIZE, 1);
    mosaic_broad << <blocksPerGrid2, threadsPerBlock2 >> > (W, C, d_mosaic_value, d_output_image_data);
    cudaDeviceSynchronize();
    checkCUDAError("Stage3");

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    Image cuda_output_image = cuda_input_image;
    cuda_output_image.data = (unsigned char*)malloc(cuda_input_image.width * cuda_input_image.height * cuda_input_image.channels * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(cuda_output_image.data, d_output_image_data, cuda_input_image.width * cuda_input_image.height * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    unsigned char* cuda_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(cuda_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    validate_broadcast(&cuda_input_image, cuda_mosaic_value, &cuda_output_image);
    free(cuda_mosaic_value);
    free(cuda_output_image.data);
#endif    
}
void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
}

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr,"CUDA ERROR: %s: %s. \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
