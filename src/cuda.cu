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


void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

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


__global__ void mosaic_sum(Image cuda_input_image, unsigned char* input_image_data, unsigned long long* d_mosaic_sum) {

    // register int row_r = 0, row_g = 0, row_b = 0;

    int t_index = (blockIdx.y * blockDim.x + blockIdx.x) * cuda_input_image.channels;
    int t_offset_index = (blockIdx.y * blockDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * cuda_input_image.channels;
    int p_offset_index = (threadIdx.y * TILE_SIZE + threadIdx.x ) * cuda_input_image.channels;
    
    /*for (int i = 0; i < TILE_SIZE; i++) {
        row_r += input_image_data[t_offset_index + i*3];
        row_g += input_image_data[t_offset_index + i*3 + 1];
        row_b += input_image_data[t_offset_index + i*3 + 2];
    }
    */

    const unsigned char pixel1 = input_image_data[t_offset_index + p_offset_index];
    const unsigned char pixel2 = input_image_data[t_offset_index + p_offset_index + 1];
    const unsigned char pixel3 = input_image_data[t_offset_index + p_offset_index + 2];

    // atomicAdd(&d_mosaic_sum[t_index], row_r);
    // atomicAdd(&d_mosaic_sum[t_index + 1], row_g);
    // atomicAdd(&d_mosaic_sum[t_index + 2], row_b);

    d_mosaic_sum[t_index] += pixel1;
    d_mosaic_sum[t_index + 1] += pixel2;
    d_mosaic_sum[t_index + 2] += pixel3;
    __syncthreads();

}


 void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);

    dim3 blocksPerGrid1(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock1(TILE_SIZE, TILE_SIZE, 1);
    mosaic_sum << <blocksPerGrid1, threadsPerBlock1 >> > (cuda_input_image, d_input_image_data, d_mosaic_sum);
    cudaDeviceSynchronize();
    checkCUDAError("Stage1");

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned long long* cuda_mosaic_sum;
    cuda_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    CUDA_CALL(cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    validate_tile_sum(&cuda_input_image, cuda_mosaic_sum);
    free(cuda_mosaic_sum);
#endif
}

 __global__ void mosaic_ave(Image cuda_input_image, unsigned long long* d_mosaic_sum, unsigned char* d_mosaic_value, unsigned long long* d_global_pixel_sum) {

     int t_index = (blockIdx.y * blockDim.x + blockIdx.x) * cuda_input_image.channels;

     if (threadIdx.x == 0) {
        d_mosaic_value[t_index] = (unsigned char)(d_mosaic_sum[t_index] / TILE_PIXELS);
        d_mosaic_value[t_index + 1] = (unsigned char)(d_mosaic_sum[t_index + 1] / TILE_PIXELS);
        d_mosaic_value[t_index + 2] = (unsigned char)(d_mosaic_sum[t_index + 2] / TILE_PIXELS);

        atomicAdd(&d_global_pixel_sum[0], d_mosaic_value[t_index]);
        atomicAdd(&d_global_pixel_sum[1], d_mosaic_value[t_index + 1]);
        atomicAdd(&d_global_pixel_sum[2], d_mosaic_value[t_index + 2]);
        // d_global_pixel_sum[0] += d_mosaic_value[t_index];
        // d_global_pixel_sum[1] += d_mosaic_value[t_index + 1];
        // d_global_pixel_sum[2] += d_mosaic_value[t_index + 2];
        
            
     }

     __syncthreads();
 }
     //__shared__ unsigned long long whole_image_sum[4];
     // unsigned int cuda_TILES_X, cuda_TILES_Y;
     // cuda_TILES_X = cuda_input_image.width / TILE_SIZE;
     // cuda_TILES_Y = cuda_input_image.height / TILE_SIZE;

     
     // int t_offset_index = (blockIdx.y * blockDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * cuda_input_image.channels;

     // d_mosaic_value[t_index] = (unsigned char)(d_mosaic_sum[t_index] / TILE_PIXELS);
     // d_mosaic_value[t_index + 1] = (unsigned char)(d_mosaic_sum[t_index + 1] / TILE_PIXELS);
     // d_mosaic_value[t_index + 2] = (unsigned char)(d_mosaic_sum[t_index + 2] / TILE_PIXELS);
   
     // d_global_pixel_sum[0] += d_mosaic_value[t_index];
     // d_global_pixel_sum[1] += d_mosaic_value[t_index + 1];
     // d_global_pixel_sum[2] += d_mosaic_value[t_index + 2];
   
    //  __syncthreads();

     // d_output_global_average[0] = (unsigned char)(whole_image_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
     // d_output_global_average[1] = (unsigned char)(whole_image_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
     // d_output_global_average[2] = (unsigned char)(whole_image_sum[2] / (cuda_TILES_X * cuda_TILES_Y));
 



void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, d_mosaic_sum, compact_mosaic, global_pixel_average);
    // CUDA_CALL(cudaMalloc(&d_whole_image_sum, 4 * sizeof(unsigned long long)));
    unsigned long long whole_image_sum[4] = { 0,0,0,0 };
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock(TILE_SIZE, 1, 1);
    mosaic_ave << <blocksPerGrid, threadsPerBlock >> > (cuda_input_image, d_mosaic_sum, d_mosaic_value, d_global_pixel_sum);
    cudaDeviceSynchronize();
    checkCUDAError("Stage2");
    
    CUDA_CALL(cudaMemcpy(&whole_image_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    output_global_average[0] = (unsigned char)(whole_image_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[1] = (unsigned char)(whole_image_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[2] = (unsigned char)(whole_image_sum[2] / (cuda_TILES_X * cuda_TILES_Y));


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

__global__ void mosaic_broad(Image cuda_input_image, unsigned char* d_mosaic_value, unsigned char* d_output_image_data) {

    
    int t_index = (blockIdx.y * blockDim.x + blockIdx.x) * cuda_input_image.channels;
    int t_offset_index = (blockIdx.y * blockDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * cuda_input_image.channels;
    int p_offset_index = (threadIdx.y * TILE_SIZE + threadIdx.x) * cuda_input_image.channels;

    d_output_image_data[t_offset_index + p_offset_index] = d_mosaic_value[t_index];
    d_output_image_data[t_offset_index + p_offset_index + 1] = d_mosaic_value[t_index + 1];
    d_output_image_data[t_offset_index + p_offset_index + 2] = d_mosaic_value[t_index + 2];
   // __syncthreads();
}

void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
    mosaic_broad << <blocksPerGrid, threadsPerBlock >> > (cuda_input_image, d_mosaic_value, d_output_image_data);
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