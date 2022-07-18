#include "openmp.h"
#include "helper.h"
#include <stdlib.h>
#include <string.h>

Image openmp_input_image;
Image openmp_output_image;
unsigned int openmp_TILES_X, openmp_TILES_Y;
unsigned long long* openmp_mosaic_sum;
unsigned char* openmp_mosaic_value;
int max_threads;

void openmp_begin(const Image *input_image) {
    openmp_TILES_X = input_image->width / TILE_SIZE;
    openmp_TILES_Y = input_image->height / TILE_SIZE;
    // Allocate buffer for calculating the sum of each tile mosaic
    openmp_mosaic_sum = (unsigned long long*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    openmp_mosaic_value = (unsigned char*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    openmp_input_image = *input_image;
    openmp_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(openmp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    openmp_output_image = *input_image;
    openmp_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

}

void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    memset(openmp_mosaic_sum, 0, openmp_TILES_X * openmp_TILES_Y * openmp_input_image.channels * sizeof(unsigned long long));
    // Sum pixel data within each tile
    int i, p;
    unsigned int tile_index, tile_offset, pixel_offset;
    // unsigned char pixel1, pixel2, pixel3;
    max_threads = omp_get_num_procs();
    omp_set_num_threads(2 * max_threads - 1);
    

#pragma omp parallel for default(none) private(i, p,tile_index,tile_offset) shared(openmp_TILES_X,openmp_TILES_Y,openmp_mosaic_sum,openmp_input_image) schedule(static)
    for ( i = 0; i < openmp_TILES_X * openmp_TILES_Y; ++i) {
        tile_index = i * openmp_input_image.channels;
        tile_offset = ((int)(i/ openmp_TILES_X)* openmp_TILES_X * TILE_PIXELS + (i % openmp_TILES_X)*TILE_SIZE) * openmp_input_image.channels;
        // For each pixel within the tile
        
        for (p = 0; p < TILE_PIXELS ; ++p) {
                // For each colour channel
                const unsigned int pixel_offset = ((int)(p / TILE_SIZE) * openmp_input_image.width + p % TILE_SIZE) * openmp_input_image.channels;
                
                openmp_mosaic_sum[tile_index] += openmp_input_image.data[tile_offset + pixel_offset];
                openmp_mosaic_sum[tile_index + 1] += openmp_input_image.data[tile_offset + pixel_offset + 1];
                openmp_mosaic_sum[tile_index + 2] += openmp_input_image.data[tile_offset + pixel_offset + 2];
        }
        
            
        
    }

    // skip_tile_sum(&openmp_input_image, openmp_mosaic_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&openmp_input_image, openmp_mosaic_sum);
#endif
}


void openmp_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    max_threads = omp_get_num_procs();
    omp_set_num_threads(2 * max_threads - 1);
  
    // Calculate the average of each tile, and sum these to produce a whole image average.
    // unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    
    unsigned long long image_sum_r = 0, image_sum_g = 0, image_sum_b = 0;
    
    int t;
#pragma omp parallel for collapse(2) default(none) private(t) shared(openmp_TILES_X,openmp_TILES_Y,openmp_mosaic_sum,output_global_average,openmp_input_image,openmp_mosaic_value) reduction(+: image_sum_r,image_sum_g,image_sum_b) schedule(static)
    for (t = 0; t < openmp_TILES_X * openmp_TILES_Y; ++t) {
        openmp_mosaic_value[t * openmp_input_image.channels] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels] / TILE_PIXELS);
        openmp_mosaic_value[t * openmp_input_image.channels + 1] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + 1] / TILE_PIXELS);
        openmp_mosaic_value[t * openmp_input_image.channels + 2] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + 2] / TILE_PIXELS);

        image_sum_r += openmp_mosaic_value[t * openmp_input_image.channels];
        image_sum_g += openmp_mosaic_value[t * openmp_input_image.channels + 1];
        image_sum_b += openmp_mosaic_value[t * openmp_input_image.channels + 2];
    }
    // Reduce the whole image sum to whole image average for the return value
    output_global_average[0] = (unsigned char)(image_sum_r / (openmp_TILES_X * openmp_TILES_Y));
    output_global_average[1] = (unsigned char)(image_sum_g / (openmp_TILES_X * openmp_TILES_Y));
    output_global_average[2] = (unsigned char)(image_sum_b / (openmp_TILES_X * openmp_TILES_Y));
    

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    validate_compact_mosaic(openmp_TILES_X, openmp_TILES_Y, openmp_mosaic_sum, openmp_mosaic_value, output_global_average);
#endif    
}


void openmp_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    // For each tile
    int t, p;
    unsigned int tile_index, tile_offset;
    max_threads = omp_get_num_procs();
    omp_set_num_threads(2* max_threads -1);

#pragma omp parallel for collapse(2) default(none) private(t, p,tile_index,tile_offset) shared(openmp_TILES_X,openmp_TILES_Y,openmp_mosaic_value,openmp_input_image) schedule(static)
    for (t = 0; t < openmp_TILES_X * openmp_TILES_Y; ++t) {
            tile_index = t * openmp_input_image.channels;
            tile_offset = ((int)(t / openmp_TILES_X)* openmp_TILES_X * TILE_PIXELS + (t % openmp_TILES_X) * TILE_SIZE) * openmp_input_image.channels;

            for (p = 0; p < TILE_PIXELS; ++p) {
                const unsigned int pixel_offset = ((int)(p/TILE_SIZE) * openmp_input_image.width + p % TILE_SIZE) * openmp_input_image.channels;
                // Copy whole pixel
                memcpy(openmp_output_image.data + tile_offset + pixel_offset, openmp_mosaic_value + tile_index, openmp_input_image.channels);
            }
            
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_broadcast(&openmp_input_image, openmp_mosaic_value, &openmp_output_image);
#endif    
}


void openmp_end(Image *output_image) {
    // Store return value
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    memcpy(output_image->data, openmp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(openmp_output_image.data);
    free(openmp_input_image.data);
    free(openmp_mosaic_value);
    free(openmp_mosaic_sum);
    
}