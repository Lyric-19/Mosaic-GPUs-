# COM4521/COM6521 Assignment Starting Code 2022

This is the code for COM4521/COM6521 assignment. 

Project objective: Use CUDA parallel computing to process the input RGB image with a size of 32x32 pixels for Mosaic processing, and output the processed Mosaic image.

Project content: 1. Allocate 32x3 thread blocks equal to the number of Mosaic squares required for the image, and calculate the pixel sum of each Mosaic square using Suffle and atomicAdd functions. 2. Increase thread utilization, allocate thread blocks with the same size of 32 as the number of Mosaic blocks, and calculate the pixel mean of each Mosaic block. 3. Allocate 32x32 thread blocks with the same number of Mosaic squares and apply shared memory between Warp to propagate the pixel mean to each pixel covered by the Mosaic squares.

Obtain from project: Proficient in the application of GRID, thread blocks, warp and threads in CUDA, and a deep understanding of cudA-related functions and the impact of thread allocation on computing speed.
