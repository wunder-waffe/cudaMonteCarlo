#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void random_number_generator(void)
{
    int tid
}

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    curandGenerator_t gen;
    // Define pointers to host
    float *h_xPos, *h_yPos;
    // Define pointers to device
    float *d_xPos, *d_yPos;

    float piApprox = 0;
    int numPoints = 1024;
    size_t size = numPoints * sizeof(float);
    
    printf("[Pi approximation with %d points]\n", numPoints);

    // Allocate host x position vector
    float * h_xPos = (float *)malloc(size);
    // Allocate device x position vector
    gpuErrchk(((void **) &d_xPos, size));

    // Allocate host y position vector
    float * h_yPos = (float *)malloc(size);
    // Allocate device y position vector    
    gpuErrchk(cudaMalloc((void **) &d_yPos, size));

    // Verify the allocations succeeded
    if (h_xPos == NULL || h_yPos == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Create a Mersenne Twister psuedorandom number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    // Set seed
    curandSetPseudoRandomGenerator(gen, 1234ULL);
    // Generate numPoints floats on device
    curandGenerateUniform(gen, d_xPos, numPoints);
    curandGenerateUniform(gen, d_yPos, numPoints);

    int blockDimX = 16;
    int blockDimY = 16;
    int threadsPerBlock = blockDimX * blockDimY; 
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

    dim3 grid(blocksPerGrid);
    dim3 block(blockDimX,blockDimY);
    
}