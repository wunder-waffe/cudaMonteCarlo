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

__global__ void monte_carlo_pi(float *xPos, float *yPos, float *distance)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int numThreadsPerBlock = blockDim.x * blockDim.y;
    int gid = tid + numThreadsPerBlock * blockIdx.x;
    distance[gid] = hypotf(xPos[gid], yPos[gid]);
    return;
}

__global__ void compares(float *distance, int *INTresult)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int numThreadsPerBlock = blockDim.x * blockDim.y;
    int gid = tid + numThreadsPerBlock * blockIdx.x;
    INTresult[gid] = (distance[gid] < 1);
    return;
}

int main(int argc, char *argv[])
{
    curandGenerator_t gen;
    // Define pointers to host
    float *h_xPos, *h_yPos;
    int *h_result;
    // Define pointers to device
    float *d_xPos, *d_yPos, *d_dist;
    int *d_result;

    float piApprox = 0;
    int numPoints = 1e8;
    size_t size = numPoints * sizeof(float);
    
    printf("[Pi approximation with %d points]\n", numPoints);

    // Allocate host x position vector
    h_xPos = (float *)malloc(size);
    // Allocate device x position vector
    gpuErrchk(cudaMalloc((void **) &d_xPos, size));

    // Allocate host y position vector
    h_yPos = (float *)malloc(size);
    // Allocate device y position vector    
    gpuErrchk(cudaMalloc((void **) &d_yPos, size));

    gpuErrchk(cudaMalloc((void **) &d_dist, size));

    // Allocate condition matrix
    h_result = (int *)malloc(numPoints*sizeof(int));
    gpuErrchk(cudaMalloc((void **) &d_result, numPoints*sizeof(int)));

    // Verify the allocations succeeded
    if (h_xPos == NULL || h_yPos == NULL || h_result == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Create a Mersenne Twister psuedorandom number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    // Set seed
    curandSetPseudoRandomGeneratorSeed(gen, 3234ULL);

    // Generate numPoints floats on device
    curandGenerateUniform(gen, d_xPos, numPoints);
    curandGenerateUniform(gen, d_yPos, numPoints);

    int blockDimX = 32;
    int blockDimY = 16;
    int threadsPerBlock = blockDimX * blockDimY; 
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

    dim3 grid(blocksPerGrid);
    dim3 block(blockDimX,blockDimY);

    
    monte_carlo_pi<<<grid, block>>>(d_xPos, d_yPos, d_dist);
    compares<<<grid, block>>>(d_dist, d_result);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(h_result, d_result, numPoints*sizeof(int), cudaMemcpyDeviceToHost));

    int sum = 0;
    for (int i = 0; i < numPoints; i++)
    {
        sum += h_result[i];
    }
    piApprox = (4.0 * sum) / numPoints;
    printf("%d\n",sum);
    printf("Approximate Pi calculated with %d points:\n %.10f", numPoints, piApprox);

    cudaFree(d_dist);
    cudaFree(d_result);
    cudaFree(d_xPos);
    cudaFree(d_yPos);
    free(h_result);
    free(h_xPos);
    free(h_yPos);

    return EXIT_SUCCESS;
}