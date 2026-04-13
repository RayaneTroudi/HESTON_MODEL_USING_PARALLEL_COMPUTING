#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

// Function that catches the error
void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}

// Init the seed for each thread that compute a path price
__global__ void init_curand_state_k(curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

// function to return an array that contains all the payoff at the maturity (N)
__global__ void MC_heston_model(float k, float theta, float sigma, // paramaters of the model
                                float dt, int N, // parameters of the discritisation (time step and maturity)
                                float K, float S0, // paramater of the option (Strike)
                                curandState* state, // array of seed of each thread
                                float* payOffGPU){ // output

    // ___ INIT BLOCK ___
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    curandState localState = state[idx]; 
    float2 G; // Contains G1 and G2 (that we need for Heston Model)    
    float S = S0; // init of the first value of the path
    float v0 = 0.1f; // init of the first value of the vol



    // ___ SIMULATION BLOCK ___
    for (int i = 0; i < N; i++)
    {
        G = curand_normal2(&localState);
    }
    

    
}


int main(void){


    return 0;
}