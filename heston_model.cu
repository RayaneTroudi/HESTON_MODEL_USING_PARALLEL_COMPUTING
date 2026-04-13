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
__global__ void MC_heston_model(float kappa, float theta, float sigma, float r, float rho, // paramaters of the model
                                float dt, int N, // parameters of the discritisation (time step and number of time steps)
                                float K, float S0, // paramater of the option (Strike)
                                curandState* state, // array of seed of each thread
                                float* payoffGPU){ // output

    // ___ INIT BLOCK ___
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    curandState localState = state[idx]; 
    float2 G; // Contains G1 and G2 (that we need for Heston Model)    
    float S = S0; // init of the first value of the path
    float V = 0.1f; // init of the first value of the vol



    // ___ SIMULATION BLOCK ___
    for (int i = 0; i < N; i++)
    {
        G = curand_normal2(&localState);
        V = fmaxf(V + kappa * (theta - V) * dt + sigma * sqrt(V) * sqrt(dt) * G.x , 0.0f);
        S = S * ( 1.0f + r * dt + sqrt(V) * sqrt(dt) * ( rho * G.x + sqrt(1-pow(rho,2)) * G.y ));
    }

    payoffGPU[idx] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K);

    state[idx] = localState;
    
}


int main(void){

    // ___ INIT BLOCK ___
    int NTPB = 512; // number of threads per block
    int NB = 512; // number of blocks
    int n = NB * NTPB;

    float T = 1.0f; // maturity
    float S0 = 50.0f; 
    float K = S0; // at the money

    int N = 1000;
    float dt = 1.0f / (float) N;

    float sigma = 0.3f;
    float kappa = 0.5f;
    float theta = 0.1f;

    float *payoffGPU;
    

    return 0;
}