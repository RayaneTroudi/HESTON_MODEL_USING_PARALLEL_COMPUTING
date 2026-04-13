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
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))


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

    int N = 100;
    float dt = 1.0f / 1000.0f ;

    float sigma = 0.3f;
    float kappa = 0.5f;
    float theta = 0.1f;
    float r = 0.024f;
    float rho = 1.0;

    float sum = 0.0f;
    float sum2 = 0.0f;

    float *payoffGPU, *payoffCPU;
    printf("%s \n","ntm");

    // init of the array that contains all the payoff from each path generated
    payoffCPU = (float*)malloc(n * sizeof(float));
    testCUDA(cudaMalloc(&payoffGPU, n * sizeof(float)));

    // init of the array that will contain the state of each thread
    curandState *states;
    testCUDA(cudaMalloc(&states, n * sizeof(curandState)));

    //init of the state of each thread
    init_curand_state_k<<<NB, NTPB>>>(states);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // init of the timer
    float Tim;
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    // launch the kernel that compute the payoff of each path at maturity
    MC_heston_model<<<NB,NTPB>>>(kappa,theta,sigma,r,rho,dt,N,K,S0,states,payoffGPU);
    testCUDA(cudaGetLastError());
  
    // stop the timer
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&Tim, start, stop));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));

    // copy the payoff of each path from the device to the host
    testCUDA(cudaMemcpy(payoffCPU, payoffGPU, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Reduction performed on the host
    for (int i = 0; i < n; i++) {
        sum += payoffCPU[i] / n;
        sum2 += payoffCPU[i] * payoffCPU[i] / n;
    }

    printf("The estimated price is equal to %f\n", sum);

    // free the memory
    free(payoffCPU);
    testCUDA(cudaFree(payoffGPU));
    testCUDA(cudaFree(states));

    return 0;
}