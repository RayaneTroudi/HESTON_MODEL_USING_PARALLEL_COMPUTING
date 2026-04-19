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
__global__ void monteCarloHestonModel(float kappa, float theta, float sigma, float r, float rho, // paramaters of the model
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
        S = S * ( 1.0f + r * dt + sqrtf(V) * sqrtf(dt) * ( rho * G.x + sqrtf(1.0f-rho*rho) * G.y ));
        V = fmaxf(V + kappa * (theta - V) * dt + sigma * sqrtf(V) * sqrtf(dt) * G.x , 0.0f);
        
    }

    payoffGPU[idx] = expf(-r * N * dt) * fmaxf(0.0f, S - K);

    state[idx] = localState;
    
}


// Gamma distribution function
__device__ float gamma_standard(float alpha, curandState* state)
{
    // Case alpha < 1:
    // Gamma(alpha) = Gamma(alpha + 1) * U^(1/alpha)
    if (alpha < 1.0f)
    {
        float u = curand_uniform(state);   // in (0,1]
        return gamma_standard(alpha + 1.0f, state) * powf(u, 1.0f / alpha);
    }

    // Marsaglia-Tsang for alpha >= 1
    float d = alpha - 1.0f / 3.0f;
    float c = rsqrtf(9.0f * d);  // 1 / sqrt(9d)

    while (true)
    {
        float x = curand_normal(state);   // N(0,1)
        float v = 1.0f + c * x;

        if (v <= 0.0f)
            continue;

        v = v * v * v;

        float u = curand_uniform(state);  // U(0,1)

        // Squeeze test
        if (u < 1.0f - 0.0331f * x * x * x * x)
            return d * v;

        // Exact acceptance test
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
            return d * v;
    }
}


__global__ void testGammaKernel(float alpha, curandState* states, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState localState = states[idx];

    out[idx] = gamma_standard(alpha, &localState);

    states[idx] = localState;
}


int main(void){


    /* ______________________________________ QUESTION 1 ______________________________________ */


    // ___ INIT BLOCK ___
    int NTPB = 512; // number of threads per block
    int NB = 512; // number of blocks
    int n = NB * NTPB;

    float T = 1.0f; // maturity
    float S0 = 1.0f; 
    float K = S0; // at the money

    int N = 1000;
    float dt = (float) T / (float) N; ;

    float sigma = 0.3f;
    float kappa = 0.5f;
    float theta = 0.1f;
    float r = 0.0f;
    float rho = -0.7f;

    float sum = 0.0f;
    float sum2 = 0.0f;

    float *payoffGPU, *payoffCPU;


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
    monteCarloHestonModel<<<NB,NTPB>>>(kappa,theta,sigma,r,rho,dt,N,K,S0,states,payoffGPU);
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
    printf("Execution time %f ms\n", Tim);
    // free the memory
    free(payoffCPU);
    testCUDA(cudaFree(payoffGPU));


    /* ______________________________________ QUESTION 2 ______________________________________ */


    float alpha = 2.0f;   // test alpha >= 1
    // float alpha = 0.5f; // test alpha < 1

    float *gammaGPU, *gammaCPU;
    gammaCPU = (float*)malloc(n * sizeof(float));
    testCUDA(cudaMalloc(&gammaGPU, n * sizeof(float)));

    // Re-init the RNG states if you want a clean independent test
    init_curand_state_k<<<NB, NTPB>>>(states);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // timer
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    testGammaKernel<<<NB, NTPB>>>(alpha, states, gammaGPU, n);
    testCUDA(cudaGetLastError());

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&Tim, start, stop));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));

    testCUDA(cudaMemcpy(gammaCPU, gammaGPU, n * sizeof(float), cudaMemcpyDeviceToHost));

    float meanGamma = 0.0f;
    float varGamma = 0.0f;

    for (int i = 0; i < n; i++) {
        meanGamma += gammaCPU[i];
    }
    meanGamma /= n;

    for (int i = 0; i < n; i++) {
        float diff = gammaCPU[i] - meanGamma;
        varGamma += diff * diff;
    }
    varGamma /= n;

    printf("\n=== Gamma test ===\n");
    printf("alpha = %f\n", alpha);
    printf("Theoretical mean     = %f\n", alpha);
    printf("Monte Carlo mean     = %f\n", meanGamma);
    printf("Theoretical variance = %f\n", alpha);
    printf("Monte Carlo variance = %f\n", varGamma);
    printf("Execution time %f ms\n", Tim);

    free(gammaCPU);
    testCUDA(cudaFree(gammaGPU));
    testCUDA(cudaFree(states));

    return 0;

    
}