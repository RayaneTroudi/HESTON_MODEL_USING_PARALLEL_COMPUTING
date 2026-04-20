#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>


/* __________________________________________________________________________  */
// Function that catches the error
void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))








/* __________________________________________________________________________  */
// Init the seed for each thread that compute a path price
__global__ void init_curand_state_k(curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}






/* __________________________________________________________________________  */
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





/* __________________________________________________________________________  */
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







/* __________________________________________________________________________  */
// Kernel to test the gamma distribution function
__global__ void testGammaKernel(float alpha, curandState* states, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState localState = states[idx];

    out[idx] = gamma_standard(alpha, &localState);

    states[idx] = localState;
}





/* __________________________________________________________________________  */
__global__ void monteCarloHestonModelWithGamma(float kappa, float theta, float sigma, float r, float rho, // paramaters of the model
                                float dt, int N, // parameters of the discritisation (time step and number of time steps)
                                float K, // paramater of the option (Strike)
                                curandState* state, // array of seed of each thread
                                float* payoffGPU){ // output

    // ___ INIT BLOCK ___
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    curandState localState = state[idx]; // state seed of each thread
    float G; // G following a standard gamma distribution
    unsigned int P; // P following a standard poisson distribution
    float V0 = 0.1f;
    float lambda;
    float V_pred = V0; // init of the first value of the vol (vt)
    float V_current = V0; // vt_dt (next step of v)
    float d = (2.0f * kappa * theta) / (sigma * sigma); // constant
   
    float alpha;
    float Vi = 0.0f;
    float m;
    float S;
    float Z; 

    for (int i = 0; i < N; i++)
    {
        lambda = (2.0f * kappa * expf(-1.0f * kappa * dt) * V_pred) / ((sigma * sigma) * (1.0f - expf(-1.0f * kappa * dt))); // parameter of poisson law
        P =  curand_poisson(&localState, lambda);
        alpha = d + (float) P;
        G = gamma_standard(alpha,&localState);
        V_current = ((sigma * sigma) * (1.0f - expf(-1.0f * kappa * dt)) * G) / (2.0f * kappa);
        Vi += 0.5f * (V_pred + V_current)*dt;
        V_pred = V_current;

    }

    m = -0.5f * Vi + rho * (1.0/sigma) * (V_current - V0 - kappa * theta + kappa * Vi);
    Z = curand_normal(&localState);
    S = expf(m + sqrtf((1.0f - rho*rho) * Vi)*Z);

    payoffGPU[idx] = expf(-r * N * dt) * fmaxf(0.0f, S - K);
    state[idx] = localState;
    
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


    float *payoffGPU2, *payoffCPU2;
    payoffCPU2 = (float*)malloc(n * sizeof(float));
    testCUDA(cudaMalloc(&payoffGPU2, n * sizeof(float)));

    // Re-init the RNG states for an independent run
    init_curand_state_k<<<NB, NTPB>>>(states);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // timer
    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    testCUDA(cudaEventRecord(start, 0));

    monteCarloHestonModelWithGamma<<<NB, NTPB>>>(kappa, theta, sigma, r, rho, dt, N, K, states, payoffGPU2);
    testCUDA(cudaGetLastError());

    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&Tim, start, stop));
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));

    testCUDA(cudaMemcpy(payoffCPU2, payoffGPU2, n * sizeof(float), cudaMemcpyDeviceToHost));

    float sum2_q2 = 0.0f;
    float sumSq_q2 = 0.0f;
    for (int i = 0; i < n; i++) {
        sum2_q2  += payoffCPU2[i] / n;
        sumSq_q2 += payoffCPU2[i] * payoffCPU2[i] / n;
    }
    float stddev_q2 = sqrtf((sumSq_q2 - sum2_q2 * sum2_q2) / n);

    printf("\n=== Monte Carlo Heston Model with Gamma (Q2) ===\n");
    printf("Estimated price      = %f\n", sum2_q2);
    printf("95%% confidence interval: [%f, %f]\n", sum2_q2 - 1.96f * stddev_q2, sum2_q2 + 1.96f * stddev_q2);
    printf("Execution time       = %f ms\n", Tim);

    free(payoffCPU2);
    testCUDA(cudaFree(payoffGPU2));
    testCUDA(cudaFree(states));

    return 0;

    
}