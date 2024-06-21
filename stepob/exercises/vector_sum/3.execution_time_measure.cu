/*
Vector addition.
* Version 0: the sum is performed by a function on the CPU
*
* Version 1: the vector sum function is accelerated on GPU and results are
* compared against the CPU counterpart. Block size is set to 64 threads
*
* Version 2: error handling has been added
*
* Version 3: execution time measure has been added 
*/

#include<stdio.h>
#include<cuda_runtime.h>
#include <sys/time.h>

#define N (1<<16)
#define BLOCKDIM 64

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} \


#define CHECK_KERNELCALL() \
{ \
  const cudaError_t err = cudaGetLastError();\
  if (err != cudaSuccess) {\
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}\

inline double milliseconds();
void vsum(int* a, int* b, int* c, int dim);
__global__ void vsumKernel(int* a, int* b, int* c, int dim);


inline double milliseconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}

/*CPU function performing vector addition c = a + b*/
void vsum(int* a, int* b, int* c, int dim){
  int i;
  for(i=0; i<dim; i++)
    c[i] = a[i] + b[i];
}

/*GPU kernel performing vector addition c = a + b*/
__global__ void vsumKernel(int* a, int* b, int* c, int dim){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<dim)
    c[i] = a[i] + b[i]; 
}

int main(){
  int h_va[N], h_vb[N], h_vc_cpu[N], h_vc_gpu[N];
  int *d_va, *d_vb, *d_vc;
  int i, ok;
  double cpu_start, cpu_end, cpu_exectime;

  cudaEvent_t gpu_start;
  cudaEvent_t gpu_end;
  float gpu_exectime;
  
  /*initialize CUDA events*/
  CHECK(cudaEventCreate(&gpu_start));
  CHECK(cudaEventCreate(&gpu_end));

  /*initialize vectors*/  
  for(i=0; i<N; i++){
    h_va[i] = i;
    h_vb[i] = N-i;    
  }
  
  /*call CPU function*/
  cpu_start = milliseconds();
  vsum(h_va, h_vb, h_vc_cpu, N);
  cpu_end = milliseconds();
  cpu_exectime = cpu_end - cpu_start;

  /*allocate memory on the GPU*/
  CHECK(cudaMalloc(&d_va, N*sizeof(int)));
  CHECK(cudaMalloc(&d_vb, N*sizeof(int)));
  CHECK(cudaMalloc(&d_vc, N*sizeof(int)));

  /*transmit data to GPU*/
  CHECK(cudaMemcpy(d_va, h_va, N*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vb, h_vb, N*sizeof(int), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  CHECK(cudaEventRecord(gpu_start));
  vsumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb, d_vc, N);
  //CHECK_KERNELCALL();
  CHECK(cudaEventRecord(gpu_end));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventElapsedTime(&gpu_exectime, gpu_start, gpu_end));
        
  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(h_vc_gpu, d_vc, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(h_vc_cpu[i] != h_vc_gpu[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");
  
  /*report execution time*/  
  printf("Execution times\n- CPU: %f\n- GPU: %f\n", cpu_exectime, gpu_exectime);

  /*free memory*/
  CHECK(cudaEventDestroy(gpu_start));
  CHECK(cudaEventDestroy(gpu_end));
  CHECK(cudaFree(d_va));
  CHECK(cudaFree(d_vb));
  CHECK(cudaFree(d_vc));
  
  return 0;
}



