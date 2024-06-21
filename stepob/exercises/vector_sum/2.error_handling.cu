#include<stdio.h>
#include<cuda_runtime.h>

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

void vsum(int* a, int* b, int* c, int dim);
__global__ void kernel(int* a, int* b, int* c, int dim);


/*CPU function performing vector addition c = a + b*/
void vsum(int* a, int* b, int* c, int dim){
  int i;
  for(i=0; i<dim; i++)
    c[i] = a[i] + b[i];
}

/*GPU kernel performing vector addition c = a + b*/
__global__ void kernel(int* a, int* b, int* c, int dim){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<dim)
    c[i] = a[i] + b[i]; 
}

int main(){
  int host_va[N],host_vb[N],host_res_CPU[N],host_res_GPU[N];
  int *d_va, *d_vb, *d_res;
  int i, ok;


  /*initialize vectors*/  
  for(i=0; i<N; i++){
    host_va[i] = i;
    host_vb[i] = N-i;    
  }
  
  /*call CPU function*/
  vsum(host_va, host_vb, host_res_CPU, N);

  /*allocate memory on the GPU*/
  CHECK(cudaMalloc(&d_va, N*sizeof(int)));
  CHECK(cudaMalloc(&d_vb, N*sizeof(int)));
  CHECK(cudaMalloc(&d_res, N*sizeof(int)));

  /*transmit data to GPU*/
  CHECK(cudaMemcpy(d_va, host_va, N*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vb, host_vb, N*sizeof(int), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb, d_res, N);
  CHECK_KERNELCALL();

  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(host_res_GPU, d_res, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(host_res_CPU[i] != host_res_GPU[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");

  /*free memory*/
  CHECK(cudaFree(d_va));
  CHECK(cudaFree(d_vb));
  CHECK(cudaFree(d_res));
  
  return 0;
}




