#include<stdio.h>
#include<cuda_runtime.h>

#define N (256)
#define BLOCKDIM 32
#define MAXVAL 1

void matrix_cpu(int* a, int* b, int* c, int dim);
__global__ void kernel(int* a, int* b, int* c, int dim);


void print_matrix(int* m, int dim){
    for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            printf("%d ",m[i*dim+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_cpu(int* a, int* b, int* c, int dim){
    int i,j,k,acc;
    acc = 0;
    for(i=0; i<dim; i++){
        for(j=0;j<dim;j++){
            for(k=0;k<dim;k++){
                acc += (a[i*dim+k]*b[k*dim+j]);
            }
        c[i*dim+j] = acc;
        acc=0;
        }
    }   
}

__global__ void kernel(int* a, int* b, int* res, int dim){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int a_tile[BLOCKDIM*BLOCKDIM];
    __shared__ int b_tile[BLOCKDIM*BLOCKDIM];
    __shared__ int res_tile[BLOCKDIM*BLOCKDIM];
    //int acc = 0;

    if(x<dim && y<dim){

        for(int i=0;i<gridDim.x;i++){
            a_tile[threadIdx.x*BLOCKDIM + threadIdx.y] = a[x*dim +((i*BLOCKDIM)+threadIdx.y)];
            b_tile[threadIdx.x*BLOCKDIM + threadIdx.y] = b[((i*BLOCKDIM)+threadIdx.x)*dim + y];
            __syncthreads();

            for(int j=0;j<BLOCKDIM;j++){
                res_tile[threadIdx.x*BLOCKDIM+threadIdx.y] += a_tile[threadIdx.x*BLOCKDIM+j]*b_tile[j*BLOCKDIM + threadIdx.y];
            }
            __syncthreads();

            // res[x*dim + y] = b_tile[threadIdx.x*BLOCKDIM+threadIdx.y];
            // break;
            
        }
        res[x*dim + y] = res_tile[threadIdx.x*BLOCKDIM+threadIdx.y];
    }
}

int main(){
    int *h_va, *h_vb, *h_vc_cpu, *h_vc_gpu;
    int *d_va, *d_vb, *d_vc;
    int i,j, ok;

    h_va = (int*)malloc(N*N*sizeof(int));
    h_vb = (int*)malloc(N*N*sizeof(int));
    h_vc_cpu = (int*)malloc(N*N*sizeof(int));
    h_vc_gpu = (int*)malloc(N*N*sizeof(int));

    /*initialize vectors*/ 
    srand(59);
    for(j=0;j<N*N;j++){ 
        h_va[j] = rand()%MAXVAL +1;
        h_vb[j] = rand()%MAXVAL +1;
    }

    /*call CPU function*/
    matrix_cpu(h_va, h_vb, h_vc_cpu, N);
    printf("Cpu res\n");
    // print_matrix(h_va,N);
    // print_matrix(h_vb,N);
    print_matrix(h_vc_cpu,N);

    /*allocate memory on the GPU*/
    cudaMalloc(&d_va, N*N*sizeof(int));
    cudaMalloc(&d_vb, N*N*sizeof(int));
    cudaMalloc(&d_vc, N*N*sizeof(int));

    /*transmit data to GPU*/
    cudaMemcpy(d_va, h_va, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, h_vb, N*N*sizeof(int), cudaMemcpyHostToDevice);

    /*invoke the kernel on the GPU*/
    dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, (N+BLOCKDIM-1)/BLOCKDIM, 1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM, 1);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb, d_vc, N);

    /*transmit data from the GPU*/
    cudaMemcpy(h_vc_gpu, d_vc, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Gpu res\n");
    //print_matrix(h_vc_gpu,N);

    /*compare results*/
    for(i=0, ok=1; i<N*N && ok; i++)
        if(h_vc_cpu[i] != h_vc_gpu[i]){
        printf("Error!\n");
        ok = 0;
        }
    if(ok)
        printf("Success!\n");

    /*free memory*/
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_vc);

    return 0;
}

