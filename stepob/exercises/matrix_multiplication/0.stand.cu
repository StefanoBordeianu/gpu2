#include<stdio.h>
#include<cuda_runtime.h>

#define N (1024)
#define BLOCKDIM 64
#define MAXVAL 3

void matrix_cpu(int* a, int* b, int* c, int dim);
__global__ void kernel(int* a, int* b, int* c, int dim);


void print_matrix(int* m, int dim){
    // for(int i=0;i<dim;i++){
    //     for(int j=0;j<dim;j++){
    //         printf("%d ",m[i*dim+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
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
    int acc = 0;

    if(x<dim && y<dim){
        for(int i=0;i<dim;i++){
            acc += (a[x*dim + i]*b[i*dim + y]);
        }
        res[x*dim + y] = acc;
    }
    res[dim*x + y] = acc;
}

int main(){
    int h_va[N*N], h_vb[N*N], h_vc_cpu[N*N], h_vc_gpu[N*N];
    int *d_va, *d_vb, *d_vc;
    int i,j, ok;


    /*initialize vectors*/ 
    srand(0);
    for(j=0;j<N*N;j++){ 
        h_va[j] = rand()%MAXVAL +1;
        h_vb[j] = rand()%MAXVAL +1;
    }

    /*call CPU function*/
    matrix_cpu(h_va, h_vb, h_vc_cpu, N);
    printf("Cpu res\n");
    print_matrix(h_va,N);
    print_matrix(h_vb,N);
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
    print_matrix(h_vc_gpu,N);

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

