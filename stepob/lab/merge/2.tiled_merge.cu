#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define SIZE_A 20123
#define SIZE_B 12768
#define SIZE_C (SIZE_A + SIZE_B)

#define BLOCK_DIM 128
#define GRID_DIM  32
#define TILE_SIZE 1024

__device__ __forceinline__ int gpu_ceil(const int a, const int b) { return ceil((double) a / b); }
__device__ __forceinline__ int gpu_min(const int a, const int b) { return min(a, b); }
__device__ __forceinline__ int gpu_max(const int a, const int b) { return max(a, b); }

// Function for CPU and GPU merge sequentia computation
__device__ __host__ void
    merge_sequential(const int *A, const int dim_A, const int *B, const int dim_B, int *C) {
  int i = 0;
  int j = 0;
  int k = 0;

  while ((i < dim_A) && (j < dim_B)) {
    if (A[i] <= B[j])
      C[k++] = A[i++];
    else
      C[k++] = B[j++];
  }
  if (i == dim_A) {
    while (j < dim_B) { C[k++] = B[j++]; }
  } else {
    while (i < dim_A) { C[k++] = A[i++]; }
  }
}

// Function to compute the co-rank
__device__ int
    co_rank(const int k, const int *__restrict__ A, const int m, const int *__restrict__ B, const int n) {
  int i     = gpu_min(k, m); // min
  int j     = k - i;
  int i_low = gpu_max(0, (k - n)); // max
  int j_low = gpu_max(0, (k - m));
  int delta;
  bool active = true;
  while (active) {
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      delta = ((i - i_low + 1) >> 1); // ceil (smallest integer greater than or equal)
      j_low = j;
      j     = j + delta;
      i     = i - delta;
    } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
      delta = ((j - j_low + 1) >> 1);
      i_low = i;
      i     = i + delta;
      j     = j - delta;
    } else {
      active = false;
    }
  }
  return i;
}

// Function for GPU merge computation
__global__ void merge_tiled_gpu(const int *__restrict__ A,
                                const int dim_A,
                                const int *__restrict__ B,
                                const int dim_B,
                                int *__restrict__ C) {
  __shared__ int A_s[TILE_SIZE];
  __shared__ int B_s[TILE_SIZE];

  const int dim_C = dim_A + dim_B;

  const int C_curr = blockIdx.x * gpu_ceil(dim_C, gridDim.x);
  const int C_next = gpu_min((blockIdx.x + 1) * gpu_ceil(dim_C, gridDim.x), dim_C);

  if (threadIdx.x == 0) {
    A_s[0] = co_rank(C_curr, A, dim_A, B, dim_B);
    A_s[1] = co_rank(C_next, A, dim_A, B, dim_B);
  }
  __syncthreads();

  const int A_curr = A_s[0];
  const int A_next = A_s[1];
  const int B_curr = C_curr - A_curr;
  const int B_next = C_next - A_next;

  int counter               = 0;
  const int C_length        = C_next - C_curr;
  const int A_length        = A_next - A_curr;
  const int B_length        = B_next - B_curr;
  const int total_iteration = gpu_ceil(C_length, TILE_SIZE);
  int C_completed           = 0;
  int A_consumed            = 0;
  int B_consumed            = 0;
  while (counter < total_iteration) {
    const int A_length_t = gpu_min(TILE_SIZE, A_length - A_consumed);
    const int B_length_t = gpu_min(TILE_SIZE, B_length - B_consumed);
    // Loading tile
    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
      if (i < A_length - A_consumed) {
        A_s[i] = A[A_curr + A_consumed + i];
      }
    }
    for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
      if (i < B_length - B_consumed) {
        B_s[i] = B[B_curr + B_consumed + i];
      }
    }
    __syncthreads();

    int c_curr = threadIdx.x * (TILE_SIZE / blockDim.x);
    int c_next = (threadIdx.x + 1) * (TILE_SIZE / blockDim.x);
    c_curr     = gpu_min(c_curr, C_length - C_completed);
    c_next     = gpu_min(c_next, C_length - C_completed);
    // Find co_rank
    int a_curr = co_rank(c_curr, A_s, A_length_t, B_s, B_length_t);
    int b_curr = c_curr - a_curr;
    int a_next = co_rank(c_next, A_s, A_length_t, B_s, B_length_t);
    int b_next = c_next - a_next;
    // Merge and update counter and completed elements
    merge_sequential(A_s + a_curr,
                     a_next - a_curr,
                     B_s + b_curr,
                     b_next - b_curr,
                     C + C_curr + C_completed + c_curr);
    counter++;
    C_completed += TILE_SIZE;
    A_consumed += co_rank(TILE_SIZE, A_s, A_length_t, B_s, B_length_t);
    B_consumed = C_completed - A_consumed;
    __syncthreads();
  }
}

// simple sort, just to sort the randomly generated arrays
void sort_array(int *arr, const int dim) {
  int i, j;

  for (i = 0; i < dim; i++) {
    for (j = i + 1; j < dim; j++) {
      if (arr[j] < arr[i]) {
        int tmp = arr[i];
        arr[i]  = arr[j];
        arr[j]  = tmp;
      }
    }
  }
}

int main() {
  // Generate random input data
  std::vector<int> A_host(SIZE_A);
  std::vector<int> B_host(SIZE_B);
  for (int i = 0; i < SIZE_A; ++i) { A_host[i] = rand() % 1000; }
  for (int i = 0; i < SIZE_B; ++i) { B_host[i] = rand() % 1000; }
  sort_array(A_host.data(), SIZE_A);
  sort_array(B_host.data(), SIZE_B);

  // Compute merge on CPU
  std::vector<int> cpu_result(SIZE_C);
  merge_sequential(A_host.data(), SIZE_A, B_host.data(), SIZE_B, cpu_result.data());

  // Allocate memory on GPU
  int *A_device, *B_device, *C_device;
  CHECK(cudaMalloc(&A_device, SIZE_A * sizeof(int)));
  CHECK(cudaMalloc(&B_device, SIZE_B * sizeof(int)));
  CHECK(cudaMalloc(&C_device, SIZE_C * sizeof(int)));
  CHECK(cudaMemcpy(A_device, A_host.data(), SIZE_A * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_device, B_host.data(), SIZE_B * sizeof(int), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(BLOCK_DIM);
  dim3 blocks_per_grid(GRID_DIM);
  // Launch GPU kernel
  merge_tiled_gpu<<<blocks_per_grid, threads_per_block>>>(A_device, SIZE_A, B_device, SIZE_B, C_device);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<int> gpu_result(SIZE_C);
  CHECK(cudaMemcpy(gpu_result.data(), C_device, SIZE_C * sizeof(int), cudaMemcpyDeviceToHost));

  // Compare CPU and GPU results
  for (int i = 0; i < SIZE_C; ++i)
    if (cpu_result[i] != gpu_result[i]) {
      std::cout << "Merge CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Merge CPU and GPU are equivalent!" << std::endl;

  // Free memory
  CHECK(cudaFree(A_device));
  CHECK(cudaFree(B_device));
  CHECK(cudaFree(C_device));

  return EXIT_SUCCESS;
}
