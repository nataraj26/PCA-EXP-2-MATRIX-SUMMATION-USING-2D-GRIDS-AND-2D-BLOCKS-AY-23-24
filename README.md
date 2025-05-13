# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM: To perform matrix summation using 2D grids and 2D blocks </h3>
<h3>ENTER YOUR NAME: NATARAJ KUMARAN S</h3>
<h3>ENTER YOUR REGISTER NO: 212223230137</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```c

#include<iostream>
#include<cuda_runtime.h>
#include"cuda_utils.cuh"

//#define N 10000

//#define N 1000 // for threads since their limit is 1024

#define N (33*1024)

// Using Blocks 

//__global__ void add(int* a, int* b, int* c) {
//	int tid = blockIdx.x;
//	if(tid<N){
//		c[tid] = a[tid] + b[tid];
//	}
//}

// Using Threads

//__global__ void add(int* a, int* b, int* c) {
//	int tid = threadIdx.x;
//	if (tid < N) {
//		c[tid] = a[tid] + b[tid];
//	}
//}

// Using Threads and Blocks

__global__ void add(int* a, int* b, int* c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


void performVecAdd() {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0;i < N;i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, &a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, &b, N * sizeof(int), cudaMemcpyHostToDevice));

	//add << <N, 1 >> > (dev_a, dev_b, dev_c); // For blocks
	//add << <1, N >> > (dev_a, dev_b, dev_c); // For threads
	add << <128, 128 >> > (dev_a, dev_b, dev_c); // For blocks and threads
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0;i < N;i++) {
		std::cout << a[i] << "+" << b[i] << " = " << c[i] << "\n";
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/30d30275-2aeb-473f-915a-e845ec58b170)

## RESULT:
The host took _________ seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in ________ seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
