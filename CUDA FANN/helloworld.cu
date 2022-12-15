#include "stdio.h"

__global__ void cuda_print(){
	printf("Hello from GPU from thread [%d,%d]\n",blockIdx.x, threadIdx.x);
}

int main (){
	int driverVersion;
	int runtimeVersion;

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("%d, %d\n",driverVersion, runtimeVersion);

	cuda_print<<<10,1>>>();
	//cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("Error %s\n",cudaGetErrorString(err));

	return 0;
}
