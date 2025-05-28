#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <nvml.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {

    int myid, ntask;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    fprintf(stdout, "Process %d see %d GPUs.\n", myid, deviceCount);

    if (error_id != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error_id));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    // Asign GPU
    int my_device = myid % deviceCount;
    cudaSetDevice(my_device);

    // Check the asignement

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Process %d: Failed to initialize NVML: %s\n", myid, nvmlErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(my_device, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Process %d: Failed to get handle for device %d: %s\n", myid, my_device, nvmlErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
    result = nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Process %d: Failed to get UUID: %s\n", myid, nvmlErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("Process %d is assigned GPU %d with UUID: %s\n", myid, my_device, uuid);

    nvmlShutdown();

    MPI_Finalize();
    return(0);
}
