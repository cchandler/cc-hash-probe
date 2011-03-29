#ifndef GPU_MODULE
#define GPU_MODULE


#define HASH_CHUNKS  5
#define MODE "CPU"

typedef struct {
	unsigned int gpuId;
	
	size_t d_intervals_pitch;
	unsigned long long int *d_intervals;
	
	size_t d_valid_pitch;
	unsigned int* d_valid;
	
	size_t d_hash_pitch;
	unsigned int* d_hash;
} cc_gpu_state_t;

typedef struct {
	unsigned int threadId;
	unsigned long start_point;
	unsigned long end_point;
	
	// unsigned int gpuId;
	cc_gpu_state_t gpu_state;
} cc_interval_t;

int setupCUDA(cc_gpu_state_t *state);
int teardownCUDA(cc_gpu_state_t *state);
int cuda_scan(cc_gpu_state_t *state, unsigned long *intervals, unsigned int *valid, int *hashes);
int getCudaDeviceCount();

#endif