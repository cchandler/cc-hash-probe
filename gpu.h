/*
Copyright (C) 2011 by Chris Chandler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN

*/

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