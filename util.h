#ifndef UTILITY_MODULE
#define UTILITY_MODULE

#include <glib.h>
#include "gpu.h"

typedef struct
{
	gchar *hash_file;
	unsigned long cc_start_point;
	unsigned long cc_end_point;
	unsigned int blocksize;
	unsigned int threadsize;
	int hashthreads;
	int scanthreads;
	int cpumode;
	int gpumode;
} Config;

int divide_hash_space_for_gpu(unsigned long start, unsigned long end, unsigned long offset, unsigned long *intervals);
void divide_hash_space_for_threads(unsigned long start, unsigned long end, cc_interval_t *data, int thread_count);
long get_work_size();
int get_block_size();
int get_thread_size();
int is_gpu_mode();
int get_scan_thread_count();
int get_hash_thread_count();
unsigned long get_cc_start_point();
unsigned long get_cc_end_point();

int load_config();
void print_config();
 
#endif