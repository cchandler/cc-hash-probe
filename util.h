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
int load_table(char *filename);
char* get_hash_file();
void print_config();
 
#endif