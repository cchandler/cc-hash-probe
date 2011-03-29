#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>

#include <glib.h>

#include "util.h"

extern Config conf_data;

int divide_hash_space_for_gpu(unsigned long start, unsigned long end, unsigned long offset, unsigned long *intervals) {
	unsigned long total = end - start;
	unsigned long chunk = floor(total / get_block_size());
	
	if(chunk < offset){ // If this is the case we've scanned the entire range
		return 0;
	}
	
	int i = 0;
	for(; i < get_block_size(); ++i){
		intervals[i] = start + (chunk * i + offset);
	}
	return 1;
}

void divide_hash_space_for_threads(unsigned long start, unsigned long end, cc_interval_t *data, int hash_thread_count) {
	unsigned long total = end - start;
	unsigned long chunk = ceil(total / hash_thread_count);
	
	int i = 0;
	for(; i < hash_thread_count; ++i){
		data[i].start_point = start + (chunk * i);
		data[i].end_point = start + ((chunk * (i + 1)) + 1);
	}
}

long get_work_size(){
	return conf_data.blocksize * conf_data.threadsize;
}

int get_block_size(){
	return conf_data.blocksize;
}

int get_thread_size(){
	return conf_data.threadsize;
}

int is_gpu_mode(){
	return conf_data.gpumode;
}

int get_scan_thread_count(){
	return conf_data.scanthreads;
}

int get_hash_thread_count(){
	return conf_data.hashthreads;
}

unsigned long get_cc_start_point(){
	return conf_data.cc_start_point;
}

unsigned long get_cc_end_point(){
	return conf_data.cc_end_point;
}

int load_config(){
	GKeyFile *conf_file;
	conf_file = g_key_file_new();
	GKeyFileFlags flags;
	GError *error = NULL;
	
	flags = G_KEY_FILE_KEEP_TRANSLATIONS;
	
	if (!g_key_file_load_from_file (conf_file, "cchash.conf", flags, &error))
	  {
		printf("Error reading config\n");
	    // g_error(error->message);
	    return -1;
	  }
	
	conf_data.hash_file = g_key_file_get_string(conf_file, "cc-hash-probe", "hashfile", NULL);
	conf_data.cc_start_point = g_key_file_get_uint64(conf_file, "cc-hash-probe", "cc_start_point", NULL);
	conf_data.cc_end_point = g_key_file_get_uint64(conf_file, "cc-hash-probe", "cc_end_point", NULL);
	conf_data.blocksize = g_key_file_get_integer(conf_file, "cc-hash-probe", "blocksize", NULL);
	conf_data.threadsize = g_key_file_get_integer(conf_file, "cc-hash-probe", "threadsize", NULL);
	conf_data.hashthreads = g_key_file_get_integer(conf_file, "cc-hash-probe", "hashthreads", NULL);
	conf_data.scanthreads = g_key_file_get_integer(conf_file, "cc-hash-probe", "scanthreads", NULL);
	conf_data.cpumode = g_key_file_get_integer(conf_file, "cc-hash-probe", "cpumode", NULL);
	conf_data.gpumode = g_key_file_get_integer(conf_file, "cc-hash-probe", "gpumode", NULL);
	return 1;
}

void print_config(){
	printf("-----Config\n");
	printf("Hash file: %s\n", conf_data.hash_file);
	printf("CC start point: %lu\n", conf_data.cc_start_point);
	printf("CC end point: %lu\n", conf_data.cc_end_point);
	printf("Block size: %d\n", conf_data.blocksize);
	printf("Thread size: %d\n", conf_data.threadsize);
	printf("Hash threads: %d\n",conf_data.hashthreads);
	printf("Scan threads: %d\n", conf_data.scanthreads);
	if(is_gpu_mode()){
		printf("GPU mode\n");
	}else{
		printf("CPU mode\n");
	}
	printf("-----End Config\n");
}