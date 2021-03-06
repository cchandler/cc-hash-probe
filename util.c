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

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <glib.h>

#include "util.h"

extern Config conf_data;
extern GHashTable *htable;
int found = 1;

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
	unsigned long chunk = floor(total / hash_thread_count);
	
	int i = 0;
	for(; i < hash_thread_count; ++i){
		data[i].start_point = start + (chunk * i);
		data[i].end_point = start + ((chunk * (i + 1)) + 1);
	}
}

int load_table(char *filename){
	htable = g_hash_table_new(g_str_hash, g_str_equal);
	gchar *contents;
	gsize *size;
	if(g_file_get_contents((gchar *)filename,&contents,NULL,NULL)){
		char *line;
		line = strtok (contents, "\n");
		int i = 0;
		while(line != NULL){
			g_hash_table_insert(htable, (gpointer)line, (gpointer)&found);
			line = strtok(NULL,"\n");
			++i;
		}
		printf("Loaded hash table with %d entrie(s)\n",i);
	}
	else{
		printf("Failed to read config file\n");
		return -1;
	}
	return 0;
}

char* get_hash_file(){
	return (char*) conf_data.hash_file;
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
	if(is_gpu_mode()){
		return getCudaDeviceCount();
	}
	
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