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
#include <stdio.h>
#include <string.h>

#include <glib.h>

GHashTable *htable;
char found_marker[] = "found";
int found = 1; //We're just testing for presence

enum work_mode {
	cpu_mode,
	gpu_mode
};

typedef struct
{
	gchar *hash_file;
	unsigned long cc_start_point;
	unsigned long cc_end_point;
	work_mode mode;
	unsigned int blocksize;
	unsigned int threadsize;
	int hashthreads;
	int scanthreads;
} Config;

Config conf_data;

int load_table(gchar *filename){
	htable = g_hash_table_new(g_str_hash, g_str_equal);
	gchar *contents;
	gsize *size;
	if(g_file_get_contents(filename,&contents,NULL,NULL)){
		// printf("Success!\n");
		char *line;
		line = strtok (contents, "\n");
		while(line != NULL){
			// printf("Line: %s\n",line);
			g_hash_table_insert(htable, (gpointer)line, (gpointer)&found);
			line = strtok(NULL,"\n");
		}
	}
	else{
		printf("Failed to read config file\n");
	}
	return 0;
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
	conf_data.cc_end_point = g_key_file_get_uint64(conf_file, "cc-hash-probe", "cc_start_point", NULL);
	conf_data.blocksize = g_key_file_get_integer(conf_file, "cc-hash-probe", "blocksize", NULL);
	conf_data.threadsize = g_key_file_get_integer(conf_file, "cc-hash-probe", "threadsize", NULL);
	conf_data.hashthreads = g_key_file_get_integer(conf_file, "cc-hash-probe", "hashthreads", NULL);
	conf_data.scanthreads = g_key_file_get_integer(conf_file, "cc-hash-probe", "scanthreads", NULL);
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
	printf("-----End Config\n");
}

int main(){	
	srand(time(NULL));
	
	load_config();
	print_config();
	load_table(conf_data.hash_file);
	
	
	int * intval = (int*)g_hash_table_lookup(htable, (gpointer)"68bfb396f35af3876fc509665b3dc23a0930aab1");
	printf("Found: %d\n",*intval);
	
	intval = (int*)g_hash_table_lookup(htable, (gpointer)"3186631ca5f40d4a1b9781f0ca5326b8206a8967");
	printf("Found: %d\n",*intval);
	
	intval = (int*)g_hash_table_lookup(htable, (gpointer)"68bfb396f35af3876fc509665b3dc23a0930aab2");
	if(!intval){
		printf("Nothing found (as expected)\n");
	}
	
	
	time_t start = time(NULL);
	// for(i = 0; i < 60000000; i++){
	// 	buffer = (char*)malloc(sizeof(char) * 4);
	// 	rand_val = rand() % 500;
	// 	sprintf(buffer, "%d", rand_val);
	// 	// printf("Searing: %s\tFound: %s\n",buffer,(char *)g_hash_table_lookup(htable, (gpointer)buffer));
	// 	free(buffer);
	// }
	time_t end = time(NULL);
	
	printf("Time: %ld\n", end-start);
	
	return 0;
}