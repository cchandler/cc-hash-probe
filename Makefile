CFLAGS=-g
#CUDA_LIB=-L/usr/local/cuda/lib64
CUDA_LIB=-L/usr/local/cuda/lib

GLIB=`pkg-config --cflags glib-2.0`
GLIB_FULL=`pkg-config --cflags --libs glib-2.0`

objects = util.o gpu.o cpu_func.o

all: $(objects) hash_probe

clean:
	rm *.o
	rm hash_probe
	rm -rf hash_probe.dSYM

util.o: util.h util.c
	g++ $(CFLAGS) -c util.c $(GLIB)

gpu.o: gpu.h gpu.cu
	nvcc $(CFLAGS) -c gpu.cu -m64 -arch sm_12 $(GLIB)

cpu_func.o: cpu_func.h cpu_func.c
	g++ $(CFLAGS) -c cpu_func.c

hash_probe: $(objects) hash_probe.c
	g++ $(CFLAGS) hash_probe.c -o hash_probe $(objects) $(CUDA_LIB) -lcuda -lcudart -DGPU $(GLIB_FULL)

cc:
	gcc cc.c -lpthread -lm

glib_test:
	g++ $(CFLAGS) glib_test.c  `pkg-config --cflags --libs glib-2.0`
