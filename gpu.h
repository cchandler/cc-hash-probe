
int setupCUDA();
int test(unsigned int *num1, unsigned int *num2,unsigned int *valid);
unsigned int swapends(unsigned int v);

typedef struct{
	int num1; //Most significant digits
	int num2;
	int num3;
	int num4;
	
	unsigned int unpacked0;
	unsigned int unpacked1;
} cc_struct;