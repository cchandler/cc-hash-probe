#ifndef CPU_MODULE
#define CPU_MODULE

int cpu_sha1(unsigned int num1, unsigned int num2, unsigned int *hash);
unsigned int cpu_luhn_on_packed(unsigned long num);
unsigned long cpu_bit_pack_CC(unsigned long num);

#endif