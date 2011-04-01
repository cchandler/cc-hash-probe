/*
 * Copyright (c) 2009 Steve Worley < m a t h g e e k@(my last name).com >
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */


/*
	This has been modified from the original to run on a general CPU
	for establishing workload equivalency.
*/

unsigned int cpu_popNextW(unsigned int *w, int &wIndex)
{
  unsigned int nextW=w[wIndex&15];
  int thisIndex=wIndex&15;
  w[thisIndex]^=w[(wIndex+16-3)&15]^w[(wIndex+16-8)&15]^w[(wIndex+16-14)&15];
  w[thisIndex]=  (w[thisIndex]<<1) | (w[thisIndex]>>31);
  ++wIndex;

  //  if (threadIdx.x==0) debugprint("pop %08x\n", nextW);
  return nextW;
}

unsigned int cpu_popFinalWs(unsigned int *w, int &wIndex)
{
  unsigned int nextW=w[wIndex&15];
  ++wIndex;
  return nextW;
}

/*
  SHA1 Hash
  Modified very of Steve Worley's
*/
int cpu_sha1(unsigned int num1, unsigned int num2, unsigned int *hash){
	
	unsigned int d_initVector[5];
	d_initVector[0] = 0x67452301;
	d_initVector[1] = 0xEFCDAB89;
	d_initVector[2] = 0x98BADCFE;
	d_initVector[3] = 0x10325476;
	d_initVector[4] = 0xC3D2E1F0;
	
	char lookup_table[10] = {48,49,50,51,52,53,54,55,56,57};
	int pos = 0;
	unsigned int digit = 0;
	unsigned int num_1a = 0;
	unsigned int num_2a = 0;
	unsigned int num_3a = 0;
	unsigned int num_4a = 0;
	
	#pragma unroll 999
	for(pos = 0; pos <= 3; ++pos) {
		digit = 0;
		digit = num2 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_1a = num_1a | lookup_table[digit];
		if(pos != 3) {num_1a = num_1a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 4; pos <= 7; ++pos) {
		digit = 0;
		digit = num2 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_2a = num_2a | lookup_table[digit];
		if(pos != 7) {num_2a = num_2a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 0; pos <= 3; ++pos) {
		digit = 0;
		digit = num1 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_3a = num_3a | lookup_table[digit];
		if(pos != 3) {num_3a = num_3a << 8;};
	}
	
	#pragma unroll 999
	for(pos = 4; pos <= 7; ++pos) {
		digit = 0;
		digit = num1 & (0xF << (pos * 4));
		digit = digit >> (pos * 4);
		
		num_4a = num_4a | lookup_table[digit];
		if(pos != 7) {num_4a = num_4a << 8;};
	}
	
	
	unsigned int w[80] = {'\0'};
	for (int i=0; i<80; i++) { w[i] = '\0'; };
	// w[0] = 1633837952; // 'abc' + 1 bit
	// num_1a = num_1a << 8;
	w[0] = num_1a;
	w[1] = num_2a;
	w[2] = num_3a;
	w[3] = num_4a;
	w[4] = (unsigned) 8 << 28;
	w[15] = 128;
	
	int wIndex=0;
	
	
	unsigned int a = d_initVector[0];
	unsigned int b = d_initVector[1];
    unsigned int c = d_initVector[2];
    unsigned int d = d_initVector[3];
    unsigned int e = d_initVector[4];
	
	#pragma unroll 999
	for (int i=0; i<20; ++i) {
	  unsigned int thisW=cpu_popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  //    unsigned int f= (b&c)|((~b)&d);
	  unsigned int f= d ^ (b & (c^d)); // alternate computation of above
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x5A827999+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	#pragma unroll 999
	for (int i=20; i<40; ++i) {
	  unsigned int thisW=cpu_popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x6ED9EBA1+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}

	#pragma unroll 999
	for (int i=40; i<60; ++i) {
	  unsigned int thisW=cpu_popNextW(w, wIndex);
	  // unsigned int thisW=w[i];
	  //    unsigned int f= (b&c) | (b&d) | (c&d);
	  unsigned int f= (b&c) | (d & (b|c)); // alternate computation of above
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0x8F1BBCDC+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}

	#pragma unroll 999
	for (int i=60; i<64; ++i) {
	  unsigned int thisW=cpu_popNextW(w, wIndex);
	// unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0xCA62C1D6+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}


	#pragma unroll 999
	for (int i=64; i<80; ++i) {
	  unsigned int thisW=cpu_popFinalWs(w, wIndex); // simpler compute for final rounds
	  // unsigned int thisW=w[i];
	  unsigned int f= b^c^d;
	  unsigned int temp=((a<<5)|(a>>27))+f+e+0xCA62C1D6+thisW;
	  e=d;
	  d=c;
	  c=(b<<30)|(b>>2);
	  b=a;
	  a=temp;
	}
	
	hash[0] = a + d_initVector[0];
	hash[1] = b + d_initVector[1];
	hash[2] = c + d_initVector[2];
	hash[3] = d + d_initVector[3];
	hash[4] = e + d_initVector[4];
	
	return 0;
}