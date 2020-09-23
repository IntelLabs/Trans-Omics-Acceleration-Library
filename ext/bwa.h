/*************************************************************************************
MIT License

Copyright (c) 2020 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Sanchit Misra <sanchit.misra@intel.com>; Vasimuddin Md <vasimuddin.md@intel.com>
*****************************************************************************************/

#ifndef BWA_H_
#define BWA_H_

#include <stdint.h>
#include <zlib.h>
#include "bntseq.h"

#define BWA_IDX_BWT 0x1
#define BWA_IDX_BNS 0x2
#define BWA_IDX_PAC 0x4
#define BWA_IDX_ALL 0x7

#define BWA_CTL_SIZE 0x10000


typedef struct {
	int l_seq, id;
	char *name, *comment, *seq, *qual, *sam;
} bseq1_t;

extern int bwa_verbose;

bseq1_t *bseq_read_orig(int64_t chunk_size, int *n_, void *ks1_, void *ks2_, int64_t *s);

bseq1_t *bseq_read(int64_t chunk_size, int *n_, void *ks1_,
				   void *ks2_, FILE* fpp, int len,
				   int64_t *sz);

bseq1_t *bseq_read_one_fasta_file(int64_t chunk_size, int *n_, gzFile fp, int64_t *s);

void bseq_destroy(bseq1_t *s);

#endif
