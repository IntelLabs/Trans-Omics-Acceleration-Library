/*************************************************************************
Intel PCL GBB - 
     Library of Optimized Genomics Building Blocks for Intel Processors
**************************************************************************
Copyright (c) 2018, Sanchit Misra, Parallel Computing Lab, 
                    Intel, Bangalore, India
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Sanchit Misra or Intel nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL SANCHIT MISRA OR INTEL BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

**************************************************************************
Author: Sanchit Misra <sanchit.misra@intel.com>
*************************************************************************/

#ifndef _PAIRWISESW_H
#define _PAIRWISESW_H

#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

typedef struct dnaSeqPair
{
    int32_t id;
    int16_t len1, len2;
    int32_t score;
}SeqPair;

typedef struct cycle_breakup
{
    int64_t sort1Ticks;
    int64_t setupTicks;
    int64_t swTicks;
    int64_t sort2Ticks;
}CycleBreakup;

#define MATRIX_MIN_CUTOFF 0
#define LOW_INIT_VALUE -63 // INT8_MIN/2 fixed
#define SORT_BLOCK_SIZE 16384
#define max(x, y) ((x)>(y)?(x):(y))
#define min(x, y) ((x)>(y)?(y):(x))

#if defined(__GNUC__) && !defined(__clang__)
#if defined(__i386__)
static inline unsigned long long __rdtsc(void)
{
        unsigned long long int x;
            __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
                return x;
}
#elif defined(__x86_64__)
static inline unsigned long long __rdtsc(void)
{
        unsigned hi, lo;
            __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
                return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif
#endif

class PairWiseSW
{
public:
    PairWiseSW(int32_t w_match, int32_t w_mismatch, int32_t w_open, int32_t w_extend);
	~PairWiseSW();
	CycleBreakup getTicks();
    void smithWatermanOnePair(SeqPair *p, uint8_t *seq1, uint8_t *seq2, int32_t *F, int32_t *H);
    void smithWatermanOnePairWrapper(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, int16_t maxSeqLen);

protected:
    void sortPairsLen(SeqPair *pairArray, int32_t count, SeqPair *tempArray, int16_t *hist, int16_t maxSeqLen);
    void sortPairsId(SeqPair *pairArray, int32_t first, int32_t count, SeqPair *tempArray);

#ifdef PERF_DEBUG
    long unsigned int mainLoopCount, setupLoopCount;
    int64_t mainLoopTicks;
#endif
    CycleBreakup ticksBreakup;
    int32_t w_match;
    int32_t w_mismatch;
    int32_t w_open;
    int32_t w_extend;
};
#endif

