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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PairWiseSW.h"


PairWiseSW::PairWiseSW(int32_t w_match, int32_t w_mismatch, int32_t w_open, int32_t w_extend)
{
    this->w_match = w_match;
    this->w_mismatch = w_mismatch;
    this->w_open = w_open;
    this->w_extend = w_extend;
	this->ticksBreakup.sort1Ticks = 0;
	this->ticksBreakup.setupTicks = 0;
	this->ticksBreakup.swTicks    = 0;
	this->ticksBreakup.sort2Ticks = 0;
#ifdef PERF_DEBUG
	this->mainLoopTicks = 0;
    this->mainLoopCount = 0;
#endif
}

PairWiseSW::~PairWiseSW()
{
}

void PairWiseSW::smithWatermanOnePair(SeqPair *p, uint8_t *seq1, uint8_t *seq2, int32_t *F, int32_t *H)
{
    int32_t nrow = p->len1;
    int32_t ncol = p->len2;

    int32_t w_match = this->w_match;
    int32_t w_mismatch = this->w_mismatch;
    int32_t w_open = this->w_open;
    int32_t w_extend = this->w_extend;

    int32_t lowInitValue = LOW_INIT_VALUE;
    int i, j;
    for(i = 0; i <= ncol; i++)
    {
        F[i] = lowInitValue;
        H[i] = 0;
    }
    int32_t maxScore = 0;
    for(i = 1; i <= nrow; i++)
    {
        int32_t eij = lowInitValue;
        int32_t hdiag = 0; 
        int32_t hij;
        for(j = 1; j <=ncol; j++)
        {
            eij = max(eij + w_extend, H[j - 1] + w_open);
            int32_t fij = F[j] = max(F[j] + w_extend, H[j] + w_open);
            int32_t mij = hdiag + ((seq1[i - 1] == seq2[j - 1]) ? w_match: w_mismatch);
            hdiag = H[j];
            hij = MATRIX_MIN_CUTOFF;
            if(eij > hij)
            {
                hij = eij;
            }
            if(F[j] > hij)
            {
                hij = F[j];
            }
            if(mij > hij)
            {
                hij = mij;
            }
            H[j] = hij;
            if(maxScore < hij)
            {
                maxScore = hij;
            }
        }
    }
    p->score = maxScore;
    return;
}

void PairWiseSW::smithWatermanOnePairWrapper(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, int16_t maxSeqLen)
{
    int32_t i;

    int32_t *F = (int32_t *)_mm_malloc(maxSeqLen * sizeof(int32_t), 64);
    int32_t *H = (int32_t *)_mm_malloc(maxSeqLen * sizeof(int32_t), 64);

    for(i = 0; i < numPairs; i++)
    {
        smithWatermanOnePair(pairArray + i, seqBuf + 2 * i * maxSeqLen, seqBuf + (2 * i  + 1) * maxSeqLen, F, H);
    }

    _mm_free(F);
    _mm_free(H);

    return;
}


void PairWiseSW::sortPairsLen(SeqPair *pairArray, int32_t count, SeqPair *tempArray, int16_t *hist, int16_t maxSeqLen)
{

    int32_t i;
#if 0
    __m512i zero512 = _mm512_setzero_si512();
    for(i = 0; i <= maxSeqLen; i+=32)
    {
        _mm512_store_si512((__m512i *)(hist + i), zero512);
    }
#endif
    memset(hist, 0, maxSeqLen * sizeof(int16_t));

    
    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        hist[sp.len1]++;
    }

    int32_t prev = 0;
    int32_t cumulSum = 0;
    for(i = 0; i <= maxSeqLen; i++)
    {
        int32_t cur = hist[i];
        hist[i] = cumulSum;
        cumulSum += cur;
    }
    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        int32_t pos = hist[sp.len1];
        tempArray[pos] = sp;
        hist[sp.len1]++;
    }

    for(i = 0; i < count; i++)
        pairArray[i] = tempArray[i];
}

void PairWiseSW::sortPairsId(SeqPair *pairArray, int32_t first, int32_t count, SeqPair *tempArray)
{
    int32_t i;
    
    for(i = 0; i < count; i++)
    {
        SeqPair sp = pairArray[i];
        int32_t pos = sp.id - first;
        tempArray[pos] = sp;
    }

    for(i = 0; i < count; i++)
        pairArray[i] = tempArray[i];
}


CycleBreakup PairWiseSW::getTicks()
{
    return ticksBreakup;
}
