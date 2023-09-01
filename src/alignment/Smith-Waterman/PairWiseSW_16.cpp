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
#include "PairWiseSW_16.h"

#ifdef IACA_ANALYSIS
#include "iacaMarks.h"
#endif

#ifdef VTUNE_ANALYSIS
#include <ittnotify.h>
#endif

#define DUMMY1 'B'
#define DUMMY2 'D'

#if defined(__AVX512BW__)

#define SIMD_WIDTH 32
#define _MM_INT_TYPE __m512i 
#define _MM_SET1_EPI16 _mm512_set1_epi16
#define _MM_SETZERO _mm512_setzero_si512
#define _MM_STORE _mm512_store_si512
#define _MM_LOAD _mm512_load_si512

#define MAIN_CODE(s1, s2, h00, h11, e11, f11, f21, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec) \
            { \
            __mmask32 cmp11 = _mm512_cmpeq_epi16_mask(s1, s2); \
            _MM_INT_TYPE sbt11 = _mm512_mask_blend_epi16(cmp11, mismatchVec, matchVec); \
            _MM_INT_TYPE m11 = _mm512_add_epi16(h00, sbt11); \
            h11 = _mm512_max_epi16(m11, e11); \
            h11 = _mm512_max_epi16(h11, f11); \
            h11 = _mm512_max_epi16(h11, zeroVec); \
            maxScoreVec = _mm512_max_epi16(maxScoreVec, h11); \
            _MM_INT_TYPE tempVec = _mm512_add_epi16(h11, gapOpenVec); \
            e11 = _mm512_add_epi16(e11, gapExtendVec); \
            e11 = _mm512_max_epi16(tempVec, e11); \
            f21 = _mm512_add_epi16(f11, gapExtendVec); \
            f21 = _mm512_max_epi16(f21, tempVec); \
            }

#elif defined(__AVX2__)

#define SIMD_WIDTH 16
#define _MM_INT_TYPE __m256i 
#define _MM_SET1_EPI16 _mm256_set1_epi16
#define _MM_SETZERO _mm256_setzero_si256
#define _MM_STORE _mm256_store_si256
#define _MM_LOAD _mm256_load_si256

#define MAIN_CODE(s1, s2, h00, h11, e11, f11, f21, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec) \
            { \
            _MM_INT_TYPE cmp11 = _mm256_cmpeq_epi16(s1, s2); \
            _MM_INT_TYPE sbt11 = _mm256_blendv_epi8(mismatchVec, matchVec, cmp11); \
            _MM_INT_TYPE m11 = _mm256_add_epi16(h00, sbt11); \
            h11 = _mm256_max_epi16(m11, e11); \
            h11 = _mm256_max_epi16(h11, f11); \
            h11 = _mm256_max_epi16(h11, zeroVec); \
            maxScoreVec = _mm256_max_epi16(maxScoreVec, h11); \
            _MM_INT_TYPE tempVec = _mm256_add_epi16(h11, gapOpenVec); \
            e11 = _mm256_add_epi16(e11, gapExtendVec); \
            e11 = _mm256_max_epi16(tempVec, e11); \
            f21 = _mm256_add_epi16(f11, gapExtendVec); \
            f21 = _mm256_max_epi16(f21, tempVec); \
            }

#else

#define SIMD_WIDTH 16
#define _MM_INT_TYPE __m128i
#define _MM_SET1_EPI16 _mm_set1_epi8
#define _MM_SETZERO _mm_setzero_si128
#define _MM_STORE _mm_store_si128
#define _MM_LOAD _mm_load_si128

#define MAIN_CODE(s1, s2, h00, h11, e11, f11, f21, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec) \
    { \
            printf("sse vectorization is not supported. Try using avx2/512. Exiting..\n");  \
            exit(0); \
    }

#endif


PairWiseSW_16::PairWiseSW_16(int32_t w_match,
                             int32_t w_mismatch,
                             int32_t w_open,
                             int32_t w_extend)
                             :PairWiseSW(w_match,
                                         w_mismatch,
                                         w_open,
                                         w_extend)
{
}

PairWiseSW_16::~PairWiseSW_16()
{
}

void PairWiseSW_16::getScores(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, uint16_t numThreads)
{
    int i;
    int64_t startTick, endTick;
    F_ = (int16_t *)_mm_malloc(MAX_SEQ_LEN_16 * SIMD_WIDTH * numThreads * sizeof(int16_t), 64);
    H_ = (int16_t *)_mm_malloc(MAX_SEQ_LEN_16 * SIMD_WIDTH * numThreads * sizeof(int16_t), 64);
    startTick = __rdtsc();
    smithWatermanBatchWrapper(pairArray, seqBuf, numPairs, numThreads);
    endTick = __rdtsc();
    _mm_free(F_);
    _mm_free(H_);
}

void PairWiseSW_16::smithWatermanInterTaskVectorized(uint16_t seq1SoA[],
                                                     uint16_t seq2SoA[],
                                                     int16_t nrow,
                                                     int16_t ncol,
                                                     SeqPair *p,
                                                     uint16_t tid)
{
    //printf("nrow = %d, ncol = %d\n", nrow, ncol);
    _MM_INT_TYPE matchVec = _MM_SET1_EPI16(this->w_match);
    _MM_INT_TYPE mismatchVec = _MM_SET1_EPI16(this->w_mismatch);
    _MM_INT_TYPE gapOpenVec = _MM_SET1_EPI16(this->w_open);
    _MM_INT_TYPE gapExtendVec = _MM_SET1_EPI16(this->w_extend);

    int16_t *F = F_ + tid * SIMD_WIDTH * MAX_SEQ_LEN_16;
    int16_t *H = H_ + tid * SIMD_WIDTH * MAX_SEQ_LEN_16;

    int16_t lowInitValue = LOW_INIT_VALUE;
    int16_t i, j;
    _MM_INT_TYPE lowInitValueVec = _MM_SET1_EPI16(lowInitValue);
    _MM_INT_TYPE zeroVec = _MM_SETZERO();

    for(j = 0; j <= ncol; j++)
    {
        _MM_STORE((_MM_INT_TYPE *)(F + j * SIMD_WIDTH), gapOpenVec);
        _MM_STORE((_MM_INT_TYPE *)(H + j * SIMD_WIDTH), zeroVec);
    }
    _MM_INT_TYPE maxScoreVec = zeroVec;

#ifdef PERF_DEBUG
    int64_t startTick, endTick;
    startTick = __rdtsc();
#endif
    for(i = 0; i < nrow; i+=4)
    {
        _MM_INT_TYPE e11, e21, e31, e41;
        e11 = e21 = e31 = e41 = gapOpenVec;
        _MM_INT_TYPE h00, h10, h20, h30, h11, h21, h31, h41;
        _MM_INT_TYPE s10 = _MM_LOAD((_MM_INT_TYPE *)(seq1SoA + (i + 0) * SIMD_WIDTH));
        _MM_INT_TYPE s11 = _MM_LOAD((_MM_INT_TYPE *)(seq1SoA + (i + 1) * SIMD_WIDTH));
        _MM_INT_TYPE s12 = _MM_LOAD((_MM_INT_TYPE *)(seq1SoA + (i + 2) * SIMD_WIDTH));
        _MM_INT_TYPE s13 = _MM_LOAD((_MM_INT_TYPE *)(seq1SoA + (i + 3) * SIMD_WIDTH));

        h00 = h10 = h20 = h30 = zeroVec;

        #pragma unroll(4)
        for(j = 0; j < ncol; j++)
        {
#ifdef IACA_ANALYSIS
            IACA_START
#endif
#ifdef PERF_DEBUG
            mainLoopCount++;
#endif
            _MM_INT_TYPE f11, f21, f31, f41, f51;
            f11 = _MM_LOAD((_MM_INT_TYPE *)(F + j * SIMD_WIDTH));
            _MM_INT_TYPE s2 = _MM_LOAD((_MM_INT_TYPE *)(seq2SoA + j * SIMD_WIDTH));

            MAIN_CODE(s10, s2, h00, h11, e11, f11, f21, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec);
            MAIN_CODE(s11, s2, h10, h21, e21, f21, f31, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec);
            MAIN_CODE(s12, s2, h20, h31, e31, f31, f41, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec);
            MAIN_CODE(s13, s2, h30, h41, e41, f41, f51, zeroVec, maxScoreVec, gapOpenVec, gapExtendVec);

            h00 = _MM_LOAD((_MM_INT_TYPE *)(H + j * SIMD_WIDTH));
            h10 = h11;
            h20 = h21;
            h30 = h31;
            _MM_STORE((_MM_INT_TYPE *)(F + j * SIMD_WIDTH), f51);
            _MM_STORE((_MM_INT_TYPE *)(H + j * SIMD_WIDTH), h41);
        }
#ifdef IACA_ANALYSIS
            IACA_END
#endif

    }
#ifdef PERF_DEBUG
    endTick = __rdtsc();
    mainLoopTicks += (endTick - startTick);
#endif

    int16_t maxScore[SIMD_WIDTH]  __attribute((aligned(64)));
    _MM_STORE((_MM_INT_TYPE *)maxScore, maxScoreVec);

    for(i = 0; i < SIMD_WIDTH; i++)
    {
        p[i].score = maxScore[i];
        //printf("score = %d\n", p[i].score);
    }
    return;
}

void PairWiseSW_16::smithWatermanBatchWrapper(SeqPair *pairArray, uint8_t *seqBuf, int32_t numPairs, uint16_t numThreads)
{
    int64_t st1, st2, st3, st4, st5;
    st1 = __rdtsc();
    uint16_t *seq1SoA = (uint16_t *)_mm_malloc(MAX_SEQ_LEN_16 * SIMD_WIDTH * numThreads * sizeof(uint16_t), 64);
    uint16_t *seq2SoA = (uint16_t *)_mm_malloc(MAX_SEQ_LEN_16 * SIMD_WIDTH * numThreads * sizeof(uint16_t), 64);

    int32_t ii;
    int32_t roundNumPairs = ((numPairs + SIMD_WIDTH - 1)/SIMD_WIDTH ) * SIMD_WIDTH;
    for(ii = numPairs; ii < roundNumPairs; ii++)
    {
        pairArray[ii].id = ii;
        pairArray[ii].len1 = 0;
        pairArray[ii].len2 = 0;
    }

    st2 = __rdtsc();
#ifdef SORT_PAIRS
    // Sort the sequences according to decreasing order of lengths
    SeqPair *tempArray = (SeqPair *)_mm_malloc(SORT_BLOCK_SIZE * numThreads * sizeof(SeqPair), 64);
    int16_t *hist = (int16_t *)_mm_malloc((MAX_SEQ_LEN_16 + 32) * numThreads * sizeof(int16_t), 64);
#pragma omp parallel num_threads(numThreads)
    {
        int32_t tid = omp_get_thread_num();
        SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;
        int16_t *myHist = hist + tid * (MAX_SEQ_LEN_16 + 32);

#pragma omp for
        for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
        {
            int32_t first, last;
            first = ii;
            last  = ii + SORT_BLOCK_SIZE;
            if(last > roundNumPairs) last = roundNumPairs;
            sortPairsLen(pairArray + first, last - first, myTempArray, myHist, MAX_SEQ_LEN_16);
        }
    }
    _mm_free(hist);
#endif
    st3 = __rdtsc();
#ifdef VTUNE_ANALYSIS
    __itt_resume();
#endif
#pragma omp parallel num_threads(numThreads)
    {
        int64_t st = __rdtsc();
        int32_t i;
        uint16_t tid = omp_get_thread_num();
        uint16_t *mySeq1SoA = seq1SoA + tid * MAX_SEQ_LEN_16 * SIMD_WIDTH;
        uint16_t *mySeq2SoA = seq2SoA + tid * MAX_SEQ_LEN_16 * SIMD_WIDTH;
        uint8_t *seq1;
        uint8_t *seq2;

#pragma omp for schedule(dynamic, 2)
        for(i = 0; i < numPairs; i+=SIMD_WIDTH)
        {
            int32_t j, k;
            int16_t maxLen1 = 0;
            int16_t maxLen2 = 0;
            int16_t minLen1 = MAX_SEQ_LEN_16 + 1;
            int16_t minLen2 = MAX_SEQ_LEN_16 + 1;
            for(j = 0; j < SIMD_WIDTH; j++)
            {
                SeqPair sp = pairArray[i + j];
                seq1 = seqBuf + 2 * (int64_t)sp.id * MAX_SEQ_LEN_16;
                for(k = 0; k < sp.len1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH + j] = seq1[k];
                }
                if(maxLen1 < sp.len1) maxLen1 = sp.len1;
                if(minLen1 > sp.len1) minLen1 = sp.len1;
            }
            maxLen1 = ((maxLen1 + 3) / 4) * 4;
            for(j = 0; j < SIMD_WIDTH; j++)
            {
                SeqPair sp = pairArray[i + j];
                for(k = sp.len1; k < maxLen1; k++)
                {
                    mySeq1SoA[k * SIMD_WIDTH + j] = DUMMY1;
                }
            }
            for(j = 0; j < SIMD_WIDTH; j++)
            {
                SeqPair sp = pairArray[i + j];
                seq2 = seqBuf + (2 * (int64_t)sp.id + 1) * MAX_SEQ_LEN_16;
                for(k = 0; k < sp.len2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH + j] = seq2[k];
                }
                if(maxLen2 < sp.len2) maxLen2 = sp.len2;
                if(minLen2 > sp.len2) minLen2 = sp.len2;
            }
            maxLen2 = ((maxLen2 + 3) / 4) * 4;
            for(j = 0; j < SIMD_WIDTH; j++)
            {
                SeqPair sp = pairArray[i + j];
                for(k = sp.len2; k < maxLen2; k++)
                {
                    mySeq2SoA[k * SIMD_WIDTH + j] = DUMMY2;
                }
            }
                smithWatermanInterTaskVectorized(mySeq1SoA, mySeq2SoA, maxLen1, maxLen2, pairArray + i, tid);
        }
        int64_t et = __rdtsc();
        //printf("%d] %ld ticks\n", tid, et - st);
    }
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
    st4 = __rdtsc();
#ifdef SORT_PAIRS
    // Sort the sequences according to increasing order of id
#pragma omp parallel num_threads(numThreads)
    {
        int32_t tid = omp_get_thread_num();
        SeqPair *myTempArray = tempArray + tid * SORT_BLOCK_SIZE;

#pragma omp for
        for(ii = 0; ii < roundNumPairs; ii+=SORT_BLOCK_SIZE)
        {
            int32_t first, last;
            first = ii;
            last  = ii + SORT_BLOCK_SIZE;
            if(last > roundNumPairs) last = roundNumPairs;
            sortPairsId(pairArray + first, first, last - first, myTempArray);
        }
    }
    _mm_free(tempArray);
#endif
    st5 = __rdtsc();
    ticksBreakup.setupTicks = st2 - st1;
    ticksBreakup.sort1Ticks = st3 - st2;
    ticksBreakup.swTicks = st4 - st3;
    ticksBreakup.sort2Ticks = st5 - st4;
    _mm_free(seq1SoA);
    _mm_free(seq2SoA);
    return;
}

