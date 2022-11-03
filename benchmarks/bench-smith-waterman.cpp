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
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "PairWiseSW_32.h"
#include "PairWiseSW_16.h"
#include "PairWiseSW_8.h"

#ifdef VTUNE_ANALYSIS
#include <ittnotify.h> 
#endif

#define MAX_LINE_LEN 10240
#define MAX_SEQ_BUF_SIZE 131072000000L
#define DEFAULT_MATCH 1
#define DEFAULT_MISMATCH -1
#define DEFAULT_OPEN 1
#define DEFAULT_EXTEND 1
#define DEFAULT_PRECISION 32
#define OUTPUT 1

int32_t w_match, w_mismatch, w_open, w_extend;
int32_t precision;
char *pairFileName;
int64_t SW_cells = 0;

int64_t getFreq()
{
    int64_t startTick, endTick;
    startTick = __rdtsc();
    sleep(1);
    endTick = __rdtsc();
    return (endTick - startTick);
}

void parseCmdLine(int argc, char *argv[])
{
    int i;
    w_match = DEFAULT_MATCH;
    w_mismatch = DEFAULT_MISMATCH;
    w_open = DEFAULT_OPEN;
    w_extend = DEFAULT_EXTEND;
    precision = DEFAULT_PRECISION;

    int pairFlag = 0;
    for(i = 1; i < argc; i+=2)
    {
        if(strcmp(argv[i], "-match") == 0)
        {
            w_match = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-mismatch") == 0)
        {
            w_mismatch = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-gapo") == 0)
        {
            w_open = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-gape") == 0)
        {
            w_extend = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-precision") == 0)
        {
            precision = atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-pairs") == 0)
        {
            pairFileName = argv[i + 1];
            pairFlag = 1;
        }
    }
    if(pairFlag == 0)
    {
        printf("ERROR! pairFileName not specified.\n");
        exit(0);
    }
}

int loadPairs(SeqPair *seqPairArray, uint8_t *seqBuf, int16_t maxSeqLen, int64_t maxNumPairs)
{
    FILE *pairFile = fopen(pairFileName, "r");

    if(pairFile == NULL)
    {
        fprintf(stderr, "Could not open file: %s\n", pairFileName);
        exit(0);
    }

    int64_t numPairs = 0;
    while(numPairs < maxNumPairs)
    {
        if(!fgets((char *)(seqBuf + numPairs * 2 * maxSeqLen), MAX_LINE_LEN, pairFile))
        {
            break;
        }
        if(!fgets((char *)(seqBuf + (numPairs * 2 + 1) * maxSeqLen), MAX_LINE_LEN, pairFile))
        {
            printf("ERROR! Odd number of sequences in %s\n", pairFileName);
            break;
        }

        SeqPair sp;
        sp.id = numPairs;
        sp.len1 = strnlen((char *)(seqBuf + numPairs * 2 * maxSeqLen), MAX_LINE_LEN) - 1;
        sp.len2 = strnlen((char *)(seqBuf + (numPairs * 2 + 1) * maxSeqLen), MAX_LINE_LEN) - 1;
        //printf("len1 = %d, len2 = %d\n", sp.len1, sp.len2);
        //printf("%s\n", sp.seq1);
        sp.score = 0;
        seqPairArray[numPairs] = sp;
        numPairs++;
        SW_cells += (sp.len1 * sp.len2);
    }
    if(numPairs == maxNumPairs)
    {
        printf("Reached max limit of number of pairs that can be processed."
                "\nPotentially there are few more pairs left to be processed\n");
    }
    fclose(pairFile);
    return numPairs;
}


int main(int argc, char *argv[])
{
#ifdef VTUNE_ANALYSIS
    __itt_pause();
#endif
    parseCmdLine(argc, argv);

    int32_t numThreads = 1;
#pragma omp parallel
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        if(tid == (nt - 1))
        {
            numThreads = nt;
        }
    }
#ifdef PERF_DEBUG
    numThreads = 1;
    printf("PERF_DEBUG is ON. Can only use 1 thread.\n");
#endif
    
    int64_t freq = getFreq();
    CycleBreakup ticks;
    int16_t maxSeqLen;
    switch(precision)
    {
        case 32:
            maxSeqLen = MAX_SEQ_LEN_32;
            break;
        case 16:
            maxSeqLen = MAX_SEQ_LEN_16;
            break;
        case  8:
            maxSeqLen = MAX_SEQ_LEN_8;
            break;
    }
    int64_t maxNumPairs = MAX_SEQ_BUF_SIZE / (maxSeqLen * 2);
     
    SeqPair *seqPairArray = (SeqPair *)_mm_malloc(maxNumPairs * sizeof(SeqPair), 64);
    uint8_t *seqBuf = (uint8_t *)_mm_malloc((maxSeqLen * 2 * maxNumPairs) * sizeof(int8_t), 64);

    int64_t numPairs = loadPairs(seqPairArray, seqBuf, maxSeqLen, maxNumPairs);

    if(precision == 32)
    {
        PairWiseSW_32 *pwsw32 = new PairWiseSW_32(w_match, w_mismatch, w_open, w_extend);
        pwsw32->getScores(seqPairArray, seqBuf, numPairs, numThreads);
        ticks = pwsw32->getTicks();
    }
    else if(precision == 16)
    {
        PairWiseSW_16 *pwsw16 = new PairWiseSW_16(w_match, w_mismatch, w_open, w_extend);
        pwsw16->getScores(seqPairArray, seqBuf, numPairs, numThreads);
        ticks = pwsw16->getTicks();
    }
    else if(precision == 8)
    {
        PairWiseSW_8 *pwsw8 = new PairWiseSW_8(w_match, w_mismatch, w_open, w_extend);
        pwsw8->getScores(seqPairArray, seqBuf, numPairs, numThreads);
        ticks = pwsw8->getTicks();
    }
    else
    {
        printf("Error! precision value of %d not supported.\n", precision);
    }
#ifdef OUTPUT
    int i;
    for(i = 0; i < numPairs; i++)
    {
        printf("%d\n", seqPairArray[i].score);
    }
#endif
    //printf("oneCount = %ld, totalCount = %ld\n", oneCount, totalCount);
    int64_t totalTicks = ticks.sort1Ticks + ticks.setupTicks + ticks.swTicks + ticks.sort2Ticks;
    printf("cost breakup: %ld, %ld, %ld, %ld, %ld\n",
            ticks.sort1Ticks, ticks.setupTicks, ticks.swTicks, ticks.sort2Ticks,
            totalTicks);
#ifdef PERF_DEBUG
    printf("mainLoopTicks = %ld, mainLoopCount = %lu\n", mainLoopTicks, mainLoopCount);
#endif
    printf("SW cycles = %ld\n", totalTicks);
    printf("SW cells  = %ld\n", SW_cells);
    printf("freq  = %ld\n", freq);
    printf("SW GCUPS  = %lf\n", 1.0 * SW_cells * freq / totalTicks / 1E9);
    
}
